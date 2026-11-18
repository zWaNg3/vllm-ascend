# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Upstream-style layerwise expert reload for fault-tolerance scale-down.

This is the Ascend port of ``vllm/v1/worker/sentinel/eplb_redistribute.py``:
stream the checkpoint through ``get_all_weights``, filter whole touched MoE
layers, buffer the loads through ``initialize_layerwise_reload``, write them
with ``model.load_weights``, and run the quantization post-processing through
``finalize_layerwise_reload``.

It exists as the reference/baseline for the selective Ascend reload in
``eplb_redistribute.py``: it reads the whole checkpoint and reloads every
expert of each touched layer, but it reuses the model's own load path instead
of the per-slot transforms. Per-slot list storage (MRV2 + EPLB) is copied
back into the existing slot tensors so captured graphs keep their addresses.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from inspect import signature

import torch
from vllm.config import LoadConfig, ModelConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.v1.worker.sentinel.expert_reload import _placement_weight_loader

from vllm_ascend.worker.sentinel.ft_metrics import FtRecoveryMetrics

_BYTES_PER_MIB = 1024 * 1024


@contextmanager
def _phase(metrics: FtRecoveryMetrics | None, name: str) -> Iterator[None]:
    with metrics.phase(name) if metrics is not None else nullcontext():
        yield


def _mapped_name(model: torch.nn.Module, name: str) -> str | None:
    """Checkpoint name in the model's runtime namespace, or None if dropped."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    return name if mapper is None else mapper._map_name(name)


def _slot_lists(layer: torch.nn.Module) -> dict[str, list[torch.Tensor]]:
    """Per-slot tensor lists on the live layer (MRV2 + EPLB storage)."""
    return {
        name: value
        for name, value in vars(layer).items()
        if name.endswith("_list") and isinstance(value, list) and value and torch.is_tensor(value[0])
    }


def _restore_slot_lists_in_place(
    layer: torch.nn.Module,
    captured: dict[str, list[torch.Tensor]],
) -> None:
    """Copy the reprocessed slots back into the tensors captured graphs reference.

    ``process_weights_after_loading`` rebuilds the per-slot lists from the
    reloaded stacked parameters. Captured graphs still point at the original
    slot tensors, so copy into them and restore the original list objects.
    """
    for name, old_slots in captured.items():
        new_slots = getattr(layer, name, None)
        if new_slots is old_slots:
            raise RuntimeError(
                f"[FT] layerwise reload did not rebuild {name!r}; "
                "process_weights_after_loading did not run for this layer."
            )
        if not isinstance(new_slots, list) or len(new_slots) != len(old_slots):
            raise RuntimeError(
                f"[FT] layerwise reload could not restore {name!r}: expected "
                f"{len(old_slots)} slots, got {type(new_slots).__name__}."
            )
        for old, new in zip(old_slots, new_slots):
            if old.shape != new.shape or old.dtype != new.dtype:
                raise RuntimeError(
                    f"[FT] layerwise reload changed {name!r} slot layout: "
                    f"{tuple(old.shape)}/{old.dtype} -> {tuple(new.shape)}/{new.dtype}."
                )
            old.copy_(new)
        setattr(layer, name, old_slots)


def _local_physical_ids(
    layer: torch.nn.Module,
    layer_idx: int,
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
) -> dict[int, list[int]]:
    """Logical expert -> every physical slot of it owned by this rank."""
    local_ids: dict[int, list[int]] = {}
    for physical_id, logical_id in enumerate(physical_to_logical_map[layer_idx].cpu().tolist()):
        if logical_id >= 0 and layer._map_global_expert_id_to_local_expert_id(physical_id) >= 0:
            local_ids.setdefault(logical_id, []).append(physical_id)
    return local_ids


def reload_experts_layerwise(
    model: torch.nn.Module,
    model_config: ModelConfig,
    load_config: LoadConfig,
    reassignments: set[tuple[int, int]],
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
    metrics: FtRecoveryMetrics | None = None,
) -> int:
    """Reload every expert of each touched MoE layer through the model loader.

    Mirrors upstream's ``reload_experts_from_disk``: the checkpoint is streamed
    once, filtered by the touched layers' runtime prefixes, loaded through
    ``model.load_weights`` with expert ids fanned out to each recovered
    physical slot, then finalized with the quant method's post-processing.
    """
    if not reassignments:
        return 0

    moe_layers = list(model.moe_layers)
    routed_layers = [getattr(layer, "routed_experts", layer) for layer in moe_layers]
    layer_indices = sorted({layer_idx for layer_idx, _ in reassignments})
    layers = [routed_layers[layer_idx] for layer_idx in layer_indices]
    prefixes = tuple(f"{layer.layer_name}." for layer in layers)

    for layer_idx, layer in zip(layer_indices, layers):
        restore_params, restore_buffers = get_layerwise_info(layer).restore_metadata
        if not restore_params and not restore_buffers:
            raise RuntimeError(
                f"[FT] layerwise reload metadata was not recorded for MoE layer {layer_idx}; "
                "the model must be built through initialize_model()."
            )

    captured = {layer_idx: _slot_lists(layer) for layer_idx, layer in zip(layer_indices, layers)}
    num_physical = physical_to_logical_map.shape[1]

    loader = get_model_loader(load_config)
    if not hasattr(loader, "get_all_weights"):
        raise NotImplementedError(
            f"[FT] layerwise expert reload requires a loader with get_all_weights, got {type(loader).__name__}."
        )

    matched_layers: set[int] = set()

    def filtered_weights() -> Iterator[tuple[str, torch.Tensor]]:
        # get_all_weights streams the whole checkpoint; the filter runs after
        # each tensor has been read, which is the cost this baseline measures.
        for name, tensor in loader.get_all_weights(model_config, model):
            if metrics is not None:
                metrics.add_value("layerwise_scanned_tensors", 1)
                metrics.add_value("layerwise_scanned_mib", tensor.numel() * tensor.element_size() / _BYTES_PER_MIB)
            mapped = _mapped_name(model, name)
            if mapped is None:
                continue
            for layer_idx, prefix in zip(layer_indices, prefixes):
                if mapped.startswith(prefix):
                    matched_layers.add(layer_idx)
                    if metrics is not None:
                        metrics.add_value("layerwise_matched_tensors", 1)
                        metrics.add_value("layerwise_read_mib", tensor.numel() * tensor.element_size() / _BYTES_PER_MIB)
                    # load_weights maps the raw checkpoint name itself.
                    yield name, tensor
                    break

    with _phase(metrics, "layerwise_init"):
        for layer_idx, layer in zip(layer_indices, layers):
            initialize_layerwise_reload(layer, defer_processing=True)
            local_ids = _local_physical_ids(layer, layer_idx, physical_to_logical_map, num_logical)
            for param in get_layer_tensors(layer).values():
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None and "expert_id" in signature(weight_loader).parameters:
                    param.weight_loader = _placement_weight_loader(
                        weight_loader,
                        local_ids,
                        num_logical,
                        num_physical,
                    )

    with _phase(metrics, "layerwise_load"):
        model.load_weights(filtered_weights())

    unmatched = set(layer_indices) - matched_layers
    if unmatched:
        raise RuntimeError(f"[FT] MoE layers {sorted(unmatched)} had no matching checkpoint weights.")

    with _phase(metrics, "layerwise_finalize"):
        for layer_idx, layer in zip(layer_indices, layers):
            finalize_layerwise_reload(layer, model_config)
            _restore_slot_lists_in_place(layer, captured[layer_idx])

    if metrics is not None:
        metrics.add_value("layerwise_layers", len(layer_indices))
    logger.info(
        "[FT] Layerwise expert reload complete: %d layer(s), %d reassigned pair(s).",
        len(layer_indices),
        len(reassignments),
    )
    return len(reassignments)
