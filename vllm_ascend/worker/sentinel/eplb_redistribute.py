# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helper functions for Ascend scale-down.

Mirrors ``vllm/v1/worker/sentinel/eplb_redistribute.py`` in PR #46370: all the
deterministic placement math and per-layer weight handling lives here as
module-level functions; the orchestration lives in
``WorkerSentinel._redistribute_experts``.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

logger = init_logger(__name__)


def compute_dead_ep_ranks(
    dead_dp_ranks: list[int],
    pcp_size: int,
    tp_size: int,
) -> set[int]:
    """Map dead DP ranks to dead EP ranks inside the Ascend MC2 group.

    The MC2 ("EP-like") group is laid out DP-major per PP stage::

        ep_rank = dp_rank * (pcp_size * tp_size) + pcp_rank * tp_size + tp_rank

    A dead DP rank therefore takes all of its ``pcp_size * tp_size`` EP slots
    out of service.
    """
    ep_per_dp = pcp_size * tp_size
    dead = set()
    for dp_rank in dead_dp_ranks:
        dead.update(range(dp_rank * ep_per_dp, (dp_rank + 1) * ep_per_dp))
    return dead


def global_placement(global_expert_map: torch.Tensor) -> torch.Tensor:
    """Invert the ``[ep_size, n_logical]`` logical->slot map into slot->logical.

    ``global_expert_map[rank][logical]`` holds the local physical slot of
    ``logical`` on ``rank`` (-1 if not hosted there). The inverse is
    ``placement[rank][slot] = logical``. The number of physical slots per rank
    is derived from the maximum slot id.
    """
    ep_size, n_logical = global_expert_map.shape
    num_local = int(global_expert_map.max().item()) + 1 if n_logical > 0 else 0
    placement = torch.full((ep_size, num_local), -1, dtype=torch.int32)
    for rank in range(ep_size):
        row = global_expert_map[rank]
        for logical in range(n_logical):
            slot = int(row[logical].item())
            if 0 <= slot < num_local:
                placement[rank, slot] = logical
    return placement


def redistribute_expert_placement(
    global_expert_map: torch.Tensor,
    dead_ep_ranks: set[int],
    ep_rank: int,
    tp_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Deterministically re-host dead ranks' experts on surviving ranks.

    The EP topology is unchanged (ranks are NOT rearranged, matching
    vllm-project/vllm PR #46370): every surviving rank keeps its EP rank id and
    its ``num_local`` physical slots. Dead ranks' slots are vacated; each
    logical expert whose only copy lived on a dead rank is moved into a free
    (redundant) slot on a surviving rank.

    Args:
        global_expert_map: ``[ep_size, n_logical]`` logical->slot map.
        dead_ep_ranks: EP ranks (MC2 group coordinates) that died.
        ep_rank: MC2 rank of the current worker.
        tp_size: tensor-parallel size used to pick a replica when a logical
            expert still has multiple surviving copies.

    Returns:
        ``(new_global_expert_map, log2phy, reassignments)`` where
        ``new_global_expert_map`` is the updated ``[ep_size, n_logical]`` map,
        ``log2phy`` is this rank's ``[n_logical]`` logical->global-physical map
        and ``reassignments`` is a ``[(local_slot, logical_expert)]`` list of
        every slot on this rank whose hosted expert changed (needs a weight
        reload).
    """
    placement = global_placement(global_expert_map)
    ep_size, num_local = placement.shape

    all_logicals = set()
    for rank in range(ep_size):
        for slot in range(num_local):
            logical = int(placement[rank, slot].item())
            if logical >= 0:
                all_logicals.add(logical)

    logical_to_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rank in range(ep_size):
        if rank in dead_ep_ranks:
            continue
        for slot in range(num_local):
            logical = int(placement[rank, slot].item())
            if logical >= 0:
                logical_to_slots[logical].append((rank, slot))

    orphaned = sorted(all_logicals - logical_to_slots.keys())

    # Keep the first copy of each logical expert on a survivor as "essential";
    # any extra copies free their slots for orphaned experts.
    free_slots: list[tuple[int, int]] = []
    for slots in logical_to_slots.values():
        if len(slots) >= 2:
            free_slots.extend(slots[1:])
    free_slots.sort()

    if len(free_slots) < len(orphaned):
        raise RuntimeError(
            f"[FT] Not enough redundant slots to scale down: "
            f"{len(free_slots)} free slots < {len(orphaned)} orphaned experts."
        )

    new_placement = placement.clone()
    for rank in dead_ep_ranks:
        new_placement[rank] = -1

    reassignments_all: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for logical, (rank, slot) in zip(orphaned, free_slots):
        new_placement[rank, slot] = logical
        reassignments_all[rank].append((slot, logical))

    new_global_expert_map = torch.full_like(global_expert_map, -1)
    for rank in range(ep_size):
        for slot in range(num_local):
            logical = int(new_placement[rank, slot].item())
            if logical >= 0:
                new_global_expert_map[rank, logical] = slot

    log2phy = generate_log2phy_map(new_global_expert_map, ep_rank, num_local, tp_size)
    return new_global_expert_map, log2phy, sorted(reassignments_all.get(ep_rank, []))


def generate_log2phy_map(
    global_expert_map: torch.Tensor,
    ep_rank: int,
    num_local: int,
    tp_size: int | None = None,
) -> torch.Tensor:
    """Build the logical->global-physical map for ``ep_rank``.

    The global physical expert id is ``rank * num_local + slot`` (the flattened
    ``[ep_size, num_local]`` weight layout). When a logical expert still has
    several surviving copies the replica is chosen deterministically, mirroring
    ``vllm_ascend.eplb.core.eplb_utils.generate_log2phy_map`` (but robust to a
    dead rank 0, whose all-``-1`` row would otherwise zero out ``valid_count``).
    """
    n_logical = global_expert_map.shape[1]
    logical_to_phy: dict[int, list[int]] = defaultdict(list)
    for rank in range(global_expert_map.shape[0]):
        row = global_expert_map[rank]
        for logical in range(n_logical):
            slot = int(row[logical].item())
            if slot >= 0:
                logical_to_phy[logical].append(rank * num_local + slot)

    log2phy = torch.full((n_logical,), -1, dtype=torch.int32)
    for logical, phy_ids in logical_to_phy.items():
        num_duplications = len(phy_ids)
        if tp_size is not None and tp_size > 1:
            tp_rank = ep_rank % tp_size
            dp_like_rank = ep_rank // tp_size
            replica_index = (tp_rank + dp_like_rank + logical) % num_duplications
        else:
            replica_index = ep_rank % num_duplications
        log2phy[logical] = phy_ids[replica_index]
    return log2phy


def rebuild_model_expert_maps(
    layer: "AscendMoERunner",
    new_global_expert_map: torch.Tensor,
    ep_rank: int,
    log2phy: torch.Tensor,
) -> None:
    """Apply the new placement to one MoE layer's maps, in place.

    Mirrors the upstream ``rebuild_model_expert_maps``. The Ascend dispatch map
    ``layer._expert_map`` (logical->slot, ``[num_logical]``) is updated in place;
    the upstream ``expert_map_manager._expert_map`` — which the weight loader
    indexes by (primary) physical expert id — is rebuilt as a ``[num_physical]``
    physical->local map so a later ``model.load_weights`` places the reloaded
    expert into its post-redistribution slot (redundant physical slots map to
    -1 and are skipped by the loader).
    """
    layer.global_expert_map.copy_(new_global_expert_map)
    local_row = new_global_expert_map[ep_rank]
    layer._expert_map.copy_(local_row)
    # physical->local map: index == logical for the primary experts, -1 for the
    # redundant slots (so the loader's _map_global_expert_id_to_local_expert_id
    # returns a valid slot for the primary and -1 for redundant copies).
    num_physical = int(layer.moe_config.num_experts)
    phys_map = torch.full((num_physical,), -1, dtype=torch.int32, device=local_row.device)
    n_logical = min(int(local_row.shape[0]), num_physical)
    phys_map[:n_logical].copy_(local_row[:n_logical])
    layer.routed_experts.expert_map_manager._expert_map = phys_map
    if layer.log2phy is not None:
        layer.log2phy.copy_(log2phy.to(layer.log2phy.device))


def reconfigure_moe(
    layer: "AscendMoERunner",
    num_logical_experts: int,
    new_num_physical: int,
    num_local: int,
) -> None:
    """Update a MoE layer's expert counts after scale-down.

    Mirrors the demo branch's ``reconfigure_moe``: after scale-down the number
    of physical experts shrinks to the surviving slots, so the per-layer counts
    (and the global redundant count) are refreshed. This keeps the MC2
    dispatch's ``moe_expert_num`` (= num_logical + global_redundant_expert_num)
    in agreement with elastic_info's ``num_physical_experts``.
    """
    new_redundant = new_num_physical - num_logical_experts
    layer.global_redundant_expert_num = new_redundant
    layer.moe_config.num_experts = new_num_physical
    layer.moe_config.num_local_experts = num_local
    layer.moe_config.num_logical_experts = num_logical_experts
    layer.moe_config.global_redundant_expert_num = new_redundant
    logger.info(
        "[FT] reconfigure_moe: num_logical=%d, new_num_physical=%d, new_redundant=%d, num_local=%d",
        num_logical_experts,
        new_num_physical,
        new_redundant,
        num_local,
    )


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config,
    layers: list["AscendMoERunner"],
    reload_set: dict[int, list[int]],
) -> int:
    """Reload expert weights from disk through the standard loader.

    Follows PR #46370's ``reload_experts_from_disk`` (quant-agnostic): the
    checkpoint weights of the reloaded ``(layer_id, logical_expert)`` pairs are
    read from disk and fed to ``model.load_weights``, which places them into the
    post-redistribution slots via the rebuilt ``expert_map_manager`` map.

    Ascend stores per-slot weights in the runtime (post
    ``process_weights_after_loading``) format while the loader writes the raw
    checkpoint format, so every affected layer's weight processing is re-run
    afterwards. For that re-run to be correct all local slots must carry raw
    data, therefore ``reload_set[layer_id]`` must list **all** local logical
    experts of the affected layer (not just the reassigned ones).

    Args:
        reload_set: ``layer_id -> [logical_expert, ...]`` to reload.

    Returns:
        Number of (layer, expert) pairs requested.
    """
    if not reload_set:
        return 0

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    loader = DefaultModelLoader(vllm_config.load_config)
    # Do not apply EP weight filtering: every reloaded expert must be read.
    loader.local_expert_ids = None

    # Checkpoint-name prefixes, e.g. "model.layers.0.mlp.experts.5.".
    prefixes: dict[str, tuple[int, int]] = {}
    for layer_id, logicals in reload_set.items():
        layer_name = layers[layer_id].layer_name
        for logical in logicals:
            prefixes[f"{layer_name}.{logical}."] = (layer_id, logical)

    all_weights = loader.get_all_weights(vllm_config.model_config, model)
    matched: set[str] = set()

    def filtered_iter():
        for name, tensor in all_weights:
            for prefix in prefixes:
                if name.startswith(prefix):
                    matched.add(prefix)
                    yield name, tensor
                    break

    logger.info("[FT] Reloading %d (layer, expert) pair(s) from disk.", len(prefixes))
    model.load_weights(filtered_iter())

    unmatched = [p for p in prefixes if p not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {sorted(unmatched)[:3]}."
        )

    # Re-run the layer weight processing to convert the raw checkpoint format
    # back to the runtime format.
    for layer_id in {lid for lid, _ in prefixes.values()}:
        _reprocess_layer(layers[layer_id])
    torch.accelerator.synchronize()

    logger.info("[FT] Expert weight reload complete: %d (layer, expert) pair(s).", len(prefixes))
    return len(prefixes)


def _reprocess_layer(layer: "AscendMoERunner") -> None:
    """Re-run a layer's ``process_weights_after_loading`` after a reload.

    Ascend keeps the per-slot expert weights in the runtime (post-processing)
    format, so reloading raw checkpoint weights requires re-applying the
    scheme's ``process_weights_after_loading``.
    """
    routed = layer.routed_experts
    quant_method = getattr(routed, "quant_method", None)
    if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
        quant_method.process_weights_after_loading(routed)
