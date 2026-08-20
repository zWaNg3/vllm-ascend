# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-specific helpers for the fault-tolerance scale-down flow.

The expert redistribution and map rebuilding for the model runner V2 reuse the
helpers in ``vllm.v1.worker.sentinel.eplb_redistribute``; this module holds the
Ascend-specific MC2 rank mapping and the expert weight reload that writes the
checkpoint tensors directly into the Ascend runtime layout.
"""

import torch
from vllm.logger import init_logger

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


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config,
    reload_set: set[tuple[int, int]],
) -> int:
    """Reload reassigned expert weights into the Ascend runtime layout.

    Supports both unquantized (BF16) and W8A8-quantized MoE layers under the
    model runner V2. The checkpoint tensors are converted to the runtime layout
    (transposed and NZ-cast; split into per-slot weight lists and per-channel
    quant scale/offset under EPLB) and written directly into
    ``AscendRoutedExperts`` instead of going through ``model.load_weights`` —
    whose ``process_weights_after_loading`` runs only once at load time and
    whose ``w13_weight``/``w2_weight`` are dropped in the EPLB list layout.

    Args:
        reload_set: ``{(moe_layer_idx, logical_expert), ...}`` from
            ``redistribute_expert_placement``.

    Returns:
        Number of (layer, expert) pairs reloaded.
    """
    if not reload_set:
        return 0

    import torch_npu
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    from vllm_ascend.quantization.methods.w8a8_dynamic import scale_from_float_to_int64
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    moe_layers = list(model.moe_layers)
    # layer_name.<expert>. -> (moe_layer_idx, logical_expert)
    prefixes = {f"{moe_layers[i].layer_name}.{e}.": (i, e) for i, e in reload_set}

    loader = DefaultModelLoader(vllm_config.load_config)
    # Do not apply EP weight filtering: every reloaded expert must be read.
    loader.local_expert_ids = None

    all_weights = loader.get_all_weights(vllm_config.model_config, model)
    saved_weights: dict[str, torch.Tensor] = {}
    matched: set[str] = set()
    for name, tensor in all_weights:
        for prefix in prefixes:
            if name.startswith(prefix):
                matched.add(prefix)
                saved_weights[name] = tensor
                break

    unmatched = [p for p in prefixes if p not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {sorted(unmatched)[:3]}."
        )

    count = 0
    for layer_idx, logical in reload_set:
        layer = moe_layers[layer_idx]
        routed = layer.routed_experts
        base = f"{layer.layer_name}.{logical}."
        slot = int(routed._expert_map[logical].item())

        w1 = saved_weights[f"{base}gate_proj.weight"]
        w3 = saved_weights[f"{base}up_proj.weight"]
        w2 = saved_weights[f"{base}down_proj.weight"]

        w13_list = getattr(routed, "w13_weight_list", None)
        is_list = isinstance(w13_list, list)
        device = w13_list[slot].device if is_list else routed.w13_weight[slot].device

        # Runtime layout: w13 = [hidden, 2*intermediate] (gate|up concatenated
        # along the output dim), w2 = [intermediate, hidden].
        w13 = torch.cat([w1, w3], dim=0).transpose(0, 1).contiguous().to(device)
        w2r = w2.contiguous().to(device)
        w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
        w2r = torch_npu.npu_format_cast(w2r, ACL_FORMAT_FRACTAL_NZ)

        if is_list:
            routed.w13_weight_list[slot].copy_(w13)
            routed.w2_weight_list[slot].copy_(w2r)
        else:
            routed.w13_weight[slot].copy_(w13)
            routed.w2_weight[slot].copy_(w2r)

        # W8A8 per-channel quant scale/offset (gate|up merged for w13).
        if hasattr(routed, "w13_weight_scale_fp32") or hasattr(routed, "w13_weight_scale_fp32_list"):
            s1 = saved_weights[f"{base}gate_proj.weight_scale"]
            s3 = saved_weights[f"{base}up_proj.weight_scale"]
            s2 = saved_weights[f"{base}down_proj.weight_scale"]
            o1 = saved_weights[f"{base}gate_proj.weight_offset"]
            o3 = saved_weights[f"{base}up_proj.weight_offset"]
            o2 = saved_weights[f"{base}down_proj.weight_offset"]

            w13_scale = torch.cat([s1, s3], dim=0).squeeze(-1).float().to(device)
            w2_scale = s2.squeeze(-1).float().to(device)
            w13_offset = torch.cat([o1, o3], dim=0).squeeze(-1).to(device)
            w2_offset = o2.squeeze(-1).to(device)

            if is_list:
                routed.w13_weight_scale_fp32_list[slot].copy_(w13_scale)
                routed.w2_weight_scale_list[slot].copy_(w2_scale)
            else:
                routed.w13_weight_scale_fp32[slot].copy_(w13_scale)
                routed.w2_weight_scale[slot].copy_(w2_scale)
            routed.w13_weight_offset.data[slot].copy_(w13_offset)
            routed.w2_weight_offset.data[slot].copy_(w2_offset)

            # FUSED_MC2 consumes int64 fused scales derived from the fp32 ones.
            if is_list:
                fused_w1 = getattr(routed, "fused_w1_scale_list", None)
                fused_w2 = getattr(routed, "fused_w2_scale_list", None)
            else:
                fused_w1 = getattr(routed, "fused_w1_scale", None)
                fused_w2 = getattr(routed, "fused_w2_scale", None)
            if fused_w1 is not None and fused_w2 is not None:
                f1 = scale_from_float_to_int64(w13_scale)
                f2 = scale_from_float_to_int64(w2_scale)
                if is_list:
                    routed.fused_w1_scale_list[slot].copy_(f1)
                    routed.fused_w2_scale_list[slot].copy_(f2)
                else:
                    routed.fused_w1_scale[slot].copy_(f1)
                    routed.fused_w2_scale[slot].copy_(f2)
        count += 1

    torch.accelerator.synchronize()
    logger.info("[FT] Expert weight reload complete: %d (layer, expert) pair(s).", count)
    return count
