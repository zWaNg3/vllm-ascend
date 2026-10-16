# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""UTs for the fault-tolerance scale-down expert weight reload.

The reload is CPU tensor algebra plus one weight-layout cast, so the
routed-experts layers are faked here: the tests pin down which runtime
attribute each scheme fills, at which shape and dtype, for both the EPLB
per-slot layout (``<attr>_list``) and the whole-tensor layout.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend import ascend_config
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, ASCEND_QUANTIZATION_METHOD, maybe_trans_nz
from vllm_ascend.worker.sentinel import eplb_redistribute as redistribute

H = 4  # hidden size
INTER = 4  # moe_intermediate_size per TP rank
SLOTS = 2  # experts re-hosted on this rank

# (scheme class, checkpoint sub-format) of every scheme the reload supports.
SCHEMES = [
    (AscendUnquantizedFusedMoEMethod, None),
    (AscendW8A8DynamicFusedMoEMethod, None),
    (AscendW4A8DynamicFusedMoEMethod, ASCEND_QUANTIZATION_METHOD),
]


@pytest.fixture(autouse=True)
def _nz_weight_layout_off(monkeypatch):
    """Pin ``weight_nz_mode`` to 0 so ``maybe_trans_nz`` keeps weights in ND.

    The reloaded tensors are then directly comparable to the checkpoint's, which
    is what the assertions below rely on. The policy-driven NZ path is covered
    separately by ``test_reload_follows_weight_nz_mode``.
    """

    def _config(nz_mode: int) -> SimpleNamespace:
        # Mirrors the fields ``_is_ascend_config_initialized`` requires.
        return SimpleNamespace(ascend_compilation_config=object(), eplb_config=object(), weight_nz_mode=nz_mode)

    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", _config(0))
    return _config


def _arange(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= dim
    return torch.arange(n, dtype=dtype).reshape(shape)


def _bits(scale: torch.Tensor) -> torch.Tensor:
    """fp32 -> int64 keeping the bit pattern, as ``_process_scale`` stores it.

    Deliberately a reimplementation: the expectation must not come from the
    function under test.
    """
    arr = scale.contiguous().numpy()
    arr.dtype = np.uint32
    return torch.from_numpy(arr.astype(np.int64))


def _checkpoint(scheme_cls: type, expert_id: int) -> dict[str, torch.Tensor]:
    """The checkpoint tensors of one expert, under msModelSlim naming."""
    bias = float(expert_id)  # keeps each expert's tensors distinguishable
    if scheme_cls is AscendUnquantizedFusedMoEMethod:
        return {
            "gate_up_proj.weight": _arange((2 * INTER, H), torch.bfloat16) + bias,
            "down_proj.weight": _arange((H, INTER), torch.bfloat16) + bias,
        }
    if scheme_cls is AscendW8A8DynamicFusedMoEMethod:
        return {
            "gate_up_proj.weight": _arange((2 * INTER, H), torch.int8),
            "gate_up_proj.weight_scale": _arange((2 * INTER, 1), torch.float32) + bias,
            "down_proj.weight": _arange((H, INTER), torch.int8),
            "down_proj.weight_scale": _arange((H, 1), torch.float32) + bias,
        }
    # W4A8 modelslim: two int4 packed per int8 along the output dim, so the
    # checkpoint rows are half the runtime width.
    return {
        "gate_proj.weight": _arange((INTER // 2, H), torch.int8),
        "up_proj.weight": _arange((INTER // 2, H), torch.int8),
        "down_proj.weight": _arange((H // 2, INTER), torch.int8),
        "gate_proj.weight_scale": _arange((INTER, 1), torch.float32) + bias,
        "up_proj.weight_scale": _arange((INTER, 1), torch.float32) + bias,
        "down_proj.weight_scale": _arange((H, 1), torch.float32) + bias,
        "gate_proj.scale_bias": _arange((INTER, 1), torch.float32) + bias,
        "up_proj.scale_bias": _arange((INTER, 1), torch.float32) + bias,
        # [out, 16 // tp_size]: the replica/group dim the runtime reduces away.
        "down_proj.scale_bias": _arange((H, 16), torch.float32) + bias,
    }


def _expected(checkpoint: dict[str, torch.Tensor], scheme_cls: type, w2_scale_dtype: torch.dtype):
    """Per-slot tensors the reload must produce, per ``process_weights_after_loading``."""
    if scheme_cls is AscendUnquantizedFusedMoEMethod:
        return {
            "w13_weight": checkpoint["gate_up_proj.weight"].transpose(0, 1).contiguous(),
            "w2_weight": checkpoint["down_proj.weight"].transpose(0, 1).contiguous(),
        }
    if scheme_cls is AscendW8A8DynamicFusedMoEMethod:
        return {
            "w13_weight": checkpoint["gate_up_proj.weight"].transpose(0, 1).contiguous(),
            "w2_weight": checkpoint["down_proj.weight"].transpose(0, 1).contiguous(),
            "w13_weight_scale_fp32": checkpoint["gate_up_proj.weight_scale"].reshape(-1).to(torch.float32),
            # The runtime slot's dtype wins over the checkpoint's fp32.
            "w2_weight_scale": checkpoint["down_proj.weight_scale"].reshape(-1).to(w2_scale_dtype),
        }
    gate_up_scale = torch.cat(
        [checkpoint["gate_proj.weight_scale"], checkpoint["up_proj.weight_scale"]], dim=0
    ).reshape(-1)
    gate_up_bias = torch.cat([checkpoint["gate_proj.scale_bias"], checkpoint["up_proj.scale_bias"]], dim=0).reshape(-1)
    return {
        # Packed dim: the runtime weight's 2 * INTER int4 are INTER int8.
        "w13_weight": torch.cat([checkpoint["gate_proj.weight"], checkpoint["up_proj.weight"]], dim=0)
        .transpose(0, 1)
        .contiguous(),
        "w2_weight": checkpoint["down_proj.weight"].transpose(0, 1).contiguous(),
        "w13_weight_scale": _bits(gate_up_scale),
        "w2_weight_scale": _bits(checkpoint["down_proj.weight_scale"].to(torch.float32).reshape(1, -1)),
        "w13_scale_bias": gate_up_bias,
        "w2_scale_bias": checkpoint["down_proj.scale_bias"].transpose(0, 1).sum(0),
    }


def _make_layer(
    scheme_cls: type,
    sub_format: str | None = None,
    *,
    per_slot: bool = True,
    w2_scale_dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    """A routed-experts layer stand-in holding ``SLOTS`` slots.

    ``per_slot`` picks between the EPLB per-slot lists and the whole-tensor
    layout; the two are mutually exclusive in production.
    """
    method = object.__new__(scheme_cls)  # skip the NPU-bound __init__
    if sub_format is not None:
        method.quant_method = sub_format
    layer = SimpleNamespace(moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_rank=0, tp_size=1)))
    if scheme_cls is AscendUnquantizedFusedMoEMethod:
        layer.quant_method = method
    else:
        # Quantized layers wrap the scheme in AscendFusedMoEMethod.
        layer.quant_method = SimpleNamespace(quant_method=method)

    if scheme_cls is AscendUnquantizedFusedMoEMethod:
        storage = {"w13_weight": ((H, 2 * INTER), torch.bfloat16), "w2_weight": ((INTER, H), torch.bfloat16)}
    elif scheme_cls is AscendW8A8DynamicFusedMoEMethod:
        storage = {
            "w13_weight": ((H, 2 * INTER), torch.int8),
            "w2_weight": ((INTER, H), torch.int8),
            "w13_weight_scale_fp32": ((2 * INTER,), torch.float32),
            "w2_weight_scale": ((H,), w2_scale_dtype),
        }
    else:
        storage = {
            "w13_weight": ((H, INTER), torch.int8),
            "w2_weight": ((INTER, H // 2), torch.int8),
            "w13_weight_scale": ((2 * INTER,), torch.int64),
            "w2_weight_scale": ((1, H), torch.int64),
            "w13_scale_bias": ((2 * INTER,), torch.float32),
            "w2_scale_bias": ((H,), torch.float32),
        }

    for attr, (shape, dtype) in storage.items():
        if per_slot:
            setattr(layer, f"{attr}_list", [torch.zeros(shape, dtype=dtype) for _ in range(SLOTS)])
        else:
            setattr(layer, attr, torch.zeros((SLOTS, *shape), dtype=dtype))
    return layer


def _run_reload(layers: list, scheme_cls: type, sub_format: str | None = None):
    """Reload every slot of every layer from a fresh per-expert checkpoint."""
    local_slots, buckets = {}, {}
    for layer_idx in range(len(layers)):
        for expert in range(SLOTS):
            pair = (layer_idx, expert)
            local_slots[pair] = expert
            buckets[pair] = _checkpoint(scheme_cls, expert)
    reloaded = redistribute._reload_local_slots(layers, local_slots, buckets)
    return reloaded, local_slots, buckets


def test_every_scheme_declares_a_layout_cast():
    """A scheme the reload supports must say how its weights are laid out."""
    assert set(redistribute._SCHEME_RECIPES) == set(SCHEMES)
    assert set(redistribute._WEIGHT_LAYOUT_CAST) == {scheme_cls for scheme_cls, _ in SCHEMES}
    # W8A8's grouped_matmul_swiglu_quant only takes NZ; the others follow the
    # hardware policy.
    assert redistribute._WEIGHT_LAYOUT_CAST[AscendW8A8DynamicFusedMoEMethod] is redistribute._to_fractal_nz
    assert redistribute._WEIGHT_LAYOUT_CAST[AscendW4A8DynamicFusedMoEMethod] is maybe_trans_nz
    assert redistribute._WEIGHT_LAYOUT_CAST[AscendUnquantizedFusedMoEMethod] is maybe_trans_nz


def test_every_weight_and_scale_suffix_is_collected():
    """The wanted-suffix set must be a superset of what the builders read."""
    assert (
        frozenset(
            {
                "gate_up_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
                "gate_up_proj.weight_scale",
                "gate_proj.weight_scale",
                "up_proj.weight_scale",
                "down_proj.weight_scale",
                "gate_up_proj.scale_bias",
                "gate_proj.scale_bias",
                "up_proj.scale_bias",
                "down_proj.scale_bias",
            }
        )
        == redistribute._WANTED_SUFFIXES
    )


@pytest.mark.parametrize("scheme_cls,sub_format", SCHEMES)
def test_reload_fills_every_slot_of_every_scheme(scheme_cls, sub_format):
    layer = _make_layer(scheme_cls, sub_format)
    reloaded, local_slots, buckets = _run_reload([layer], scheme_cls, sub_format)

    assert reloaded == len(local_slots)
    for (_, expert), slot in local_slots.items():
        expected = _expected(buckets[(0, expert)], scheme_cls, torch.float32)
        # No scheme may silently gain or lose a reloaded tensor.
        assert set(expected) == set(redistribute._SCHEME_RECIPES[(scheme_cls, sub_format)])
        for attr, want in expected.items():
            got = redistribute._slot_tensor(layer, attr, slot)
            assert got.dtype == want.dtype
            torch.testing.assert_close(got, want)


def test_reload_lands_w8a8_w2_scale_at_the_slot_dtype():
    """The w2 scale is stored at whatever dtype the runtime slot holds."""
    layer = _make_layer(AscendW8A8DynamicFusedMoEMethod, w2_scale_dtype=torch.bfloat16)
    _, local_slots, buckets = _run_reload([layer], AscendW8A8DynamicFusedMoEMethod)

    for (_, expert), slot in local_slots.items():
        want = _expected(buckets[(0, expert)], AscendW8A8DynamicFusedMoEMethod, torch.bfloat16)["w2_weight_scale"]
        got = redistribute._slot_tensor(layer, "w2_weight_scale", slot)
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got, want)


def test_reload_splits_chunks_on_slot_dtype():
    """Layers whose slots disagree on dtype must not share a stacked chunk."""
    fp32_layer = _make_layer(AscendW8A8DynamicFusedMoEMethod, w2_scale_dtype=torch.float32)
    bf16_layer = _make_layer(AscendW8A8DynamicFusedMoEMethod, w2_scale_dtype=torch.bfloat16)
    layers = [fp32_layer, bf16_layer]
    _, local_slots, buckets = _run_reload(layers, AscendW8A8DynamicFusedMoEMethod)

    for (layer_idx, expert), slot in local_slots.items():
        layer = layers[layer_idx]
        slot_dtype = layer.w2_weight_scale_list[slot].dtype
        want = _expected(buckets[(layer_idx, expert)], AscendW8A8DynamicFusedMoEMethod, slot_dtype)
        got = redistribute._slot_tensor(layer, "w2_weight_scale", slot)
        assert got.dtype == slot_dtype
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("scheme_cls,sub_format", SCHEMES)
def test_reload_handles_the_whole_tensor_layout(scheme_cls, sub_format):
    """Without per-slot lists the slot is a dim-0 slice of the base tensor."""
    layer = _make_layer(scheme_cls, sub_format, per_slot=False)
    _, local_slots, buckets = _run_reload([layer], scheme_cls, sub_format)

    for (_, expert), slot in local_slots.items():
        expected = _expected(buckets[(0, expert)], scheme_cls, torch.float32)
        for attr, want in expected.items():
            got = getattr(layer, attr)[slot]
            assert got.dtype == want.dtype
            torch.testing.assert_close(got, want)


def test_reload_keeps_w8a8_weights_in_fractal_nz():
    """W8A8 weights are always laid out NZ, whatever the hardware policy."""
    layer = _make_layer(AscendW8A8DynamicFusedMoEMethod)
    with patch.object(
        redistribute.torch_npu, "npu_format_cast", side_effect=lambda weight, fmt, **kwargs: weight
    ) as format_cast:
        _run_reload([layer], AscendW8A8DynamicFusedMoEMethod)

    # One cast per weight tensor per chunk.
    assert [call.args[1] for call in format_cast.call_args_list] == [ACL_FORMAT_FRACTAL_NZ] * 2


def test_reload_follows_weight_nz_mode(monkeypatch, _nz_weight_layout_off):
    """W4A8 weights take the policy-driven NZ cast when the config asks for it."""
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", _nz_weight_layout_off(1))
    layer = _make_layer(AscendW4A8DynamicFusedMoEMethod, ASCEND_QUANTIZATION_METHOD)
    with patch.object(
        redistribute.torch_npu, "npu_format_cast", side_effect=lambda weight, fmt, **kwargs: weight
    ) as format_cast:
        _, local_slots, buckets = _run_reload([layer], AscendW4A8DynamicFusedMoEMethod, ASCEND_QUANTIZATION_METHOD)

    assert [call.args[1] for call in format_cast.call_args_list] == [ACL_FORMAT_FRACTAL_NZ] * 2
    for (_, expert), slot in local_slots.items():
        expected = _expected(buckets[(0, expert)], AscendW4A8DynamicFusedMoEMethod, torch.float32)
        for attr, want in expected.items():
            torch.testing.assert_close(redistribute._slot_tensor(layer, attr, slot), want)


def test_reload_rejects_moe_expert_bias():
    scheme, layer = AscendW8A8DynamicFusedMoEMethod, _make_layer(AscendW8A8DynamicFusedMoEMethod)
    layer.w13_bias = torch.zeros(SLOTS, 2 * INTER)
    with pytest.raises(NotImplementedError, match="expert bias"):
        redistribute._reload_local_slots([layer], {(0, 0): 0}, {(0, 0): _checkpoint(scheme, 0)})


def test_compressed_tensors_w4a8_is_rejected():
    """CT W4A8 ships one int4 per int8 and derives scale_bias: unsupported."""
    layer = _make_layer(AscendW4A8DynamicFusedMoEMethod, "compressed-tensors")
    with pytest.raises(NotImplementedError, match="compressed-tensors"):
        redistribute._resolve_scheme(layer)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.3.mlp.experts.12.gate_proj.weight", ((3, 12), "gate_proj.weight")),
        ("model.layers.3.mlp.experts.12.down_proj.weight", ((3, 12), "down_proj.weight")),
        ("model.layers.3.mlp.experts.12.down_proj.scale_bias", ((3, 12), "down_proj.scale_bias")),
    ],
)
def test_expert_tensor_names_resolve(name, expected):
    wanted = {"model.layers.3.mlp.experts": {12: (3, 12)}}
    assert redistribute._resolve_expert_tensor(name, wanted) == expected


def test_expert_tensor_names_reject_unwanted():
    wanted = {"model.layers.3.mlp.experts": {12: (3, 12)}}
    # Non-expert tensors, another expert, and another layer all fall through.
    for name in (
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.experts.13.gate_proj.weight",
        "model.layers.4.mlp.experts.12.gate_proj.weight",
        "model.layers.3.mlp.experts.notanexpert.gate_proj.weight",
    ):
        assert redistribute._resolve_expert_tensor(name, wanted) is None
