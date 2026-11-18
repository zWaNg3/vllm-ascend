# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Per-slot list restore for the upstream-style layerwise reload."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.sentinel.eplb_layerwise_reload import (
    _restore_slot_lists_in_place,
    _slot_lists,
)


def _layer_with_slots() -> SimpleNamespace:
    layer = SimpleNamespace()
    layer.w13_weight_list = [torch.zeros(2, 3), torch.ones(2, 3)]
    layer.w2_weight_list = [torch.full((3, 2), 2.0)]
    layer.expert_weights = [layer.w13_weight_list, layer.w2_weight_list]  # not a slot list
    return layer


def test_slot_lists_only_capture_tensor_lists():
    layer = _layer_with_slots()

    captured = _slot_lists(layer)

    assert set(captured) == {"w13_weight_list", "w2_weight_list"}


def test_restore_copies_into_original_storage_and_keeps_the_list_objects():
    layer = _layer_with_slots()
    captured = _slot_lists(layer)
    original_storage = [slot.data_ptr() for slot in layer.w13_weight_list]

    # Simulate process_weights_after_loading rebuilding every slot.
    layer.w13_weight_list = [torch.full((2, 3), 5.0), torch.full((2, 3), 6.0)]
    layer.w2_weight_list = [torch.full((3, 2), 7.0)]

    _restore_slot_lists_in_place(layer, captured)

    assert [slot.data_ptr() for slot in layer.w13_weight_list] == original_storage
    assert torch.equal(layer.w13_weight_list[0], torch.full((2, 3), 5.0))
    assert torch.equal(layer.w13_weight_list[1], torch.full((2, 3), 6.0))
    assert torch.equal(layer.w2_weight_list[0], torch.full((3, 2), 7.0))


def test_restore_rejects_a_list_process_weights_after_loading_did_not_rebuild():
    layer = _layer_with_slots()
    captured = _slot_lists(layer)

    with pytest.raises(RuntimeError, match="did not rebuild"):
        _restore_slot_lists_in_place(layer, captured)


def test_restore_rejects_a_changed_slot_layout():
    layer = _layer_with_slots()
    captured = _slot_lists(layer)

    layer.w13_weight_list = [torch.zeros(4, 4), torch.zeros(4, 4)]
    layer.w2_weight_list = [torch.zeros(3, 2)]

    with pytest.raises(RuntimeError, match="changed"):
        _restore_slot_lists_in_place(layer, captured)
