# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scale-down expert-placement redistribution helpers.

These are pure torch functions with no vLLM / vllm-ascend imports, so they can
be tested without NPU hardware.
"""

import unittest

import torch

from vllm_ascend.worker.sentinel.eplb_redistribute import (
    compute_dead_ep_ranks,
    generate_log2phy_map,
    global_placement,
    redistribute_expert_placement,
)


def _logical_to_slot_map(placement: torch.Tensor) -> torch.Tensor:
    """Build ``[ep_size, n_logical]`` logical->slot from a slot->logical table."""
    ep_size, num_local = placement.shape
    n_logical = int(placement[placement >= 0].max().item()) + 1
    expert_map = torch.full((ep_size, n_logical), -1, dtype=torch.int32)
    for rank in range(ep_size):
        for slot in range(num_local):
            logical = int(placement[rank, slot].item())
            if logical >= 0:
                expert_map[rank, logical] = slot
    return expert_map


class TestComputeDeadEpRanks(unittest.TestCase):
    def test_tp_one_identity(self):
        self.assertEqual(compute_dead_ep_ranks([1, 3], pcp_size=1, tp_size=1), {1, 3})

    def test_tp_two(self):
        self.assertEqual(compute_dead_ep_ranks([1], pcp_size=1, tp_size=2), {2, 3})
        self.assertEqual(compute_dead_ep_ranks([0, 2], pcp_size=1, tp_size=2), {0, 1, 4, 5})

    def test_pcp_and_tp(self):
        self.assertEqual(compute_dead_ep_ranks([1], pcp_size=2, tp_size=2), {4, 5, 6, 7})


class TestGlobalPlacement(unittest.TestCase):
    def test_inverse_roundtrip(self):
        placement = torch.tensor(
            [[0, 1], [2, 3], [-1, -1], [0, 1]],
            dtype=torch.int32,
        )
        expert_map = _logical_to_slot_map(placement)
        self.assertTrue(torch.equal(global_placement(expert_map), placement))


class TestRedistributeExpertPlacement(unittest.TestCase):
    def test_scale_down_rehosts_orphaned_experts(self):
        # 4 EP ranks, 2 slots each, 6 logical experts + 2 redundant (copies of
        # logicals 0,1 placed on rank 3).
        placement = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [0, 1]],
            dtype=torch.int32,
        )
        expert_map = _logical_to_slot_map(placement)

        # Kill rank 2, which hosts the only copies of logicals 4 and 5.
        new_map, log2phy, reassignments = redistribute_expert_placement(expert_map, {2}, ep_rank=3)

        expected = torch.tensor(
            [[0, 1, -1, -1, -1, -1], [-1, -1, 0, 1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, 0, 1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(new_map, expected))

        # rank 3 takes over logicals 4 and 5 in its formerly redundant slots.
        self.assertEqual(reassignments, [(0, 4), (1, 5)])

        # Every logical expert maps to exactly one global physical slot and the
        # physical ids address the flattened [ep_size, num_local] layout
        # (rank 2 is dead so rank 3's slots sit at physical ids 6 and 7).
        self.assertTrue(torch.equal(log2phy, torch.tensor([6, 7, 2, 3, 6, 7], dtype=torch.int32)))

    def test_scale_down_dead_rank_covered_by_redundancy(self):
        # 4 ranks, 2 slots, 4 logical experts, 4 redundant (each logical has a
        # copy on two ranks). Killing rank 1 orphans nothing.
        placement = torch.tensor(
            [[0, 1], [2, 3], [0, 1], [2, 3]],
            dtype=torch.int32,
        )
        expert_map = _logical_to_slot_map(placement)
        new_map, log2phy, reassignments = redistribute_expert_placement(expert_map, {1}, ep_rank=0)

        self.assertEqual(reassignments, [])
        expected = torch.tensor(
            [[0, 1, -1, -1], [-1, -1, -1, -1], [0, 1, -1, -1], [-1, -1, 0, 1]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(new_map, expected))
        # logicals 2 and 3 live on rank 3 (physical ids 6 and 7).
        self.assertEqual(int(log2phy[2]), 3 * 2 + 0)
        self.assertEqual(int(log2phy[3]), 3 * 2 + 1)

    def test_scale_down_insufficient_redundancy_raises(self):
        placement = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            dtype=torch.int32,
        )
        expert_map = _logical_to_slot_map(placement)
        with self.assertRaises(RuntimeError):
            redistribute_expert_placement(expert_map, {3}, ep_rank=0)

    def test_multiple_copies_pick_replica(self):
        # Logical 0 appears on ranks 0 and 2; logical 1 only on rank 1.
        placement = torch.tensor([[0, -1], [1, -1], [0, -1], [-1, -1]], dtype=torch.int32)
        expert_map = _logical_to_slot_map(placement)
        log2phy = generate_log2phy_map(expert_map, ep_rank=2, num_local=2)
        self.assertEqual(int(log2phy[0]), 0)  # rank0 slot0
        self.assertEqual(int(log2phy[1]), 1 * 2 + 0)  # rank1 slot0


class TestGenerateLog2phyMapRobustToDeadRankZero(unittest.TestCase):
    def test_dead_rank_zero_does_not_zero_valid_count(self):
        # rank 0 fully dead (all -1); rank 1 hosts logicals 0,1.
        expert_map = torch.tensor(
            [[-1, -1, -1], [-1, -1, -1], [0, 1, -1], [-1, -1, -1]],
            dtype=torch.int32,
        )
        log2phy = generate_log2phy_map(expert_map, ep_rank=2, num_local=2)
        # physical ids are rank * num_local + slot, unaffected by dead rank 0.
        self.assertEqual(int(log2phy[0]), 2 * 2 + 0)
        self.assertEqual(int(log2phy[1]), 2 * 2 + 1)


if __name__ == "__main__":
    unittest.main()
