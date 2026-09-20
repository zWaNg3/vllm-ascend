# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch_npu
import vllm.v1.worker.sentinel.gpu_worker_sentinel as _gpu_worker_sentinel
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.state import refresh_model_routing_tables
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    build_orig_to_dense_rank_table,
    densify_routing_table_physical_ids,
    mark_dead_expert_slots_inplace,
    rebuild_logical_expert_maps,
    rebuild_model_expert_maps,
    redistribute_expert_placement,
    reload_draft_experts_from_disk,
    reload_experts_from_disk,
)

# Route the reload call inside the inherited upstream
# GPUWorkerSentinel._redistribute_experts to the Ascend implementation: the
# upstream reloader writes via model.load_weights, which cannot produce
# Ascend's runtime expert layout (transpose / NZ / per-slot lists / quant
# scales). The signatures match (a set of (layer, logical) reassignments), so
# super() flows pick up the Ascend reloader without any upstream change.
_gpu_worker_sentinel.reload_experts_from_disk = reload_experts_from_disk

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def fault_barrier_wrapper(func: Callable):
    """Barrier between device faults and the async step loop.

    On the first device-touching fault (e.g. an EP-group allreduce failing on a
    dead peer that poisons the model stream) it quarantines the worker and
    resets the device immediately, before any tensor teardown on the broken
    stream can std::terminate the process. While quarantined, wrapped methods
    short-circuit with an empty output so in-flight async steps drain safely
    without re-hitting the device; retry lifts the quarantine only after the
    groups are rebuilt.
    """

    def wrapped(self, *args, **kwargs):
        sentinel = getattr(self, "worker_sentinel", None)
        if sentinel is not None and sentinel.worker_faulted:
            return EMPTY_MODEL_RUNNER_OUTPUT
        try:
            return func(self, *args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            if sentinel is not None:
                sentinel.worker_faulted = True
                logger.warning("[FT] Quarantining worker %d after fault: %s", self.rank, exc)
                try:
                    sentinel.reset_device()
                except Exception:
                    logger.exception("[FT] self device reset failed on worker %d.", self.rank)
                return EMPTY_MODEL_RUNNER_OUTPUT
            raise

    return wrapped


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.device = device
        self.worker = worker
        # Set once a device-touching method faults, to keep this worker off the
        # device until FT recovery rebuilds the groups.
        self.worker_faulted = False

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead)."""
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # Reset first so hung device collectives are aborted, then run the
        # base flow and lift the quarantine after the groups are rebuilt.
        self.reset_device()
        super().retry(ft_request)
        self.worker_faulted = False

    def init_num_local_experts(self) -> None:
        """Record the per-rank physical expert slot count after model load."""
        if self.worker.model_runner.eplb_state is None:
            return
        eplb_model_state = self._eplb_model_state()
        num_local_experts = eplb_model_state.physical_to_logical_map.shape[1] // get_ep_group().world_size
        get_ep_all2all_manager().set_num_local_physical_experts(num_local_experts)

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Scale down over the surviving DP ranks, reusing the upstream flow.

        ``super().scale_down`` runs the deterministic dead-rank masking and
        expert redistribution, dispatching ``retry`` / ``_redistribute_experts``
        to the Ascend overrides below. Ascend adds its platform preconditions
        and a dummy-batch runnability check on top.
        """
        self._validate_scale_down_preconditions()
        super().scale_down(ft_request)

        # Verify the redistributed model is runnable before reporting healthy.
        self.worker.execute_dummy_batch()
        torch.npu.synchronize()

    def _validate_scale_down_preconditions(self) -> None:
        if not self.worker.use_v2_model_runner:
            raise ValueError("[FT] scale_down on Ascend NPU requires the v2 model runner.")
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 to re-host the dead rank's experts."
            )
        ascend_config = get_ascend_config()
        if ascend_config.enable_fused_mc2:
            raise ValueError(
                "[FT] scale_down is not supported with enable_fused_mc2: the "
                "fused dispatch_ffn_combine operators take no elastic_info."
            )
        if ascend_config.enable_mc2_hierarchy_comm:
            raise ValueError(
                "[FT] scale_down (elastic_info) is mutually exclusive with mc2 hierarchy comm (comm_alg='hierarchy')."
            )
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise ValueError(
                "[FT] scale_down requires npu_moe_distribute_dispatch_v2 "
                "(aclnn V3+); please upgrade the CANN/torch_npu version."
            )

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """Redistribute experts onto the surviving slots after scale-down.

        Reuses the upstream redistribution (mark dead slots, steal spare slots
        for the missing experts, rebuild the logical maps and reload reassigned
        weights through the Ascend reloader patched into the shared flow). On
        top of that, refreshes the Ascend kernel-facing routing tables into the
        densified id space and shrinks the MC2 physical-expert width.
        """
        # Precompute the reassignment set up front: super()._redistribute_experts
        eplb_model_state = self._eplb_model_state()
        p2l = eplb_model_state.physical_to_logical_map
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        num_logical = eplb_model_state.logical_replica_count.shape[1]
        scratch = p2l.detach().clone()
        mark_dead_expert_slots_inplace(scratch, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(scratch, num_logical, num_local_experts)

        super()._redistribute_experts(dead_ep_ranks)

        eplb_model_state = self._eplb_model_state()
        # Propagate the new placement into the Ascend routing tables (in-place,
        # so captured graphs keep pointing at valid storage), then renumber
        # their ids into the densified space for the MC2 kernels.
        refresh_model_routing_tables(eplb_model_state)
        self._densify_routing_tables(eplb_model_state)

        # The speculative drafter(MTP/DSpark) keeps independent EPLB layer
        # states / routing tables and expert weights, sharing the main model's
        # placement; scale down bypasses it, so its tables stay stale (still
        # referencing dead ranks' physical ids). Re-sync them to the new
        # redistributed placement and reload the re-hosted expert weights.
        draft_model = getattr(self.worker.model_runner, "speculator", None)
        draft_model = getattr(draft_model, "model", None)
        if draft_model is not None and getattr(draft_model, "moe_layers", None):
            self._sync_drafter_eplb(draft_model, eplb_model_state.physical_to_logical_map, reassignments)

    def _sync_drafter_eplb(
        self,
        draft_model,
        p2l: torch.Tensor,
        reassignments: set[tuple[int, int]],
    ) -> None:
        """Re-sync the speculative drafter's EPLB routing and expert weights
        after scale-down
        """
        num_draft_layers = len(draft_model.moe_layers)

        draft_p2l = p2l[:num_draft_layers]
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        active_mask = get_ep_all2all_manager().query_active_mask()
        dead_ranks = {rank for rank, is_dead in enumerate(active_mask.tolist()) if is_dead}
        orig_to_dense = build_orig_to_dense_rank_table(ep_world_size, dead_ranks)

        for layer in draft_model.moe_layers:
            layer_state = getattr(layer, "eplb_state", None)
            if layer_state is None:
                continue

            rebuild_logical_expert_maps(
                draft_p2l[:1],
                layer_state.logical_to_physical_map[None],
                layer_state.logical_replica_count[None],
            )
            layer_state.refresh_expert_replica_routing_table()
            routing_table = layer_state.expert_replica_routing_table
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense, num_local_experts)

            # v2 model-side expert map
            rebuild_model_expert_maps(draft_model, draft_p2l)

            # Reload the re-hosted experts into the drafter's independent weight
            reload_draft_experts_from_disk(draft_model, self.worker.vllm_config, reassignments, p2l)


    def _densify_routing_tables(self, eplb_model_state) -> None:
        """Renumber the kernel-facing routing tables into the densified id space.

        The MC2 kernels consume the routing tables in the densified id space,
        so the kernel-facing values are renumbered in place after the refresh.
        The dead set is cumulative (accumulated across recovery rounds), so it
        is derived from the manager's mask rather than this round's ranks.
        """
        p2l = eplb_model_state.physical_to_logical_map
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        active_mask = get_ep_all2all_manager().query_active_mask()
        dead_ranks = {rank for rank, is_dead in enumerate(active_mask.tolist()) if is_dead}
        orig_to_dense_rank = build_orig_to_dense_rank_table(ep_world_size, dead_ranks)
        for layer in eplb_model_state.model.moe_layers:
            routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense_rank, num_local_experts)
