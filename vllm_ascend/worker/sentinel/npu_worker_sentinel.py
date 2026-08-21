# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.distributed.parallel_state import get_active_elastic_info_mask
from vllm_ascend.worker.sentinel.eplb_redistribute import compute_dead_ep_ranks

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart, DP group re-initialization on retry, and expert
    redistribution + DP shrink on scale-down.

    ``retry`` is kept a thin override (``reset_device`` + ``super().retry``);
    the upstream retry flow handles the worker-state cleanup and the DP
    Gloo re-initialization — including the reduced ``dp_group_rank`` /
    ``dp_group_size`` supplied on scale-down. ``scale_down`` only adds the
    Ascend-specific expert redistribution on top of ``self.retry``.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        self.dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Query mask for fault tolerance.

        Ascend's MC2 kernel has no kernel-side mask query, so the mask is
        reported from the worker-side ``ElasticInfoMask`` state via
        ``_NpuAll2AllManager`` (0 = live, 1 = dead, upstream convention).
        """
        if get_active_elastic_info_mask() is None:
            # Fault tolerance not fully set up; report all ranks live.
            parallel_config = self.worker.parallel_config
            ep_world_size = (
                parallel_config.data_parallel_size
                * parallel_config.prefill_context_parallel_size
                * parallel_config.tensor_parallel_size
            )
            return {"mask": ep_world_size * [0]}
        from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager

        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        import torch_npu

        from vllm_ascend.platform import NPUPlatform

        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # Reset the device first so any hung device-side collectives are
        # aborted, then run the upstream flow (clean worker state, re-initialize
        # the DP Gloo group with the reduced rank/size from the FT params).
        self.reset_device()
        super().retry(ft_request)

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """One-shot expert redistribution after scale-down (model runner V2).

        The V2 runner keeps the EPLB maps in ``model_runner.eplb_state``
        (``AscendEplbState``). Redistribution masks the dead EP ranks'
        physical slots, re-hosts the orphaned logical experts into spare
        (redundant) slots, rebuilds the logical maps in place and reloads the
        affected weights into the Ascend runtime layout (unquantized or W8A8).

        The persistent ``logical_to_physical_map`` is DENSIFIED in place
        (surviving EP ranks renumbered 0..k-1) so the kernel-facing
        ``expert_ids`` stay inside ``[0, alive_ep*num_local)``; the DP rank
        coordinates are left unchanged (slots model, handled by the upstream
        engine/sentinel).
        """
        from vllm.distributed import get_ep_group
        from vllm.v1.worker.sentinel.eplb_redistribute import (
            check_redundancy_sufficient,
            mark_dead_expert_slots_inplace,
            rebuild_logical_expert_maps,
            rebuild_model_expert_maps,
            redistribute_expert_placement,
        )

        from vllm_ascend.worker.sentinel.eplb_redistribute import reload_experts_from_disk

        model_runner = self.worker.model_runner
        eplb_state = getattr(model_runner, "eplb_state", None)
        if eplb_state is None:
            raise RuntimeError(
                "[FT] scale_down requires EPLB with redundant expert slots "
                "(model_runner.eplb_state is None). Enable EPLB and set "
                "num_redundant_experts > 0."
            )

        ms = eplb_state.model_states[model_runner.model_config.compute_hash()]
        p2l = ms.physical_to_logical_map
        l2p = ms.logical_to_physical_map
        lrc = ms.logical_replica_count
        num_logical = lrc.shape[1]
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size

        check_redundancy_sufficient(num_logical, num_local_experts, ep_world_size, dead_ep_ranks)
        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(p2l, num_logical, num_local_experts)
        rebuild_logical_expert_maps(p2l, l2p, lrc)
        # The MC2 dispatch/combine kernels expect expert_ids in the DENSIFIED
        # space [0, alive_ep*num_local) after scale-down, so densify the
        # persistent logical_to_physical_map in place: the surviving EP ranks
        # are renumbered 0..k-1 and physical ids become
        # local_rank*num_local+slot. The upstream rebuild keeps the ORIGINAL
        # physical ids, which overflow the kernel's scaled-down expert space
        # and fault with a CCU instruction address check on the dummy batch.
        surviving = sorted(set(range(ep_world_size)) - dead_ep_ranks)
        rank_to_local = {orig: i for i, orig in enumerate(surviving)}
        if rank_to_local and len(rank_to_local) < ep_world_size:
            l2p_cpu = l2p.cpu()
            lrc_cpu = lrc.cpu()
            for layer_idx in range(l2p_cpu.shape[0]):
                for logical in range(l2p_cpu.shape[1]):
                    for rep in range(int(lrc_cpu[layer_idx, logical])):
                        pid = int(l2p_cpu[layer_idx, logical, rep])
                        if pid < 0:
                            continue
                        orig_rank, slot = divmod(pid, num_local_experts)
                        l2p_cpu[layer_idx, logical, rep] = rank_to_local[orig_rank] * num_local_experts + slot
            l2p.copy_(l2p_cpu)
            # Keep elastic_info's physical expert width consistent with the
            # densified expert id space (alive_ep * num_local).
            mask = getattr(self.worker, "elastic_info_mask", None)
            if mask is not None:
                mask.set_num_physical_experts(len(surviving) * num_local_experts)
        # The Ascend MC2 dispatch reads each layer's derived
        # ``expert_replica_routing_table`` (built from l2p/lrc + ep_rank), so
        # refresh it after the logical maps change (now built from the
        # densified l2p).
        from vllm_ascend.distributed.eplb_state import refresh_model_routing_tables

        refresh_model_routing_tables(ms)
        rebuild_model_expert_maps(model_runner.model, p2l)
        if reassignments:
            reload_experts_from_disk(model_runner.model, self.worker.vllm_config, reassignments)

        logger.info(
            "[FT] Expert redistribution: ep_world_size=%d, num_logical=%d, dead_ep_ranks=%s, reassignments=%d",
            ep_world_size,
            num_logical,
            sorted(dead_ep_ranks),
            len(reassignments),
        )

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Remove dead DP ranks and re-host their experts on surviving ranks.

        Validates the dead ranks, masks the dead EP ranks, suppresses EPLB,
        delegates the mechanical recovery to ``retry`` (which forwards to
        ``super().retry``), then runs the one-shot expert redistribution.

        Ascend's EP topology is NOT rearranged: dead EP ranks are masked via
        ``ElasticInfoMask`` (the equivalent of the all2all ``update_mask``) and
        the orphaned experts are deterministically re-hosted on surviving
        ranks' redundant slots, with weights reloaded from disk.

        Slots model (see vllm-project/vllm#46370): rank coordinates never
        change. ``parallel_config.data_parallel_rank/size`` and the model
        runner's cached ``dp_rank``/``dp_size`` stay frozen at their initial
        values; only the DP Gloo ``cpu_group`` is rebuilt dense over the
        surviving slots. Dead EP ranks are excluded at the MC2 kernel via the
        elastic_info mask. The EXPERT id space, however, is densified
        (surviving EP ranks renumbered 0..k-1 in the persistent
        ``logical_to_physical_map``) because the MC2 kernels require
        ``expert_ids`` in ``[0, alive_ep*num_local)`` after scale-down.

        Only the model runner V2 is supported: the EPLB maps live in
        ``model_runner.eplb_state`` (``AscendEplbState``).
        """
        params = ft_request.params
        dead_dp_ranks = params.get("dead_dp_ranks") or params.get("removed_dp_ranks")
        if dead_dp_ranks is None:
            raise ValueError("[FT] scale_down params missing dead_dp_ranks/removed_dp_ranks")

        parallel_config = self.worker.vllm_config.parallel_config
        dead_ep_ranks = compute_dead_ep_ranks(
            dead_dp_ranks,
            parallel_config.prefill_context_parallel_size,
            parallel_config.tensor_parallel_size,
        )

        # Match the upstream flow: suppress EPLB async rebalancing BEFORE
        # retry, so retry skips re-initializing the EP/EPLB Gloo groups
        # (which still contain the dead ranks and would hang).
        self.worker.model_runner.eep_eplb_suppressed = True

        # Reuse the upstream retry flow: reset device, clean worker state and
        # re-create the DP Gloo group over the surviving slots (dense internal
        # ranks, read from dp_group_rank/dp_group_size in params). Under the
        # slots model the ORIGINAL rank coordinates are kept everywhere else:
        # parallel_config.data_parallel_rank/size and the v2 model runner's
        # cached dp_rank/dp_size stay frozen at their initial values, so the
        # per-step DP metadata sync keeps the original tensor width and indexes
        # each rank by its original DP rank (dead columns are neutralized via
        # get_dp_group().dead_dp_ranks, see vllm/v1/worker/gpu/dp_utils.py).
        self.retry(ft_request)

        from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager

        mgr = get_ep_all2all_manager()
        for ep_rank in sorted(dead_ep_ranks):
            mgr.update_mask(ep_rank, masked=True)
        # Re-anchor the fault-detection baseline to the post-scale-down mask so
        # the now-known dead ranks are not reported as a fresh fault by
        # ``query_fault`` on the next model output.
        mgr.clean_buffers()

        # One-shot expert redistribution + weight reload. The dead EP ranks are
        # already masked on the all2all manager by ``super().retry`` (replay of
        # ``params["dead_dp_ranks"]``), so no manual update_mask is needed here.
        self._redistribute_experts(dead_ep_ranks)

        # The FT elastic_info / masked expert maps are MC2-specific, so pin
        # the MoE communication to MC2 for every subsequent forward.
        from vllm_ascend.ascend_forward_context import set_force_mc2

        set_force_mc2(True)

        # Verify the reconfigured MoE can actually run.
        self.worker.execute_dummy_batch()
        torch.npu.synchronize()

        logger.info(
            "[FT] Worker scale_down complete: original dp_size=%d, original dp_rank=%d, dead_ep_ranks=%s",
            self.dp_size,
            self.dp_rank,
            sorted(dead_ep_ranks),
        )
