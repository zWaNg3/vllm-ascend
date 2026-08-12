# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.distributed.parallel_state import (
    get_active_elastic_info_mask,
    get_elastic_info,
    get_mc2_group,
)
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    compute_dead_ep_ranks,
    rebuild_model_expert_maps,
    reconfigure_moe,
    redistribute_expert_placement,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart, DP group re-initialization on retry, and expert
    redistribution + DP shrink on scale-down.

    ``retry`` is kept a thin override (``reset_device`` + ``super().retry``);
    the upstream flow (PR #46370) handles the worker-state cleanup and the DP
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
        # the DP Gloo group — using the reduced rank/size from params once
        # PR #46370 lands).
        self.reset_device()
        super().retry(ft_request)

    def _get_moe_layers(self) -> list:
        """Collect the AscendMoERunner layers (main + drafter) on this PP rank.

        Ascend equivalent of the upstream ``model.moe_layers``.
        """
        from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

        layers = [m for m in self.worker.model_runner.get_model().modules() if isinstance(m, AscendMoERunner)]
        drafter = getattr(self.worker.model_runner, "drafter", None)
        if drafter is not None and hasattr(drafter, "model"):
            layers.extend(m for m in drafter.model.modules() if isinstance(m, AscendMoERunner))
        return layers

    def _redistribute_experts_v2(self, dead_ep_ranks: set[int]) -> None:
        """One-shot expert redistribution for the v2 model runner.

        The v2 runner keeps the EPLB maps in ``model_runner.eplb_state``
        (``AscendEplbState``) instead of the per-layer ``global_expert_map``
        used by v1, so the redistribution mirrors the upstream
        ``GPUWorkerSentinel._redistribute_experts`` flow (PR #46370): mask the
        dead EP ranks' physical slots, re-host the orphaned logical experts
        into spare (redundant) slots, rebuild the logical maps in place and
        reload the affected weights through the standard loader.
        """
        from vllm.distributed import get_ep_group
        from vllm.v1.worker.sentinel.eplb_redistribute import (
            check_redundancy_sufficient,
            mark_dead_expert_slots_inplace,
            rebuild_logical_expert_maps,
            rebuild_model_expert_maps,
            redistribute_expert_placement,
            reload_experts_from_disk,
        )

        model_runner = self.worker.model_runner
        eplb_state = getattr(model_runner, "eplb_state", None)
        if eplb_state is None:
            raise RuntimeError(
                "[FT] v2 scale_down requires EPLB with redundant expert slots "
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

        # TODO(FT-DEBUG): remove after diagnosing v2 dummy-batch device error.
        def _d(name, t):
            if t is None:
                logger.info("[FT][DEBUG] v2 %s=None", name)
                return
            try:
                logger.info(
                    "[FT][DEBUG] v2 %s shape=%s val=%s",
                    name,
                    tuple(t.shape),
                    (t.cpu().flatten()[:24].tolist() if t.numel() else []),
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("[FT][DEBUG] v2 %s dump failed: %r", name, exc)

        _d("p2l", p2l)
        _d("l2p", l2p)
        _d("lrc", lrc)
        _d("elastic_info", get_elastic_info())
        try:
            _d("layer0_rt_before", model_runner.model.moe_layers[0].eplb_state.expert_replica_routing_table)
            _d("layer0_expert_map_before", model_runner.model.moe_layers[0].routed_experts._expert_map)
        except Exception as exc:  # noqa: BLE001
            logger.info("[FT][DEBUG] v2 layer0 dump failed: %r", exc)

        check_redundancy_sufficient(num_logical, num_local_experts, ep_world_size, dead_ep_ranks)
        mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
        reassignments = redistribute_expert_placement(p2l, num_logical, num_local_experts)
        rebuild_logical_expert_maps(p2l, l2p, lrc)
        # The Ascend MC2 kernel's elastic_info expects the DENSIFIED id space
        # after scale-down (mirroring the v1 flow's densified log2phy): the
        # surviving EP ranks are renumbered 0..k-1 and physical ids become
        # local_rank*num_local+slot in [0, alive_ep*num_local). The upstream
        # rebuild keeps the ORIGINAL physical ids (0..num_physical), which
        # overflow the kernel's scaled-down expert space and fault with a CCU
        # instruction address check on the dummy batch.
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
            # Keep elastic_info's expert width consistent with the densified
            # ids (alive_ep * num_local), matching the v1 reconfigure_moe +
            # set_num_physical_experts behavior.
            mask = getattr(self.worker, "elastic_info_mask", None)
            if mask is not None:
                mask.set_num_physical_experts(len(surviving) * num_local_experts)
        # The Ascend MC2 dispatch reads each layer's derived
        # ``expert_replica_routing_table`` (built from l2p/lrc + ep_rank), so
        # refresh it after the logical maps change, or the kernel routes on a
        # stale table and faults (CCU instruction address check) on the dummy
        # batch.
        from vllm_ascend.distributed.eplb_state import refresh_model_routing_tables

        refresh_model_routing_tables(ms)
        rebuild_model_expert_maps(model_runner.model, p2l)
        # TODO(FT-DEBUG): remove after diagnosing v2 dummy-batch device error.
        logger.info("[FT][DEBUG] v2 reassignments=%s", sorted(reassignments) if reassignments else [])
        _d("p2l_after", p2l)
        _d("l2p_after", l2p)
        _d("lrc_after", lrc)
        _d("elastic_info_after", get_elastic_info())
        try:
            _d("layer0_rt_after", model_runner.model.moe_layers[0].eplb_state.expert_replica_routing_table)
            _d("layer0_expert_map_after", model_runner.model.moe_layers[0].routed_experts._expert_map)
        except Exception as exc:  # noqa: BLE001
            logger.info("[FT][DEBUG] v2 layer0 after dump failed: %r", exc)
        if reassignments:
            reload_experts_from_disk(model_runner.model, self.worker.vllm_config, reassignments)

        logger.info(
            "[FT] v2 Expert redistribution: ep_world_size=%d, num_logical=%d, dead_ep_ranks=%s, reassignments=%d",
            ep_world_size,
            num_logical,
            sorted(dead_ep_ranks),
            len(reassignments),
        )

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """One-shot expert redistribution after scale-down.

        Mirrors the upstream ``_redistribute_experts`` (PR #46370): for every
        MoE layer, deterministically re-host the dead ranks' orphaned experts
        into free redundant slots (no cross-rank communication), rebuild the
        maps in place and reload the affected layers from disk through the
        standard loader (quant-agnostic).

        The v1 and v2 model runners keep the EPLB maps in different places:
        v1 stores ``global_expert_map``/``log2phy``/``ascend_expert_map`` on
        ``AscendRoutedExperts``, while v2 keeps them in
        ``model_runner.eplb_state`` (``AscendEplbState``). Route to the
        matching implementation.
        """
        if self.worker.use_v2_model_runner:
            self._redistribute_experts_v2(dead_ep_ranks)
            return

        model_runner = self.worker.model_runner
        ep_rank = get_mc2_group().rank_in_group
        tp_size = self.worker.parallel_config.tensor_parallel_size
        layers = self._get_moe_layers()

        # TODO(FT-DEBUG): remove after diagnosing garbled output post-scale-down.
        from vllm_ascend.ascend_config import _MEGA_MOE_SUPPORTED, get_ascend_config

        _cfg = get_ascend_config()
        first = layers[0].routed_experts if layers else None
        logger.info(
            "[FT][DEBUG] scale_down cfg: enable_fused_mc2=%s mega_moe=%s num_layers=%d "
            "w13_weight=%s w13_weight_list=%s",
            _cfg.enable_fused_mc2,
            _MEGA_MOE_SUPPORTED,
            len(layers),
            (hasattr(first, "w13_weight") if first is not None else None),
            (isinstance(getattr(first, "w13_weight_list", None), list) if first is not None else None),
        )

        new_expert_maps = []
        # layer_id -> [(slot, logical)] experts to reload into their new slots.
        reload_set: dict[int, list[tuple[int, int]]] = {}
        new_num_physical: int | None = None
        for layer_id, layer in enumerate(layers):
            # In the refactored MoE layer, the EPLB maps live on the inner
            # AscendRoutedExperts object, not on the AscendMoERunner wrapper.
            routed = layer.routed_experts
            if getattr(routed, "global_expert_map", None) is None:
                raise RuntimeError(
                    "[FT] scale_down requires EPLB with redundant expert slots "
                    "(global_expert_map is None). Set num_redundant_experts > 0 "
                    "and enable dynamic EPLB."
                )
            num_local = int(layer.moe_config.num_local_experts)
            num_logical = int(routed.global_expert_map.shape[1])
            new_map, log2phy, reassignments, new_num_physical = redistribute_expert_placement(
                routed.global_expert_map,
                dead_ep_ranks,
                ep_rank,
                tp_size=tp_size,
            )
            # TODO(FT-DEBUG): remove after diagnosing garbled output post-scale-down.
            logger.info(
                "[FT][DEBUG] layer %d BEFORE: old_global_row=%s old_expert_map=%s old_log2phy=%s",
                layer_id,
                routed.global_expert_map[ep_rank].cpu().tolist(),
                (routed.ascend_expert_map.cpu().tolist() if routed.ascend_expert_map is not None else None),
                (routed.log2phy.cpu().tolist() if routed.log2phy is not None else None),
            )
            rebuild_model_expert_maps(layer, new_map, ep_rank, log2phy)
            # Shrink the physical-expert counts to the surviving slots (demo
            # reconfigure_moe), so moe_expert_num matches elastic_info.
            reconfigure_moe(layer, num_logical, new_num_physical, num_local)
            # TODO(FT-DEBUG): remove after diagnosing garbled output post-scale-down.
            logger.info(
                "[FT][DEBUG] layer %d AFTER:  new_global_row=%s new_expert_map=%s new_log2phy=%s reassignments=%s",
                layer_id,
                routed.global_expert_map[ep_rank].cpu().tolist(),
                (routed.ascend_expert_map.cpu().tolist() if routed.ascend_expert_map is not None else None),
                (routed.log2phy.cpu().tolist() if routed.log2phy is not None else None),
                reassignments,
            )
            for slot, logical in reassignments:
                map_slot = routed.ascend_expert_map[logical].item()
                phys = routed.log2phy[logical].item()
                derived_slot = phys % num_local
                logger.info(
                    "[FT][DEBUG] layer %d logical=%d: map_slot=%d log2phy=%d "
                    "derived_slot(=log2phy%%num_local)=%d  => %s",
                    layer_id,
                    logical,
                    map_slot,
                    phys,
                    derived_slot,
                    "OK" if map_slot == slot == derived_slot else "MISMATCH",
                )
            if reassignments:
                reload_set[layer_id] = reassignments
            new_expert_maps.append(new_map.cpu())

        # Keep the EPLB subprocess shared state consistent with the new
        # placement (the subprocess is not woken again after scale-down).
        shared_dict = getattr(model_runner, "shared_dict", None)
        if shared_dict is not None and "expert_maps" in shared_dict:
            shared_dict["expert_maps"] = torch.stack(new_expert_maps)

        if reload_set:
            reload_experts_from_disk(
                model_runner.get_model(),
                self.worker.vllm_config,
                layers,
                reload_set,
            )

        # elastic_info's num_physical_experts shrinks to the surviving slots.
        mask = getattr(self.worker, "elastic_info_mask", None)
        if mask is not None and new_num_physical is not None:
            mask.set_num_physical_experts(new_num_physical)

        logger.info(
            "[FT] Expert redistribution: ep_rank=%d, dead_ep_ranks=%s, reload_layers=%s, new_num_physical=%s",
            ep_rank,
            sorted(dead_ep_ranks),
            sorted(reload_set),
            new_num_physical,
        )

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Remove dead DP ranks and re-host their experts on surviving ranks.

        Mirrors the upstream ``GPUWorkerSentinel.scale_down`` structure
        (PR #46370): validate, mask the dead EP ranks, suppress EPLB, delegate
        the mechanical recovery to ``retry`` (which forwards to
        ``super().retry``), then run the one-shot expert redistribution.

        Ascend's EP topology is NOT rearranged: dead EP ranks are masked via
        ``ElasticInfoMask`` (the equivalent of the all2all ``update_mask``) and
        the orphaned experts are deterministically re-hosted on surviving
        ranks' redundant slots, with weights reloaded from the CPU stash.
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

        if self.worker.use_v2_model_runner:
            # Match the upstream flow: suppress EPLB async rebalancing BEFORE
            # retry, so retry skips re-initializing the EP/EPLB Gloo groups
            # (which still contain the dead ranks and would hang). v1 keeps
            # the suppression after retry, mirroring the demo branch.
            self.worker.model_runner.eep_eplb_suppressed = True

        # Reuse the upstream retry flow: reset device, clean worker state and
        # re-create the DP Gloo group with the reduced (densified) membership
        # (the upstream retry reads dp_group_rank/dp_group_size from params).
        self.retry(ft_request)

        # The upstream retry does not touch the model runner's cached DP state,
        # but the Ascend v1 runner keeps dp_size/dp_rank and uses them for the
        # DP metadata all-reduce, so refresh them here (v2 reads the parallel
        # config directly and keeps no runner-local copy).
        dp_group_rank = int(params.get("dp_group_rank", params.get("new_dp_rank")))
        dp_group_size = int(params.get("dp_group_size", params.get("new_dp_size")))
        self.dp_rank = dp_group_rank
        self.dp_size = dp_group_size
        parallel_config.data_parallel_rank = dp_group_rank
        parallel_config.data_parallel_size = dp_group_size
        model_runner = self.worker.model_runner
        if not self.worker.use_v2_model_runner:
            model_runner.dp_rank = dp_group_rank
            model_runner.dp_size = dp_group_size

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

        # Suppress EPLB async rebalancing (demo style): keep dynamic_eplb
        # enabled but stop the updator's cross-rank all_gather / d2d updates,
        # which would hang on the dead ranks. Re-enabled by a future scale-up.
        model_runner.eep_eplb_suppressed = True
        if hasattr(model_runner, "dynamic_eplb") and model_runner.dynamic_eplb:
            model_runner.dynamic_eplb = False
            model_runner.eplb_enabled = False

        # Verify the reconfigured MoE can actually run.
        self.worker.execute_dummy_batch()
        torch.npu.synchronize()

        logger.info(
            "[FT] Worker scale_down complete: dp_size=%d, dp_rank=%d, dead_ep_ranks=%s",
            self.dp_size,
            self.dp_rank,
            sorted(dead_ep_ranks),
        )
