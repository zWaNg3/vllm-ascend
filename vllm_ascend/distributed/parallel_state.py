import torch
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator, get_world_group, init_model_parallel_group

from vllm_ascend.ascend_config import get_ascend_config

# Currently, mc2 op need their own group coordinator.
_MC2: GroupCoordinator | None = None

# The process-wide ElasticInfoMask instance (one per worker, created by
# `NPUWorker._init_elastic_info_mask`). It owns both the FT mask state and the
# derived elastic_info tensor consumed by the MC2 token-dispatch kernel. See
# `ElasticInfoMask` below for the layout and the mask semantics it encodes.
# Consumed by `_NpuAll2AllManager` so the FT sentinel can go through
# `get_ep_all2all_manager()`.
_ACTIVE_ELASTIC_INFO_MASK: "ElasticInfoMask | None" = None

# Module specific tensor parallel groups
_MLP_TP: GroupCoordinator | None = None
_OTP: GroupCoordinator | None = None
_LMTP: GroupCoordinator | None = None
_EMBED_TP: GroupCoordinator | None = None

_P_TP: GroupCoordinator | None = None

_DYNAMIC_EPLB: GroupCoordinator | None = None


def init_ascend_model_parallel(
    parallel_config: ParallelConfig,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        global_pcp_size,
        global_tp_size,
    )

    pd_tp_ratio = get_ascend_config().pd_tp_ratio
    pd_head_ratio = get_ascend_config().pd_head_ratio
    global _P_TP
    assert _P_TP is None, "distributed prefill tensor parallel group is already initialized"
    prefill_tensor_model_parallel_size = pd_tp_ratio
    # divide alltoall groups
    if pd_head_ratio > 1 and get_current_vllm_config().kv_transfer_config.is_kv_producer:
        num_head_replica = get_ascend_config().num_head_replica
        remote_tp_size = global_tp_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(-1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                global_dp_size * global_pp_size * global_pcp_size, -1, num_head_replica
            )  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(-1, group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                global_dp_size * global_pp_size * global_pcp_size,
                num_head_replica,
                -1,
                alltoall_group_size,
            )  # [DP_size, num_head_replica, num_alltoall_group, alltoall_group_size]
            group_ranks = group_ranks.reshape(-1, alltoall_group_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        local_rank = get_world_group().local_rank
        num = next((i for i, ranks in enumerate(group_ranks) if local_rank in ranks), None)
        _P_TP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=f"p_tp_{num}")

    # EP like group ranks
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            global_dp_size * global_pcp_size * global_tp_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]

    global _MC2
    _MC2 = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="mc2")

    if get_ascend_config().eplb_config.dynamic_eplb:
        global _DYNAMIC_EPLB
        _DYNAMIC_EPLB = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dynamic_eplb"
        )

    # Initialize fine-grained TP process groups on Ascend for four components:
    # 1. LM Head: output logits projection (`lmhead_tensor_parallel_size`)
    # 2. O Proj: attention output projection (`oproj_tensor_parallel_size`)
    # 3. Embedding: The token embedding table at the input of the model (`embedding_tensor_parallel_size`)
    # 4. MLP: feed-forward network in transformer blocks (`mlp_tensor_parallel_size`)
    _group_cache = {}

    def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
        if group_size is None:
            return None
        if group_size not in _group_cache:
            rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
            _group_cache[group_size] = pg

        return _group_cache[group_size]

    otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size

    global _OTP, _LMTP, _EMBED_TP, _MLP_TP

    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")


def model_parallel_initialized():
    return _MC2 is not None


def get_elastic_info() -> torch.Tensor | None:
    """Return the elastic_info tensor consumed by the MC2 token-dispatch kernel.

    ``None`` means fault tolerance is disabled and no elastic_info is passed to
    the dispatch op. The tensor is owned by ``ElasticInfoMask`` and rebuilt in
    place on every mask update, so the kernel always observes the latest
    dead-rank set.
    """
    mask = _ACTIVE_ELASTIC_INFO_MASK
    return mask.elastic_info if mask is not None else None


def get_active_elastic_info_mask() -> "ElasticInfoMask | None":
    """Return the process-wide ElasticInfoMask instance.

    ``None`` means fault tolerance is disabled (or the worker's FT state has
    not been set up yet). Consumed by ``_NpuAll2AllManager`` so the FT sentinel
    can use ``get_ep_all2all_manager().update_mask(...)`` / ``query_fault()``.
    """
    return _ACTIVE_ELASTIC_INFO_MASK


class ElasticInfoMask:
    """Ascend equivalent of the all2all rank mask used in fault tolerance.

    The FT scale-down masks dead EP ranks on the all2all manager
    (``mgr.update_mask(ep_rank, masked=True)``). Ascend's MC2 kernel has no
    such mask query, so the dead ranks are encoded in an ``elastic_info``
    tensor consumed by ``npu_moe_distribute_dispatch``. This class exposes the
    same ``update_mask`` / ``query_active_mask`` semantics on top of
    elastic_info.

    Tensor layout:
        ``[is_scaled_down, scaled_down_ep_size, share_expert_rank_num,
        num_physical_experts] + table1(ep_size) + table2(ep_size)``

    - ``table1[ep_rank]``: local (densified) EP rank after scale-down, ``-1``
      for dead ranks
    - ``table2[local_ep_rank]``: original EP rank of that local slot, ``-1``
      for invalid slots

    The surviving EP ranks are DENSIFIED (renumbered 0..k-1 with no holes),
    matching the MC2 kernel's expected elastic_info semantics. The EP device
    group itself keeps its original world size; only the kernel's rank
    mapping is compacted.
    """

    def __init__(
        self,
        ep_size: int,
        num_physical_experts: int,
        share_expert_rank_num: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.ep_size = ep_size
        self.num_physical_experts = num_physical_experts
        self.share_expert_rank_num = share_expert_rank_num
        self.device = device if device is not None else torch.device("cpu")
        self._masked: set[int] = set()
        # Derived elastic_info tensor consumed by the MC2 dispatch kernel;
        # rebuilt in place on every mask change (see ``_build``).
        self.elastic_info: torch.Tensor | None = None
        global _ACTIVE_ELASTIC_INFO_MASK
        _ACTIVE_ELASTIC_INFO_MASK = self
        self._build()

    def update_mask(self, ep_rank: int, masked: bool = True) -> None:
        """Mark an EP rank as dead (``masked=True``) or alive again."""
        if masked:
            self._masked.add(ep_rank)
        else:
            self._masked.discard(ep_rank)
        self._build()

    def set_num_physical_experts(self, num_physical_experts: int) -> None:
        """Update the physical-expert count (shrinks after scale-down).

        Passes the reduced expert count so the kernel's expert space matches
        the surviving ranks.
        """
        self.num_physical_experts = num_physical_experts
        self._build()

    def query_active_mask(self) -> list[int]:
        """Return a per-EP-rank mask (1 = alive, 0 = dead)."""
        return [0 if rank in self._masked else 1 for rank in range(self.ep_size)]

    def _build(self) -> None:
        is_scaled_down = 1 if self._masked else 0
        valid = [rank for rank in range(self.ep_size) if rank not in self._masked]
        scaled_down_ep_size = len(valid)
        base_config = torch.tensor(
            [is_scaled_down, scaled_down_ep_size, self.share_expert_rank_num, self.num_physical_experts],
            dtype=torch.int32,
            device=self.device,
        )
        # Densified mapping over the surviving ranks: the MC2 kernel expects
        # the surviving EP ranks to be renumbered 0..k-1 (no holes), with dead
        # ranks at -1.
        # table1[ep_rank] = local (densified) EP rank, -1 for dead ranks.
        table1 = torch.full((self.ep_size,), -1, dtype=torch.int32, device=self.device)
        table1[valid] = torch.arange(len(valid), dtype=torch.int32, device=self.device)
        # table2[local_ep_rank] = original EP rank, -1 for invalid slots.
        table2 = torch.full((self.ep_size,), -1, dtype=torch.int32, device=self.device)
        table2[: len(valid)] = torch.tensor(valid, dtype=torch.int32, device=self.device)
        elastic_info = torch.cat([base_config, table1, table2], dim=0).to(torch.int32)
        # Keep the tensor storage stable across rebuilds so references captured
        # by the token dispatcher (e.g. for graph capture) stay valid.
        if self.elastic_info is None:
            self.elastic_info = elastic_info
        else:
            assert self.elastic_info.shape == elastic_info.shape
            self.elastic_info.copy_(elastic_info)


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, "mc2 group is not initialized"
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, "mlp group is not initialized"
    return _MLP_TP


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, "output tensor parallel group is not initialized"
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP


def get_embed_tp_group() -> GroupCoordinator:
    assert _EMBED_TP is not None, "emtp group is not initialized"
    return _EMBED_TP


def get_p_tp_group() -> GroupCoordinator:
    assert _P_TP is not None, "distributed prefill tensor parallel group is not initialized"
    return _P_TP


def get_dynamic_eplb_group() -> GroupCoordinator:
    assert _DYNAMIC_EPLB is not None, "Dynamic eplb group is not initialized"
    return _DYNAMIC_EPLB


def destroy_ascend_model_parallel():
    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLP_TP
    if _MLP_TP:
        _MLP_TP.destroy()
    _MLP_TP = None

    global _LMTP
    if _LMTP:
        _LMTP.destroy()
    _LMTP = None

    global _EMBED_TP
    if _EMBED_TP:
        _EMBED_TP.destroy()
    _EMBED_TP = None

    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None

    global _P_TP
    if _P_TP:
        _P_TP.destroy()
    _P_TP = None

    global _DYNAMIC_EPLB
    if _DYNAMIC_EPLB:
        _DYNAMIC_EPLB.destroy()
    _DYNAMIC_EPLB = None


def get_global_rank(parallel_config: ParallelConfig | None = None) -> int:
    """Return a globally unique rank for the current worker across all parallel
     dimensions (TP/PP/CP/DP), compatible with both dense and MoE models.

     vLLM does not expose a single ready-to-use cross-DP global rank:
       - For dense models each DP rank is launched as an independent DP=1 engine,
         so ``data_parallel_rank`` is reset to 0 and ``get_world_group()`` only
         spans one replica (``rank_in_group`` is the local rank in the replica).
       - For MoE DP / external_launcher the world group spans all DP ranks, so
         ``rank_in_group`` already encodes the DP offset.

     ``data_parallel_index`` always keeps the true DP rank (it is never reset),
     and ``rank_in_group % replica_size`` yields the local rank within a replica
     in both cases, so the formula below is correct everywhere. It mirrors vLLM's
     own ``data_parallel_rank * world_size + rank`` (see
     vllm/distributed/parallel_state.py).

    Note: DCP (decode context parallel) reuses the TP NPUs and EP overlays
    TP/DP, so neither adds new ranks and they are intentionally excluded from
    ``replica_size``.
    """
    if parallel_config is None:
        parallel_config = get_current_vllm_config().parallel_config
    # Number of NPUs in a single DP replica (TP * PP * prefill-CP).
    replica_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    rank_in_replica = get_world_group().rank_in_group % replica_size
    return parallel_config.data_parallel_index * replica_size + rank_in_replica
