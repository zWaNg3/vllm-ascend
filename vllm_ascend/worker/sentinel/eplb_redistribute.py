# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-specific helpers for the fault-tolerance scale-down flow.

The actual expert redistribution and weight reload for the model runner V2
reuse the helpers in ``vllm.v1.worker.sentinel.eplb_redistribute``; this
module only holds the Ascend-specific MC2 rank mapping.
"""


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
