#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.distributed.parallel_state import get_active_elastic_info_mask


class _NpuAll2AllManager:
    """Ascend all2all_manager backed by ``ElasticInfoMask``.

    Ascend's MC2 kernel has no kernel-side all2all mask (unlike DeepEP/NixlEP),
    so the FT mask lives in the worker-side ``ElasticInfoMask`` (elastic_info)
    state. This manager exposes the upstream ``All2AllManagerBase`` interface
    (``update_mask`` / ``query_active_mask`` / ``query_fault`` /
    ``clean_buffers``) so the sentinel can go through ``get_ep_all2all_manager``
    exactly like the GPU path in PR #46370.
    """

    def __init__(self) -> None:
        # Baseline recorded at the last recovery point; ``None`` means unknown.
        self._last_mask: torch.Tensor | None = None

    @property
    def support_fault_tolerance(self) -> bool:
        return True

    def _get_mask(self):
        mask = get_active_elastic_info_mask()
        if mask is None:
            raise RuntimeError(
                "ElasticInfoMask is not initialized: fault tolerance is not "
                "enabled or the worker's FT state was not set up."
            )
        return mask

    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Set the mask for a specific EP rank (used during scale-down)."""
        self._get_mask().update_mask(rank, masked)

    def query_active_mask(self) -> torch.Tensor:
        """Return an int32 tensor where 0 = live, 1 = dead.

        This follows the upstream all2all mask convention (see
        ``All2AllManagerBase.query_active_mask``), which is the inverse of the
        ``ElasticInfoMask.query_active_mask`` (1 = alive).

        The tensor is built on the CPU on purpose: ``query_active_mask`` /
        ``query_fault`` are called while a fault is being probed, when the NPU
        may be hung, so they must not touch the device (any device op would
        fail with an ACL stream synchronize error).
        """
        mask = self._get_mask()
        return torch.tensor(
            [0 if alive else 1 for alive in mask.query_active_mask()],
            dtype=torch.int32,
        )

    def query_fault(self) -> torch.Tensor:
        """Return a scalar bool tensor, True if a new fault appeared.

        Compares the current mask against the baseline recorded at the last
        recovery point (all live until ``clean_buffers`` resets it).
        """
        current = self.query_active_mask()
        if self._last_mask is None or self._last_mask.shape != current.shape:
            self._last_mask = torch.zeros_like(current)
        return (current != self._last_mask).any()

    def clean_buffers(self) -> None:
        """Reset the ``query_fault`` baseline.

        Unlike DeepEP/NixlEP there is no kernel-side mask buffer to clean: the
        elastic_info tensor is passed fresh to the MC2 dispatch op on every
        call, so the mask itself is intentionally left untouched here.
        """
        self._last_mask = None


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        # TODO(hz): Refer to CudaCommunicator's implementation to integrate PyHcclCommunicator
        # init device according to rank
        self.device = torch.npu.current_device()

        # For compatibility (mainly for reusing graph capturing code in vllm),
        # init custom all-reduce implementation interface as in CUDACommunicator.
        self.ca_comm = None
        self.all2all_manager = _NpuAll2AllManager()

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
            output_list = []
            tensor_shape_base = input_list[self.rank].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

        else:
            input_list = [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]
            output_list = [torch.empty_like(input_list[i]) for i in range(self.world_size)]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor
