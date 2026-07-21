#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Fault injection patch for fault tolerance e2e testing.

This module wraps ``NPUModelRunner._sync_batch_across_dp`` to inject a
simulated RuntimeError after a configurable number of all_reduce iterations
on a target dp_rank.  It is gated by environment variables and is a no-op
when the variables are absent.

Because the NPUModelRunner class is defined in ``model_runner_v1.py``, which
triggers ``patch/worker/__init__.py`` during its own import (via a
``patch_draft_quarot`` import), this module must **not** import
``model_runner_v1`` at module level — that would create a circular import.
Instead, ``model_runner_v1.py`` calls :func:`inject_fault` at the end of the
file once the class is fully defined.

Environment variables
---------------------
VLLM_FAULT_INJECT_COUNT : int
    Number of all_reduce calls after which the fault is injected (0 = disabled).
VLLM_FAULT_INJECT_RANK : int
    Target dp_rank on which to raise the exception (default 0).
"""

import os

_INJECT_COUNT = int(os.environ.get("VLLM_FAULT_INJECT_COUNT", "0"))
_INJECT_RANK = int(os.environ.get("VLLM_FAULT_INJECT_RANK", "0"))
_counter = 0


def inject_fault(cls):
    """Replace cls._sync_batch_across_dp with a fault-injecting wrapper.

    Only applies the monkey-patch when ``VLLM_FAULT_INJECT_COUNT`` > 0.
    Call this after the NPUModelRunner class is fully defined.
    """
    if _INJECT_COUNT <= 0:
        return

    _original = cls._sync_batch_across_dp

    def _patched(self, *args, **kwargs):
        global _counter
        result = _original(self, *args, **kwargs)
        _counter += 1
        if _counter >= _INJECT_COUNT and self.dp_rank == _INJECT_RANK:
            _counter = 0
            raise RuntimeError(
                f"Simulated fault injection: dp_rank={self.dp_rank} "
                f"after {_INJECT_COUNT} all_reduce iterations"
            )
        return result

    cls._sync_batch_across_dp = _patched
