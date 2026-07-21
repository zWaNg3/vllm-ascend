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
"""E2E tests for the fault tolerance retry and scale_down flows.

These tests verify that the fault-tolerance system correctly detects,
pauses, recovers (retry), or reconfigures (scale_down) when a real fault
is injected during active inference on DP=2.
"""

import threading
import time

import psutil
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


def _get_ft_status(server: RemoteOpenAIServer, timeout: float = 10) -> dict:
    """Get fault tolerance engine status dict from the server."""
    resp = requests.get(
        server.url_for("fault_tolerance", "status"),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _wait_engine_status(
    server: RemoteOpenAIServer,
    engine_id: int,
    target_status: str,
    timeout: float = 120,
) -> None:
    """Poll /fault_tolerance/status until *engine_id* reaches *target_status*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _get_ft_status(server, timeout=10)
        engines = status.get("engines", [])
        for eng in engines:
            if eng["id"] == engine_id and eng["status"] == target_status:
                return
        time.sleep(1)
    raise TimeoutError(
        f"Engine {engine_id} did not reach '{target_status}' within {timeout}s "
        f"(last status: {status})"
    )


@wait_until_npu_memory_free()
def test_fault_tolerance_retry() -> None:
    """E2E retry: inject a RuntimeError after N all_reduce calls on rank 0.

    1. Start vllm serve (DP=2, FT enabled) with fault injection env vars.
    2. Run inference in a background thread to hit _sync_batch_across_dp.
    3. After VLLM_FAULT_INJECT_COUNT iterations, rank 0 throws RuntimeError.
    4. Fault-tolerant wrapper auto-pauses both engines.
    5. Send the retry instruction via the HTTP API.
    6. Verify inference works after recovery.
    """
    model = "Qwen/Qwen3-0.6B"
    port = get_open_port()
    FAULT_INJECT_COUNT = 10

    env_dict = {
        "VLLM_FAULT_INJECT_COUNT": str(FAULT_INJECT_COUNT),
        "VLLM_FAULT_INJECT_RANK": "0",
    }
    server_args = [
        "--max_model_len", "1024",
        "--max_num_seqs", "8",
        "--data-parallel-size", "2",
        "--enable-fault-tolerance",
        "--gloo_timeout_seconds", "15",
        "--port", str(port),
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        server_port=port,
        auto_port=False,
        env_dict=env_dict,
    ) as server:
        client = server.get_client()

        # Run inference in background to trigger the fault injection
        stop_flag = threading.Event()

        def _run_inference():
            while not stop_flag.is_set():
                try:
                    client.completions.create(
                        model=model,
                        prompt="Hello",
                        max_tokens=16,
                        temperature=0,
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_run_inference, daemon=True)
        t.start()

        # Wait for the injected fault to pause both engines
        _wait_engine_status(server, 0, "paused", timeout=120)
        _wait_engine_status(server, 1, "paused", timeout=30)

        stop_flag.set()
        t.join(timeout=10)

        # Recovery: retry
        resp = requests.post(
            server.url_for("fault_tolerance", "apply"),
            json={
                "instruction": "retry",
                "params": {"timeout": 60},
            },
        )
        assert resp.status_code == 200, f"Retry failed: {resp.text}"

        # Verify inference works after recovery
        resp = client.completions.create(
            model=model,
            prompt="Hello",
            max_tokens=16,
            temperature=0,
        )
        assert resp.choices[0].text, "Inference should work after retry"


@wait_until_npu_memory_free()
def test_fault_tolerance_scale_down() -> None:
    """E2E scale_down: kill worker 0 during inference then issue scale_down.

    1. Start vllm serve (DP=2, TP=1, FT enabled).
    2. Run inference in a background thread.
    3. Kill the VllmWorker-0 OS process.
    4. The remaining worker's all_reduce times out → auto-pause.
    5. Send the scale_down instruction via the HTTP API.
    6. Verify inference works on the remaining DP rank.
    """
    model = "Qwen/Qwen3-30B-A3B"
    port = get_open_port()

    server_args = [
        "--max_model_len", "8192",
        "--tensor_parallel_size", "1",
        "--data-parallel-size", "2",
        "--enable-fault-tolerance",
        "--enable-expert-parallel",
        "--gloo_timeout_seconds", "30",
        "--port", str(port),
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        server_port=port,
        auto_port=False,
    ) as server:
        client = server.get_client()

        # Run inference in background to keep workers active
        stop_flag = threading.Event()

        def _run_inference():
            while not stop_flag.is_set():
                try:
                    client.completions.create(
                        model=model,
                        prompt="What is AI?",
                        max_tokens=32,
                        temperature=0,
                    )
                except Exception:
                    pass

        t = threading.Thread(target=_run_inference, daemon=True)
        t.start()
        time.sleep(5)

        # Kill worker 0: match OS process name "VllmWorker-0"
        # (vllm multiproc_executor.py L672: name=f"VllmWorker-{rank}")
        vllm_proc = server.proc
        children = psutil.Process(vllm_proc.pid).children(recursive=True)
        workers = [c for c in children if c.name().startswith("VllmWorker-")]
        worker0 = next((c for c in workers if c.name() == "VllmWorker-0"), None)
        assert worker0 is not None, "Could not find VllmWorker-0 process"
        worker0.kill()

        # Wait for the fault to be detected and remaining engine to pause
        _wait_engine_status(server, 0, "dead", timeout=180)
        _wait_engine_status(server, 1, "paused", timeout=30)

        stop_flag.set()
        t.join(timeout=10)

        # Recovery: scale down — exclude the dead rank
        resp = requests.post(
            server.url_for("fault_tolerance", "apply"),
            json={
                "instruction": "scale_down",
                "params": {
                    "timeout": 120,
                    "exclude_dp_ranks": [0],
                },
            },
        )
        assert resp.status_code == 200, f"Scale down failed: {resp.text}"

        # Verify inference works after scale down (on remaining DP rank)
        resp = client.completions.create(
            model=model,
            prompt="What is AI?",
            max_tokens=32,
            temperature=0,
        )
        assert resp.choices[0].text, "Inference should work after scale down"
