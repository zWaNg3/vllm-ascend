# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fault-tolerance recovery metrics used by the scale-down report."""

import time

import pytest

from vllm_ascend.worker.sentinel.ft_metrics import FtRecoveryMetrics, format_metrics


def test_phase_accumulates_and_total_covers_it():
    metrics = FtRecoveryMetrics("scale_down")
    with metrics.phase("reload"):
        time.sleep(0.001)
    with metrics.phase("reload"):
        time.sleep(0.001)
    with metrics.phase("dummy_batch"):
        pass

    assert set(metrics.phases) == {"reload", "dummy_batch"}
    assert metrics.phases["reload"] > 0
    assert metrics.phases["dummy_batch"] >= 0
    assert metrics.total_s >= metrics.phases["reload"]


def test_phase_records_when_the_block_fails():
    metrics = FtRecoveryMetrics("retry")
    with pytest.raises(RuntimeError), metrics.phase("group_reinit"):
        raise RuntimeError("gloo timeout")

    assert metrics.phases["group_reinit"] > 0


def test_values_accumulate_and_are_formatted():
    metrics = FtRecoveryMetrics("scale_down")
    metrics.add_value("reassignments", 2)
    metrics.add_value("reassignments", 3)
    metrics.add_value("checkpoint_read_mib", 1.5)

    assert metrics.values["reassignments"] == 5
    line = format_metrics({"reload": 1.234}, metrics.values)
    assert "reload=1.23 s" in line
    assert "reassignments=5" in line
    assert "checkpoint_read_mib=1.50" in line


def test_format_without_values_has_no_trailing_separator():
    assert format_metrics({"reset_device": 0.5}, {}) == "reset_device=0.50 s"
