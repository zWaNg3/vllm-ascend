# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Recovery metrics for fault-tolerance retry and scale-down.

One round's phase timings and reload volumes, recorded on the worker that
runs the recovery, so a scale-down can report the phase breakdown used for
the upstream fault-tolerance test results: fault detection, group reinit,
expert redistribution, weight reload and the dummy-batch check.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class FtRecoveryMetrics:
    """Timings and volumes of one retry or scale_down round."""

    instruction: str
    started_at: float = field(default_factory=time.perf_counter)
    # Phase name -> accumulated seconds. Nested phases are allowed and are
    # reported separately, e.g. "reload" and its "reload_scan" sub-phase.
    phases: dict[str, float] = field(default_factory=dict)
    # Counter name -> accumulated value, e.g. reassignments or bytes read.
    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Record the wall-clock time of the wrapped block under ``name``."""
        # Register on entry so nested phases stay next to their parent in the
        # report; accumulation happens on exit.
        self.phases.setdefault(name, 0.0)
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.add_phase(name, time.perf_counter() - started_at)

    def add_phase(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    def add_value(self, name: str, value: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + value

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self.started_at


def format_metrics(phases: dict[str, float], values: dict[str, float]) -> str:
    """One log-friendly line: ``a=1.23 s, b=0.45 s; x=7, y=1.20``."""
    timings = ", ".join(f"{name}={seconds:.2f} s" for name, seconds in phases.items())
    counts = ", ".join(f"{name}={_format_value(value)}" for name, value in values.items())
    return f"{timings}; {counts}" if counts else timings


def _format_value(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"
