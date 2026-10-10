# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.patch_v2 import patch_dspark


@dataclass
class FakeEplbConfig:
    num_redundant_experts: int = 0
    use_async: bool = False


@dataclass
class FakeParallelConfig:
    enable_eplb: bool = False
    eplb_config: FakeEplbConfig = field(default_factory=FakeEplbConfig)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_elastic_ep: bool = False


def _upstream_get_dspark_parallel_config(parallel_config, tensor_parallel_size):
    """Mirror vllm.v1.worker.gpu.spec_decode.dspark.utils._get_dspark_parallel_config."""
    return patch_dspark.replace(
        parallel_config,
        pipeline_parallel_size=1,
        tensor_parallel_size=tensor_parallel_size,
        enable_eplb=False,
        eplb_config=patch_dspark.replace(parallel_config.eplb_config, num_redundant_experts=0),
        enable_elastic_ep=False,
    )


@pytest.fixture
def upstream_parallel_config(monkeypatch):
    monkeypatch.setattr(patch_dspark, "_original_get_dspark_parallel_config", _upstream_get_dspark_parallel_config)


def test_same_checkpoint_draft_inherits_the_targets_eplb_settings(upstream_parallel_config):
    eplb_config = FakeEplbConfig(num_redundant_experts=128, use_async=True)
    target = FakeParallelConfig(
        enable_eplb=True,
        eplb_config=eplb_config,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
    )

    draft = patch_dspark._get_dspark_parallel_config_inheriting_eplb(target, 1)

    # The draft keeps its own TP/PP shape ...
    assert draft.tensor_parallel_size == 1
    assert draft.pipeline_parallel_size == 1
    # ... but the expert geometry EplbState validates must match the target's.
    assert draft.enable_eplb is True
    assert draft.eplb_config is eplb_config


def test_draft_config_is_left_to_upstream_when_the_target_has_no_eplb(upstream_parallel_config):
    target = FakeParallelConfig(enable_eplb=False, eplb_config=FakeEplbConfig(num_redundant_experts=0))

    draft = patch_dspark._get_dspark_parallel_config_inheriting_eplb(target, 1)

    assert draft.enable_eplb is False
    assert draft.eplb_config.num_redundant_experts == 0


@pytest.mark.parametrize("same_checkpoint", [True, False])
def test_redirect_is_scoped_to_same_checkpoint_drafts(monkeypatch, same_checkpoint):
    target_name = "deepseek-ai/DeepSeek-V4"
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(model=target_name),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(model=target_name if same_checkpoint else "qwen3-dspark"),
        ),
    )
    original = patch_dspark.dspark_utils._get_dspark_parallel_config
    installed = {}

    def load(target, received_config):
        assert received_config is config
        installed["redirect"] = patch_dspark.dspark_utils._get_dspark_parallel_config
        return target

    monkeypatch.setattr(patch_dspark, "_original_load_dspark_model", load)

    target = object()
    assert patch_dspark._load_dspark_model_with_target_quant(target, config) is target

    if same_checkpoint:
        assert installed["redirect"] is patch_dspark._get_dspark_parallel_config_inheriting_eplb
    else:
        assert installed["redirect"] is original
    assert patch_dspark.dspark_utils._get_dspark_parallel_config is original
