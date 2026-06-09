"""Tests for the Nous-GPUCLOUD-3/4 non-agentic warning detector.

Prior to this check, the warning fired on any model whose name contained
``"gpucloud"`` anywhere (case-insensitive). That false-positived on unrelated
local Modelfiles such as ``gpucloud-brain:qwen3-14b-ctx16k`` — a tool-capable
Qwen3 wrapper that happens to live under the "gpucloud" tag namespace.

``is_nous_gpucloud_non_agentic`` should only match the actual Nous Research
GPUCLOUD-3 / GPUCLOUD-4 chat family.
"""

from __future__ import annotations

import pytest

from gpucloud_cli.model_switch import (
    _GPUCLOUD_MODEL_WARNING,
    _check_gpucloud_model_warning,
    is_nous_gpucloud_non_agentic,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "NousResearch/GPUCLOUD-3-Llama-3.1-70B",
        "NousResearch/GPUCLOUD-3-Llama-3.1-405B",
        "gpucloud-3",
        "GPUCLOUD-3",
        "gpucloud-4",
        "gpucloud-4-405b",
        "gpucloud_4_70b",
        "openrouter/gpucloud3:70b",
        "openrouter/nousresearch/gpucloud-4-405b",
        "NousResearch/GPUCLOUD3",
        "gpucloud-3.1",
    ],
)
def test_matches_real_nous_gpucloud_chat_models(model_name: str) -> None:
    assert is_nous_gpucloud_non_agentic(model_name), (
        f"expected {model_name!r} to be flagged as Nous GPUCLOUD 3/4"
    )
    assert _check_gpucloud_model_warning(model_name) == _GPUCLOUD_MODEL_WARNING


@pytest.mark.parametrize(
    "model_name",
    [
        # Kyle's local Modelfile — qwen3:14b under a custom tag
        "gpucloud-brain:qwen3-14b-ctx16k",
        "gpucloud-brain:qwen3-14b-ctx32k",
        "gpucloud-honcho:qwen3-8b-ctx8k",
        # Plain unrelated models
        "qwen3:14b",
        "qwen3-coder:30b",
        "qwen2.5:14b",
        "claude-opus-4-6",
        "anthropic/claude-sonnet-4.5",
        "gpt-5",
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
        "deepseek-chat",
        # Non-chat GPUCLOUD models we don't warn about
        "gpucloud-llm-2",
        "gpucloud2-pro",
        "nous-gpucloud-2-mistral",
        # Edge cases
        "",
        "gpucloud",  # bare "gpucloud" isn't the 3/4 family
        "gpucloud-brain",
        "brain-gpucloud-3-impostor",  # "3" not preceded by /: boundary
    ],
)
def test_does_not_match_unrelated_models(model_name: str) -> None:
    assert not is_nous_gpucloud_non_agentic(model_name), (
        f"expected {model_name!r} NOT to be flagged as Nous GPUCLOUD 3/4"
    )
    assert _check_gpucloud_model_warning(model_name) == ""


def test_none_like_inputs_are_safe() -> None:
    assert is_nous_gpucloud_non_agentic("") is False
    # Defensive: the helper shouldn't crash on None-ish falsy input either.
    assert _check_gpucloud_model_warning("") == ""
