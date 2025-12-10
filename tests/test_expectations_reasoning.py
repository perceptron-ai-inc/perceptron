from perceptron.expectations import resolve_structured_expectation
from perceptron.dsl.perceive import _is_thinking_model


def test_resolve_reasoning_expectation_allows_think():
    resolved, allow_multiple = resolve_structured_expectation("think", context="expects value")
    assert resolved == "think"
    assert allow_multiple is False


def test_is_thinking_model_detects_qwen_and_thinking_tokens():
    assert _is_thinking_model("qwen3-vl-235b-a22b-thinking") is True
    assert _is_thinking_model("my-cool-thinking-model") is True
    assert _is_thinking_model("QWEN3-XL") is True
    assert _is_thinking_model("isaac-0.1") is False
    assert _is_thinking_model(None) is False
    assert _is_thinking_model(123) is False
