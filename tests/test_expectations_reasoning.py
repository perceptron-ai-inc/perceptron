from perceptron.expectations import REASONING_HINT, resolve_structured_expectation, expectation_hint_text


def test_resolve_reasoning_expectation_allows_think():
    resolved, allow_multiple = resolve_structured_expectation("think", context="expects value")
    assert resolved == "think"
    assert allow_multiple is False


def test_expectation_hint_returns_think_hint():
    assert expectation_hint_text("think") == REASONING_HINT
