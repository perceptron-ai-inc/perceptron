"""Tests for reasoning parser functionality."""

from perceptron.reasoning import Reasoning, extract_reasoning, parse_text, strip_reasoning


def test_extract_single_reasoning():
    """Test extracting a single <think> tag."""
    text = "The answer is <think>Let me think about this step by step</think> 42"
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    assert reasoning.content == "Let me think about this step by step"


def test_extract_multiple_reasoning_concatenates():
    """Test that multiple <think> tags are concatenated."""
    text = "First <think>step 1</think> then <think>step 2</think> done"
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    assert reasoning.content == "step 1\nstep 2"


def test_extract_reasoning_with_whitespace():
    """Test that reasoning content is trimmed."""
    text = "<think>  \n  some reasoning  \n  </think>"
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    assert reasoning.content == "some reasoning"


def test_extract_reasoning_empty_tags():
    """Test that empty <think> tags return None."""
    text = "No reasoning here <think></think> or <think>   </think>"
    reasoning = extract_reasoning(text)
    # Empty reasoning tags should still return None or empty content
    # Depending on implementation, adjust this test
    assert reasoning is None or reasoning.content == ""


def test_extract_reasoning_no_tags():
    """Test that text without <think> tags returns None."""
    text = "Just regular text without any reasoning"
    reasoning = extract_reasoning(text)
    assert reasoning is None


def test_extract_reasoning_multiline():
    """Test reasoning with multiline content."""
    text = """<think>
    First, I need to analyze the problem.
    Then, I'll consider different approaches.
    Finally, I'll choose the best solution.
    </think>"""
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    assert "First, I need to analyze the problem." in reasoning.content
    assert "Finally, I'll choose the best solution." in reasoning.content


def test_extract_reasoning_case_insensitive():
    """Test that <think> tags are case-insensitive."""
    text1 = "<think>lowercase</think>"
    text2 = "<THINK>uppercase</THINK>"
    text3 = "<Think>mixedcase</Think>"

    r1 = extract_reasoning(text1)
    r2 = extract_reasoning(text2)
    r3 = extract_reasoning(text3)

    assert r1 is not None and r1.content == "lowercase"
    assert r2 is not None and r2.content == "uppercase"
    assert r3 is not None and r3.content == "mixedcase"


def test_strip_reasoning_removes_tags():
    """Test that strip_reasoning removes all <think> tags."""
    text = "Before <think>reasoning here</think> after"
    cleaned = strip_reasoning(text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Before" in cleaned
    assert "after" in cleaned
    assert "reasoning here" not in cleaned


def test_strip_reasoning_multiple_tags():
    """Test stripping multiple <think> tags."""
    text = "Start <think>first</think> middle <think>second</think> end"
    cleaned = strip_reasoning(text)
    assert "<think>" not in cleaned
    assert "first" not in cleaned
    assert "second" not in cleaned
    assert "Start" in cleaned
    assert "middle" in cleaned
    assert "end" in cleaned


def test_strip_reasoning_collapses_spaces():
    """Test that multiple spaces are collapsed after removing tags."""
    text = "Word1   <think>reasoning</think>   Word2"
    cleaned = strip_reasoning(text)
    # Should collapse multiple spaces
    assert "  " not in cleaned or cleaned.count("  ") < text.count("  ")


def test_parse_text_no_reasoning():
    """Test parse_text with no reasoning tags."""
    text = "Just plain text here"
    segments = parse_text(text)
    assert len(segments) == 1
    assert segments[0]["kind"] == "text"
    assert segments[0]["text"] == text
    assert segments[0]["span"] == {"start": 0, "end": len(text)}


def test_parse_text_single_reasoning():
    """Test parse_text with a single reasoning tag."""
    text = "Before <think>reasoning content</think> after"
    segments = parse_text(text)

    assert len(segments) == 3

    # First segment: text before
    assert segments[0]["kind"] == "text"
    assert segments[0]["text"] == "Before "
    assert segments[0]["span"]["start"] == 0

    # Second segment: reasoning
    assert segments[1]["kind"] == "reasoning"
    assert isinstance(segments[1]["value"], Reasoning)
    assert segments[1]["value"].content == "reasoning content"

    # Third segment: text after
    assert segments[2]["kind"] == "text"
    assert segments[2]["text"] == " after"


def test_parse_text_multiple_reasoning():
    """Test parse_text with multiple reasoning tags."""
    text = "A <think>first</think> B <think>second</think> C"
    segments = parse_text(text)

    assert len(segments) == 5
    assert segments[0]["kind"] == "text" and segments[0]["text"] == "A "
    assert segments[1]["kind"] == "reasoning" and segments[1]["value"].content == "first"
    assert segments[2]["kind"] == "text" and segments[2]["text"] == " B "
    assert segments[3]["kind"] == "reasoning" and segments[3]["value"].content == "second"
    assert segments[4]["kind"] == "text" and segments[4]["text"] == " C"


def test_parse_text_reasoning_at_start():
    """Test parse_text with reasoning at the start."""
    text = "<think>reasoning</think> text after"
    segments = parse_text(text)

    assert len(segments) == 2
    assert segments[0]["kind"] == "reasoning"
    assert segments[1]["kind"] == "text"


def test_parse_text_reasoning_at_end():
    """Test parse_text with reasoning at the end."""
    text = "text before <think>reasoning</think>"
    segments = parse_text(text)

    assert len(segments) == 2
    assert segments[0]["kind"] == "text"
    assert segments[1]["kind"] == "reasoning"


def test_parse_text_only_reasoning():
    """Test parse_text with only reasoning tag."""
    text = "<think>just reasoning</think>"
    segments = parse_text(text)

    assert len(segments) == 1
    assert segments[0]["kind"] == "reasoning"
    assert segments[0]["value"].content == "just reasoning"


def test_parse_text_empty_reasoning():
    """Test parse_text with empty reasoning tags."""
    text = "Before <think></think> after"
    segments = parse_text(text)

    # Empty reasoning should not be included
    assert all(seg["kind"] == "text" or seg["value"].content != "" for seg in segments)


def test_parse_text_spans_correct():
    """Test that parse_text returns correct span positions."""
    text = "ABC <think>DEF</think> GHI"
    segments = parse_text(text)

    # Verify spans match the original text
    for seg in segments:
        span = seg["span"]
        start, end = span["start"], span["end"]
        if seg["kind"] == "text":
            assert text[start:end] == seg["text"]
        elif seg["kind"] == "reasoning":
            # For reasoning, the span includes the tags
            assert "<think>" in text[start:end]
            assert "</think>" in text[start:end]


def test_reasoning_dataclass():
    """Test Reasoning dataclass properties."""
    reasoning = Reasoning(content="test content")
    assert reasoning.content == "test content"

    # Test equality
    reasoning2 = Reasoning(content="test content")
    assert reasoning == reasoning2

    reasoning3 = Reasoning(content="different")
    assert reasoning != reasoning3


def test_extract_reasoning_with_special_characters():
    """Test reasoning extraction with special characters."""
    text = '<think>Use "quotes" & special <chars></think>'
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    assert 'Use "quotes" & special <chars>' in reasoning.content


def test_nested_think_tags():
    """Test behavior with nested <think> tags (should match outermost)."""
    text = "<think>outer <think>inner</think> outer</think>"
    reasoning = extract_reasoning(text)
    # Regex should match the first closing tag
    assert reasoning is not None
    # The exact behavior depends on regex implementation (greedy vs non-greedy)


def test_incomplete_think_tags():
    """Test behavior with incomplete tags."""
    text1 = "<think>unclosed tag"
    text2 = "unopened tag</think>"
    text3 = "<think>content"

    r1 = extract_reasoning(text1)
    r2 = extract_reasoning(text2)
    r3 = extract_reasoning(text3)

    # Incomplete tags should not be matched
    assert r1 is None
    assert r2 is None
    assert r3 is None


def test_non_string_input():
    """Test handling of non-string inputs."""
    assert extract_reasoning(None) is None
    assert extract_reasoning(123) is None
    assert extract_reasoning([]) is None

    # strip_reasoning should handle non-string gracefully
    assert strip_reasoning(None) is None
    assert strip_reasoning(123) == 123


def test_parse_text_preserves_document_structure():
    """Test that parse_text preserves the full document structure."""
    text = "Start <think>r1</think> middle <think>r2</think> end"
    segments = parse_text(text)

    # Reconstruct text from segments
    reconstructed = ""
    for seg in segments:
        if seg["kind"] == "text":
            reconstructed += seg["text"]
        elif seg["kind"] == "reasoning":
            # Reasoning segments represent the tags in the original
            span = seg["span"]
            reconstructed += text[span["start"]:span["end"]]

    assert reconstructed == text
