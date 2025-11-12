"""Tests for reasoning parser functionality using structured test cases."""

import pytest
from dataclasses import dataclass
from typing import Any

from perceptron.reasoning import Reasoning, extract_reasoning, parse_text, strip_reasoning


@dataclass
class ReasoningTestCase:
    """Test case structure for reasoning parser tests.

    Each test case defines the exact expected outputs for all three formats:
    1. Text with <think> tags
    2. Chat completion with reasoning_content
    3. Streaming chat completion chunks
    """
    name: str
    content: str  # Normal text content
    reasoning: str | None  # Raw reasoning content (may have whitespace)

    # Expected outputs
    expected_text_with_tags: str  # Exact text with <think> tags
    expected_stripped_text: str  # Expected text after stripping reasoning
    expected_reasoning: Reasoning | None  # Expected Reasoning object after extraction
    expected_streaming_chunks: list[dict[str, Any]]  # Expected streaming chunks


# Test cases covering various scenarios
REASONING_TEST_CASES = [
    ReasoningTestCase(
        name="single_reasoning",
        content="The answer is 42",
        reasoning="Let me think about this step by step",
        expected_text_with_tags="The answer is 42 <think>Let me think about this step by step</think>",
        expected_stripped_text="The answer is 42",
        expected_reasoning=Reasoning(content="Let me think about this step by step", parsed_segments=None),
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "The answer", "reasoning_content": "Let me think about"}}]},
            {"choices": [{"delta": {"content": " is 42", "reasoning_content": " this step by step"}}]},
        ],
    ),
    ReasoningTestCase(
        name="no_reasoning",
        content="Just regular text without any reasoning",
        reasoning=None,
        expected_text_with_tags="Just regular text without any reasoning",
        expected_stripped_text="Just regular text without any reasoning",
        expected_reasoning=None,
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "Just regular text without any reasoning"}}]},
        ],
    ),
    ReasoningTestCase(
        name="empty_reasoning",
        content="Text with empty reasoning",
        reasoning="",
        expected_text_with_tags="Text with empty reasoning",
        expected_stripped_text="Text with empty reasoning",
        expected_reasoning=None,
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "Text with empty reasoning"}}]},
        ],
    ),
    ReasoningTestCase(
        name="multiline_reasoning",
        content="The solution is correct",
        reasoning="First, I need to analyze the problem.\nThen, I'll consider different approaches.\nFinally, I'll choose the best solution.",
        expected_text_with_tags="The solution is correct <think>First, I need to analyze the problem.\nThen, I'll consider different approaches.\nFinally, I'll choose the best solution.</think>",
        expected_stripped_text="The solution is correct",
        expected_reasoning=Reasoning(
            content="First, I need to analyze the problem.\nThen, I'll consider different approaches.\nFinally, I'll choose the best solution.",
            parsed_segments=None
        ),
        expected_streaming_chunks=[
            {"choices": [{"delta": {
                "content": "The solution ",
                "reasoning_content": "First, I need to analyze the problem.\nThen, I'll consi"
            }}]},
            {"choices": [{"delta": {
                "content": "is correct",
                "reasoning_content": "der different approaches.\nFinally, I'll choose the best solution."
            }}]},
        ],
    ),
    ReasoningTestCase(
        name="reasoning_with_whitespace",
        content="Answer found",
        reasoning="  \n  some reasoning  \n  ",
        expected_text_with_tags="Answer found <think>  \n  some reasoning  \n  </think>",
        expected_stripped_text="Answer found",
        expected_reasoning=Reasoning(content="some reasoning", parsed_segments=None),  # Trimmed
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "Answer", "reasoning_content": "  \n  some"}}]},
            {"choices": [{"delta": {"content": " found", "reasoning_content": " reasoning  \n  "}}]},
        ],
    ),
    ReasoningTestCase(
        name="special_characters",
        content="Result: correct",
        reasoning='Use "quotes" & special <chars>',
        expected_text_with_tags='Result: correct <think>Use "quotes" & special <chars></think>',
        expected_stripped_text="Result: correct",
        expected_reasoning=Reasoning(content='Use "quotes" & special <chars>', parsed_segments=None),
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "Result:", "reasoning_content": 'Use "quote'}}]},
            {"choices": [{"delta": {"content": " correct", "reasoning_content": 's" & special <chars>'}}]},
        ],
    ),
    ReasoningTestCase(
        name="reasoning_with_all_point_types",
        content="Found multiple elements",
        reasoning="Object at <point>(10,20)</point>, region <point_box>(30,40) (50,60)</point_box>, shape <polygon>(0,0) (10,0) (5,10)</polygon>, and items <collection><point mention=\"A\">(70,80)</point><point mention=\"B\">(90,100)</point></collection>",
        expected_text_with_tags='Found multiple elements <think>Object at <point>(10,20)</point>, region <point_box>(30,40) (50,60)</point_box>, shape <polygon>(0,0) (10,0) (5,10)</polygon>, and items <collection><point mention="A">(70,80)</point><point mention="B">(90,100)</point></collection></think>',
        expected_stripped_text="Found multiple elements",
        # For this test case, we'll validate parsed_segments structure separately
        # since it contains complex nested objects (Point, Box, Polygon, Collection)
        expected_reasoning=None,  # Sentinel value - will be validated specially
        expected_streaming_chunks=[
            {"choices": [{"delta": {"content": "Found multiple", "reasoning_content": "Object at <point>(10,20)</point>, region <point_box>(30,40) "}}]},
            {"choices": [{"delta": {"content": " elements", "reasoning_content": '(50,60)</point_box>, shape <polygon>(0,0) (10,0) (5,10)</polygon>, and items <collection><point mention="A">(70,80)</point><point mention="B">(90,100)</point></collection>'}}]},
        ],
    ),
]


@pytest.mark.parametrize("test_case", REASONING_TEST_CASES, ids=lambda tc: tc.name)
def test_extract_reasoning_from_think_tags(test_case: ReasoningTestCase):
    """Test extracting reasoning from <think> tags."""
    # Skip the points test case here - it has a dedicated test
    if test_case.name == "reasoning_with_all_point_types":
        pytest.skip("Tested separately with point parsing enabled")

    text = test_case.expected_text_with_tags
    reasoning = extract_reasoning(text, parse_points=False)

    if test_case.expected_reasoning:
        assert reasoning is not None
        assert reasoning == test_case.expected_reasoning
    else:
        assert reasoning is None


@pytest.mark.parametrize("test_case", REASONING_TEST_CASES, ids=lambda tc: tc.name)
def test_strip_reasoning_from_think_tags(test_case: ReasoningTestCase):
    """Test stripping reasoning from <think> tags."""
    text = test_case.expected_text_with_tags
    cleaned = strip_reasoning(text)

    # Should not contain think tags
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    # Should match expected stripped text
    assert cleaned == test_case.expected_stripped_text


@pytest.mark.parametrize("test_case", REASONING_TEST_CASES, ids=lambda tc: tc.name)
def test_parse_text_with_think_tags(test_case: ReasoningTestCase):
    """Test parsing text with <think> tags into segments."""
    # Skip the points test case here - it has a dedicated test
    if test_case.name == "reasoning_with_all_point_types":
        pytest.skip("Tested separately with point parsing enabled")

    text = test_case.expected_text_with_tags
    segments = parse_text(text, parse_points=False)

    # Extract reasoning segments
    reasoning_segments = [seg for seg in segments if seg["kind"] == "reasoning"]

    if test_case.expected_reasoning:
        assert len(reasoning_segments) > 0
        # Concatenate all reasoning content and compare
        combined_reasoning = Reasoning(
            content="\n".join(seg["value"].content for seg in reasoning_segments),
            parsed_segments=None
        )
        assert combined_reasoning == test_case.expected_reasoning
    else:
        assert len(reasoning_segments) == 0


@pytest.mark.parametrize("test_case", REASONING_TEST_CASES, ids=lambda tc: tc.name)
def test_stream_processor_with_api_reasoning_content(test_case: ReasoningTestCase):
    """Test StreamProcessor handles reasoning_content from streaming API."""
    # Skip the points test case here - it has a dedicated test
    if test_case.name == "reasoning_with_all_point_types":
        pytest.skip("Tested separately with point parsing enabled")

    from perceptron.client import _ClientCore, _StreamProcessor

    client_core = _ClientCore()
    processor = _StreamProcessor(
        client_core=client_core,
        expects=None,
        parse_points=False,
        parse_reasoning=True,
        max_buffer_bytes=None,
    )

    # Process the exact expected streaming chunks
    for chunk in test_case.expected_streaming_chunks:
        processor.handle_payload(chunk)

    # Finalize and check result
    final_event = processor.finalize()
    result = final_event["result"]

    # Check text content matches expected
    assert result["text"] == test_case.content

    # Check reasoning content
    if test_case.expected_reasoning:
        assert result["reasoning"] is not None
        # Compare the Reasoning objects
        assert result["reasoning"] == test_case.expected_reasoning
    else:
        assert result["reasoning"] is None


@pytest.mark.parametrize("test_case", REASONING_TEST_CASES, ids=lambda tc: tc.name)
def test_combined_api_and_tag_reasoning(test_case: ReasoningTestCase):
    """Test combining API reasoning_content with <think> tags."""
    from perceptron.client import _ClientCore, _StreamProcessor

    # Skip the points test case here - it has a dedicated test
    if test_case.name == "reasoning_with_all_point_types":
        pytest.skip("Tested separately with point parsing enabled")

    # Skip if no reasoning in test case
    if not test_case.expected_reasoning:
        pytest.skip("Test case has no reasoning to combine")

    client_core = _ClientCore()
    processor = _StreamProcessor(
        client_core=client_core,
        expects=None,
        parse_points=False,
        parse_reasoning=True,
        max_buffer_bytes=None,
    )

    # Create payload with both API reasoning and <think> tags in content
    tag_reasoning = "tag reasoning"
    content_with_tags = f"{test_case.content} <think>{tag_reasoning}</think>"
    # Expected combined reasoning object
    expected_combined = Reasoning(
        content=f"{test_case.expected_reasoning.content}\n{tag_reasoning}",
        parsed_segments=None
    )

    payload = {
        "choices": [{
            "delta": {
                "content": content_with_tags,
                "reasoning_content": test_case.reasoning
            }
        }]
    }

    processor.handle_payload(payload)
    final_event = processor.finalize()
    result = final_event["result"]

    # Both types of reasoning should be present and combined correctly
    assert result["reasoning"] is not None
    assert result["reasoning"] == expected_combined

    # Text should have <think> tags removed
    assert result["text"] == test_case.content
    assert "<think>" not in result["text"]


def test_reasoning_with_all_point_types_comprehensive():
    """Comprehensive test for reasoning containing all point types (points, boxes, polygons, collections)."""
    # Get the test case
    test_case = next(tc for tc in REASONING_TEST_CASES if tc.name == "reasoning_with_all_point_types")

    # Test 1: Extract reasoning from <think> tags with point parsing
    text = test_case.expected_text_with_tags
    reasoning = extract_reasoning(text, parse_points=True)

    assert reasoning is not None
    assert "Object at" in reasoning.content
    assert reasoning.parsed_segments is not None

    # Validate all point types are parsed
    point_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "point"]
    box_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "box"]
    polygon_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "polygon"]
    collection_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "collection"]

    assert len(point_segments) == 1, "Should have 1 point"
    assert len(box_segments) == 1, "Should have 1 box"
    assert len(polygon_segments) == 1, "Should have 1 polygon"
    assert len(collection_segments) == 1, "Should have 1 collection"

    # Verify point values
    point_value = point_segments[0]["value"]
    assert point_value.x == 10
    assert point_value.y == 20

    # Verify box values
    box_value = box_segments[0]["value"]
    assert box_value.top_left.x == 30
    assert box_value.top_left.y == 40
    assert box_value.bottom_right.x == 50
    assert box_value.bottom_right.y == 60

    # Verify polygon values
    polygon_value = polygon_segments[0]["value"]
    assert len(polygon_value.hull) == 3
    assert polygon_value.hull[0].x == 0
    assert polygon_value.hull[0].y == 0

    # Verify collection values
    collection_value = collection_segments[0]["value"]
    assert len(collection_value.points) == 2
    assert collection_value.points[0].mention == "A"
    assert collection_value.points[0].x == 70
    assert collection_value.points[0].y == 80
    assert collection_value.points[1].mention == "B"
    assert collection_value.points[1].x == 90
    assert collection_value.points[1].y == 100

    # Test 2: Strip reasoning tags
    cleaned = strip_reasoning(text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert cleaned == test_case.expected_stripped_text

    # Test 3: Parse text into segments with point parsing
    segments = parse_text(text, parse_points=True)
    reasoning_segments = [seg for seg in segments if seg["kind"] == "reasoning"]
    assert len(reasoning_segments) == 1
    assert reasoning_segments[0]["value"].parsed_segments is not None

    # Verify all point types are in the segments' parsed results
    seg_parsed = reasoning_segments[0]["value"].parsed_segments
    seg_point_segments = [s for s in seg_parsed if s.get("kind") == "point"]
    seg_box_segments = [s for s in seg_parsed if s.get("kind") == "box"]
    seg_polygon_segments = [s for s in seg_parsed if s.get("kind") == "polygon"]
    seg_collection_segments = [s for s in seg_parsed if s.get("kind") == "collection"]

    assert len(seg_point_segments) >= 1, "Segments should have at least 1 point"
    assert len(seg_box_segments) >= 1, "Segments should have at least 1 box"
    assert len(seg_polygon_segments) >= 1, "Segments should have at least 1 polygon"
    assert len(seg_collection_segments) >= 1, "Segments should have at least 1 collection"


# Additional specific tests that don't fit the parameterized structure

def test_extract_multiple_think_tags_concatenates():
    """Test that multiple <think> tags are concatenated."""
    text = "First <think>step 1</think> then <think>step 2</think> done"
    reasoning = extract_reasoning(text)
    expected = Reasoning(content="step 1\nstep 2", parsed_segments=None)
    assert reasoning == expected


def test_case_insensitive_think_tags():
    """Test that <think> tags are case-insensitive."""
    test_cases = [
        ("<think>lowercase</think>", Reasoning(content="lowercase", parsed_segments=None)),
        ("<THINK>uppercase</THINK>", Reasoning(content="uppercase", parsed_segments=None)),
        ("<Think>mixedcase</Think>", Reasoning(content="mixedcase", parsed_segments=None)),
    ]

    for text, expected in test_cases:
        reasoning = extract_reasoning(text)
        assert reasoning == expected


def test_incomplete_think_tags():
    """Test behavior with incomplete tags."""
    incomplete_texts = [
        "<think>unclosed tag",
        "unopened tag</think>",
        "<think>content"
    ]

    for text in incomplete_texts:
        reasoning = extract_reasoning(text)
        assert reasoning is None


def test_nested_think_tags():
    """Test behavior with nested <think> tags (should match first closing tag)."""
    text = "<think>outer <think>inner</think> outer</think>"
    reasoning = extract_reasoning(text)
    assert reasoning is not None
    # Regex is non-greedy, so it matches up to first </think>
    assert "outer" in reasoning.content
    assert "inner" in reasoning.content


def test_parse_text_preserves_spans():
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


def test_reasoning_dataclass():
    """Test Reasoning dataclass properties."""
    reasoning = Reasoning(content="test content")
    assert reasoning.content == "test content"

    # Test equality
    reasoning2 = Reasoning(content="test content")
    assert reasoning == reasoning2

    reasoning3 = Reasoning(content="different")
    assert reasoning != reasoning3


def test_non_string_input():
    """Test handling of non-string inputs."""
    assert extract_reasoning(None) is None
    assert extract_reasoning(123) is None
    assert extract_reasoning([]) is None

    # strip_reasoning should handle non-string gracefully by converting to string
    assert strip_reasoning(None) == ""
    assert strip_reasoning(123) == "123"


def test_strip_reasoning_collapses_spaces():
    """Test that multiple spaces are collapsed after removing tags."""
    text = "Word1   <think>reasoning</think>   Word2"
    cleaned = strip_reasoning(text)
    expected = "Word1 Word2"
    assert cleaned == expected


def test_combine_reasoning_sources():
    """Test combining reasoning from API reasoning_content and <think> tags."""
    from perceptron.client import _ClientCore

    # Test with both sources
    api_reasoning = "API reasoning content"
    tag_reasoning = Reasoning(content="Tag reasoning content")
    combined = _ClientCore._combine_reasoning_sources(tag_reasoning, api_reasoning)
    expected = Reasoning(content="API reasoning content\nTag reasoning content")
    assert combined == expected

    # Test with only API reasoning
    combined = _ClientCore._combine_reasoning_sources(None, api_reasoning)
    expected = Reasoning(content="API reasoning content")
    assert combined == expected

    # Test with only tag reasoning
    combined = _ClientCore._combine_reasoning_sources(tag_reasoning, "")
    assert combined == tag_reasoning

    # Test with neither
    combined = _ClientCore._combine_reasoning_sources(None, "")
    assert combined is None


# Tests for reasoning with point parsing

def test_extract_reasoning_with_points():
    """Test extracting reasoning that contains point tags."""
    text = "<think>The object is at <point>(100,200)</point></think>"
    reasoning = extract_reasoning(text, parse_points=True)

    assert reasoning is not None
    assert "The object is at" in reasoning.content
    assert reasoning.parsed_segments is not None

    # Should have parsed segments with point
    point_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "point"]
    assert len(point_segments) == 1

    # Verify the point value
    point_value = point_segments[0]["value"]
    assert point_value.x == 100
    assert point_value.y == 200


def test_extract_reasoning_with_box():
    """Test extracting reasoning that contains box tags."""
    text = "<think>The region is <point_box>(10,20) (30,40)</point_box></think>"
    reasoning = extract_reasoning(text, parse_points=True)

    assert reasoning is not None
    assert reasoning.parsed_segments is not None

    # Should have parsed segments with box
    box_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "box"]
    assert len(box_segments) == 1

    # Verify the box value
    box_value = box_segments[0]["value"]
    assert box_value.top_left.x == 10
    assert box_value.top_left.y == 20
    assert box_value.bottom_right.x == 30
    assert box_value.bottom_right.y == 40


def test_extract_reasoning_with_polygon():
    """Test extracting reasoning that contains polygon tags."""
    text = "<think>Shape: <polygon>(0,0) (10,0) (5,10)</polygon></think>"
    reasoning = extract_reasoning(text, parse_points=True)

    assert reasoning is not None
    assert reasoning.parsed_segments is not None

    # Should have parsed segments with polygon
    polygon_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "polygon"]
    assert len(polygon_segments) == 1

    # Verify the polygon value
    polygon_value = polygon_segments[0]["value"]
    assert len(polygon_value.hull) == 3
    assert polygon_value.hull[0].x == 0
    assert polygon_value.hull[0].y == 0


def test_extract_reasoning_with_nested_collection():
    """Test extracting reasoning with nested points in a collection."""
    text = """<think>
    Multiple objects:
    <collection>
        <point mention="object1">(50,60)</point>
        <point mention="object2">(70,80)</point>
    </collection>
    </think>"""
    reasoning = extract_reasoning(text, parse_points=True)

    assert reasoning is not None
    assert reasoning.parsed_segments is not None

    # Should have parsed segments with collection
    collection_segments = [seg for seg in reasoning.parsed_segments if seg.get("kind") == "collection"]
    assert len(collection_segments) == 1

    # Verify the collection has points
    collection_value = collection_segments[0]["value"]
    assert len(collection_value.points) == 2
    assert collection_value.points[0].mention == "object1"
    assert collection_value.points[1].mention == "object2"


def test_extract_reasoning_without_point_parsing():
    """Test that point parsing can be disabled."""
    text = "<think>Point at <point>(10,20)</point></think>"
    reasoning = extract_reasoning(text, parse_points=False)
    expected = Reasoning(content="Point at <point>(10,20)</point>", parsed_segments=None)
    assert reasoning == expected


def test_extract_reasoning_no_points():
    """Test reasoning without any point tags."""
    text = "<think>Just plain reasoning text</think>"
    reasoning = extract_reasoning(text, parse_points=True)
    expected = Reasoning(content="Just plain reasoning text", parsed_segments=None)
    assert reasoning == expected


def test_parse_text_with_reasoning_containing_points():
    """Test parse_text when reasoning contains point tags."""
    text = "Answer: <think>Located at <point>(15,25)</point></think> done"
    segments = parse_text(text, parse_points=True)

    # Should have 3 segments: text, reasoning, text
    assert len(segments) == 3

    # Check reasoning segment
    reasoning_seg = segments[1]
    assert reasoning_seg["kind"] == "reasoning"
    assert reasoning_seg["value"].parsed_segments is not None

    # Verify point in reasoning
    point_segments = [s for s in reasoning_seg["value"].parsed_segments if s.get("kind") == "point"]
    assert len(point_segments) == 1
    assert point_segments[0]["value"].x == 15
    assert point_segments[0]["value"].y == 25


def test_parse_text_multiple_reasoning_with_points():
    """Test parse_text with multiple reasoning blocks containing points."""
    text = "<think>First <point>(1,2)</point></think> middle <think>Second <point>(3,4)</point></think>"
    segments = parse_text(text, parse_points=True)

    reasoning_segments = [seg for seg in segments if seg["kind"] == "reasoning"]
    assert len(reasoning_segments) == 2

    # Both should have parsed points
    for r_seg in reasoning_segments:
        assert r_seg["value"].parsed_segments is not None
        point_segs = [s for s in r_seg["value"].parsed_segments if s.get("kind") == "point"]
        assert len(point_segs) >= 1


def test_parse_text_reasoning_with_mixed_content():
    """Test reasoning with text and multiple point types."""
    text = """<think>
    First object at <point>(10,20)</point>
    Then region <point_box>(30,40) (50,60)</point_box>
    And shape <polygon>(0,0) (10,0) (5,10)</polygon>
    </think>"""
    segments = parse_text(text, parse_points=True)

    reasoning_seg = segments[0]
    assert reasoning_seg["kind"] == "reasoning"
    assert reasoning_seg["value"].parsed_segments is not None

    parsed = reasoning_seg["value"].parsed_segments
    # Should have text, point, text, box, text, polygon segments
    point_count = sum(1 for s in parsed if s.get("kind") == "point")
    box_count = sum(1 for s in parsed if s.get("kind") == "box")
    polygon_count = sum(1 for s in parsed if s.get("kind") == "polygon")

    assert point_count == 1
    assert box_count == 1
    assert polygon_count == 1


def test_reasoning_dataclass_with_parsed_segments():
    """Test Reasoning dataclass with parsed_segments."""
    reasoning1 = Reasoning(content="test", parsed_segments=None)
    assert reasoning1.content == "test"
    assert reasoning1.parsed_segments is None

    reasoning2 = Reasoning(content="test", parsed_segments=[{"kind": "text", "text": "test"}])
    assert reasoning2.content == "test"
    assert reasoning2.parsed_segments is not None
    assert len(reasoning2.parsed_segments) == 1

    # Test equality with parsed_segments
    reasoning3 = Reasoning(content="test", parsed_segments=None)
    assert reasoning1 == reasoning3
