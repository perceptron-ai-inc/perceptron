from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotations import annotate_image, canonicalize_text_collections, serialize_annotations
from .client import _PROVIDER_CONFIG, _select_model
from .config import settings
from .dsl.nodes import (
    Sequence as SequenceNode,
)
from .dsl.nodes import (
    agent,
    system,
    text,
)
from .dsl.nodes import (
    image as image_node,
)
from .dsl.perceive import perceive
from .errors import BadRequestError
from .expectations import expectation_hint_text, resolve_structured_expectation
from .pointing.types import bbox
from .prompting import (
    CaptionPromptTemplate,
    DetectPromptTemplate,
    HighLevelPromptProfile,
    OcrPromptTemplate,
    QuestionPromptTemplate,
    resolve_prompt_profile,
)

COCO_BBOX_MIN_VALUES = 4


@dataclass
class _NormalizedExample:
    image: Any
    prompt: str | None
    tags: str


@dataclass
class CocoDetectResult:
    image_path: Path
    coco_image: dict[str, Any]
    result: Any


def _prompt_profile_from_kwargs(gen_kwargs: Mapping[str, Any]) -> tuple[HighLevelPromptProfile, str | None]:
    """Resolve the active prompt profile (and model) for a high-level helper call."""

    env = settings()
    provider_override = gen_kwargs.get("provider")
    provider_name = provider_override or env.provider or "fal"
    provider_key = provider_name.lower() if isinstance(provider_name, str) else provider_name
    provider_cfg = _PROVIDER_CONFIG.get(provider_key or "") or {}
    requested_model = gen_kwargs.get("model")
    if requested_model is None:
        requested_model = env.model
    resolved_model = _select_model(provider_cfg, requested_model, provider_name=provider_key or "fal")
    if resolved_model is None:
        resolved_model = provider_cfg.get("default_model")
    profile = resolve_prompt_profile(resolved_model)
    return profile, resolved_model


def _normalize_examples(examples: Sequence[Any], class_order: Sequence[str] | None) -> list[_NormalizedExample]:
    normalized: list[_NormalizedExample] = []
    order_lookup = {label: idx for idx, label in enumerate(class_order)} if class_order else None
    for example in examples:
        if not isinstance(example, Mapping) or "image" not in example:
            raise BadRequestError(
                "Detection examples must be dicts with 'image' and annotation lists (boxes/polygons/points)"
            )
        image_obj = example["image"]
        prompt = canonicalize_text_collections(example.get("prompt"))

        annotations_payload = example.get("annotations")
        if annotations_payload is None:
            annotations_payload = []
            for key in ("boxes", "polygons", "points", "collections"):
                values = example.get(key)
                if values:
                    annotations_payload.extend(values)

        if not annotations_payload:
            raise BadRequestError("Detection examples must include at least one annotation")

        annotated = annotate_image(image_obj, annotations_payload)
        tags = serialize_annotations(
            annotated.get("boxes"),
            annotated.get("polygons"),
            annotated.get("points"),
            annotated.get("collections"),
            order_lookup,
        )
        if not tags:
            raise BadRequestError("Detection examples must include at least one annotation")
        normalized.append(_NormalizedExample(image=image_obj, prompt=prompt, tags=tags))
    return normalized


def _run_perceive_sequence(
    *,
    builder: Callable[[], SequenceNode],
    perceive_base_kwargs: Mapping[str, Any],
    gen_kwargs: dict[str, Any],
):
    perceive_kwargs = dict(perceive_base_kwargs)
    perceive_kwargs.update(gen_kwargs)
    runner = perceive(**perceive_kwargs)

    @runner
    def _run():
        return builder()

    return _run()


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------


def _caption_sequence(
    image_obj: Any,
    style: str,
    expects: str | None,
    template: CaptionPromptTemplate,
) -> SequenceNode:
    style_map = template.style_prompts
    if style not in style_map:
        raise BadRequestError(f"Unsupported caption style: {style}")
    hint = expectation_hint_text(expects)
    nodes = []
    if template.system_instruction:
        nodes.append(system(template.system_instruction))
    if hint:
        nodes.append(text(hint))
    nodes.append(image_node(image_obj))
    nodes.append(text(style_map[style]))
    return SequenceNode(nodes)


def caption(
    image_obj: Any,
    *,
    style: str = "concise",
    expects: str = "box",
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Generate a caption for an image using predefined best-practice prompts."""

    profile, _ = _prompt_profile_from_kwargs(gen_kwargs)
    caption_template = profile.caption
    structured_expectation, allow_multiple = resolve_structured_expectation(expects, context="caption expects value")

    base_kwargs = {
        "stream": stream,
        "expects": structured_expectation,
        "allow_multiple": allow_multiple,
    }

    return _run_perceive_sequence(
        builder=lambda: _caption_sequence(image_obj, style, structured_expectation, caption_template),
        perceive_base_kwargs=base_kwargs,
        gen_kwargs=gen_kwargs,
    )


# ---------------------------------------------------------------------------
# Question Answering
# ---------------------------------------------------------------------------


def _question_sequence(
    image_obj: Any,
    question_text: str,
    expects: str | None,
    template: QuestionPromptTemplate,
) -> SequenceNode:
    if expects in {"point", "box", "polygon"}:
        system_instruction = template.grounded_instruction
    else:
        system_instruction = template.open_instruction

    nodes: list = []
    if system_instruction:
        nodes.append(system(system_instruction))
    hint = expectation_hint_text(expects)
    if hint:
        nodes.append(text(hint))
    nodes.append(image_node(image_obj))
    nodes.append(text(question_text))
    return SequenceNode(nodes)


def question(
    image_obj: Any,
    question_text: str,
    *,
    expects: str = "text",
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Answer a question about an image, optionally requesting structured outputs."""

    profile, _ = _prompt_profile_from_kwargs(gen_kwargs)
    question_template = profile.question
    structured_expectation, allow_multiple = resolve_structured_expectation(expects, context="expects value")

    base_kwargs = {
        "stream": stream,
        "expects": structured_expectation,
        "allow_multiple": allow_multiple,
    }

    return _run_perceive_sequence(
        builder=lambda: _question_sequence(image_obj, question_text, structured_expectation, question_template),
        perceive_base_kwargs=base_kwargs,
        gen_kwargs=gen_kwargs,
    )


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _ocr_sequence(
    image_obj: Any,
    prompt: str | None,
    template: OcrPromptTemplate,
) -> SequenceNode:
    nodes = []
    if template.system_instruction:
        nodes.append(system(template.system_instruction))
    nodes.append(image_node(image_obj))
    if prompt:
        nodes.append(text(prompt))
    return SequenceNode(nodes)


def _run_ocr(
    image_obj: Any,
    *,
    prompt: str | None,
    stream: bool,
    mode: str,
    gen_kwargs: dict[str, Any],
):
    profile, resolved_model = _prompt_profile_from_kwargs(gen_kwargs)
    ocr_template = profile.ocr
    effective_prompt = prompt
    if effective_prompt is None:
        available_modes = set(ocr_template.prompts.keys())
        if mode not in ocr_template.prompts:
            model_label = resolved_model or profile.key
            available_display = ", ".join(sorted(available_modes)) or "none"
            raise BadRequestError(
                f"OCR mode '{mode}' is not configured for model '{model_label}'. Available modes: {available_display}."
            )
        effective_prompt = ocr_template.prompts.get(mode)
    base_kwargs = {
        "stream": stream,
        "expects": None,
        "allow_multiple": False,
    }

    return _run_perceive_sequence(
        builder=lambda: _ocr_sequence(image_obj, effective_prompt, ocr_template),
        perceive_base_kwargs=base_kwargs,
        gen_kwargs=gen_kwargs,
    )


def ocr(
    image_obj: Any,
    *,
    prompt: str | None = None,
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Perform OCR on an image (plain text)."""

    return _run_ocr(image_obj, prompt=prompt, stream=stream, mode="plain", gen_kwargs=gen_kwargs)


def ocr_markdown(
    image_obj: Any,
    *,
    prompt: str | None = None,
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Perform OCR and request Markdown output when supported by the provider."""

    return _run_ocr(image_obj, prompt=prompt, stream=stream, mode="markdown", gen_kwargs=gen_kwargs)


def ocr_html(
    image_obj: Any,
    *,
    prompt: str | None = None,
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Perform OCR and request HTML output when supported by the provider."""

    return _run_ocr(image_obj, prompt=prompt, stream=stream, mode="html", gen_kwargs=gen_kwargs)


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def _detect_system_message(
    classes: Sequence[str] | None,
    template: DetectPromptTemplate,
) -> SequenceNode:
    if classes:
        categories = ", ".join(str(c) for c in classes)
        message = template.category_instruction_template.format(categories=categories)
    else:
        message = template.general_instruction
    return SequenceNode([system(message)])


def _detect_sequence(
    image_obj: Any,
    *,
    classes: Sequence[str] | None,
    examples: Sequence[Any] | None,
    template: DetectPromptTemplate,
) -> SequenceNode:
    sequence = _detect_system_message(classes, template)
    normalized_examples = _normalize_examples(examples, classes) if examples else []
    for ex in normalized_examples:
        sequence = sequence + image_node(ex.image)
        if ex.prompt:
            sequence = sequence + text(ex.prompt)
        sequence = sequence + agent(ex.tags)
    im = image_node(image_obj)
    sequence = sequence + im
    return sequence


def detect(  # noqa: PLR0913
    image_obj: Any,
    *,
    classes: Sequence[str] | None = None,
    examples: Sequence[Any] | None = None,
    strict: bool | None = None,
    max_outputs: int | None = None,
    stream: bool = False,
    **gen_kwargs: Any,
):
    """High-level object detection helper."""

    profile, _ = _prompt_profile_from_kwargs(gen_kwargs)
    detect_template = profile.detect
    base_kwargs: dict[str, Any] = {
        "stream": stream,
        "expects": "box",
        "allow_multiple": True,
        "max_outputs": max_outputs,
    }
    if strict is not None:
        base_kwargs["strict"] = strict

    return _run_perceive_sequence(
        builder=lambda: _detect_sequence(image_obj, classes=classes, examples=examples, template=detect_template),
        perceive_base_kwargs=base_kwargs,
        gen_kwargs=gen_kwargs,
    )


# ---------------------------------------------------------------------------
# COCO utilities
# ---------------------------------------------------------------------------


def _load_coco_annotations(
    dataset_dir: Path,
    *,
    annotation_file: Path | None,
    split: str | None,
) -> tuple[Path, dict[str, Any]]:
    def _is_coco_payload(payload: dict[str, Any]) -> bool:
        return isinstance(payload, dict) and "images" in payload and "annotations" in payload

    if annotation_file:
        ann_path = dataset_dir / annotation_file if not annotation_file.is_absolute() else annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        if not _is_coco_payload(payload):
            raise BadRequestError(f"Annotation file does not appear to be COCO formatted: {ann_path}")
        return ann_path, payload

    candidates: list[Path] = []
    search_roots: list[Path] = []
    if split:
        split_dir = dataset_dir / split
        if split_dir.exists():
            search_roots.append(split_dir)
    annotation_dir = dataset_dir / "annotations"
    if annotation_dir.exists():
        search_roots.append(annotation_dir)
    search_roots.append(dataset_dir)

    for root in search_roots:
        for path in sorted(root.glob("**/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _is_coco_payload(payload):
                candidates.append(path)
                if split and split.lower() in path.name.lower():
                    return path, payload
    if not candidates:
        raise FileNotFoundError(f"Could not find a COCO annotation JSON under {dataset_dir}")

    ann_path = candidates[0]
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    return ann_path, payload


def _candidate_image_roots(dataset_dir: Path, split: str | None) -> Iterable[Path]:
    roots: list[Path] = []
    if split:
        roots.append(dataset_dir / split / "images")
        roots.append(dataset_dir / split)
    roots.append(dataset_dir / "images")
    roots.append(dataset_dir)
    seen: set[Path] = set()
    for root in roots:
        if root not in seen and root.exists():
            seen.add(root)
            yield root


def _resolve_coco_image_path(dataset_dir: Path, file_name: str, *, split: str | None) -> Path:
    image_name = Path(file_name)
    if image_name.is_absolute():
        return image_name

    for root in _candidate_image_roots(dataset_dir, split):
        candidate = root / image_name
        if candidate.exists():
            return candidate

    fallback = (dataset_dir / image_name).resolve()
    return fallback


def _sorted_categories(categories: Sequence[dict[str, Any]]) -> list[str]:
    sorted_categories = sorted(categories, key=lambda cat: cat.get("id", 0))
    return [cat.get("name") for cat in sorted_categories if cat.get("name")]


def _build_coco_examples(  # noqa: PLR0912, PLR0913, PLR0915
    dataset_path: Path,
    payload: dict[str, Any],
    *,
    allowed_category_ids: Sequence[int],
    category_names: Sequence[str],
    split: str | None,
    shots: int,
) -> list[dict[str, Any]]:
    if shots <= 0:
        return []

    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    if not images or not annotations:
        return []

    image_meta_by_id = {img.get("id"): img for img in images if img.get("id") is not None}
    category_by_id = {cat.get("id"): cat for cat in payload.get("categories", []) if cat.get("id") is not None}

    # Map image_id -> list of bounding boxes
    annotations_by_image: dict[int, list[Any]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id not in image_meta_by_id or category_id not in allowed_category_ids:
            continue
        bbox_values = ann.get("bbox")
        if not bbox_values or len(bbox_values) < COCO_BBOX_MIN_VALUES:
            continue
        x, y, width, height = bbox_values[:4]
        x2 = x + width
        y2 = y + height
        name = (category_by_id.get(category_id) or {}).get("name")
        if not name:
            continue
        box = bbox(
            round(x),
            round(y),
            round(x2),
            round(y2),
            mention=name,
        )
        annotations_by_image[image_id].append(box)

    if not annotations_by_image:
        return []

    category_to_images: dict[str, list[int]] = defaultdict(list)
    for image_id, boxes in annotations_by_image.items():
        mentions = {box.mention for box in boxes if box.mention}
        for mention in mentions:
            category_to_images[mention].append(image_id)

    ordered_categories = (
        list(category_names)
        if category_names
        else list({box.mention for boxes in annotations_by_image.values() for box in boxes if box.mention})
    )
    if not ordered_categories:
        return []

    # Ensure deterministic order
    for cat in category_to_images.values():
        cat.sort()

    examples: list[dict[str, Any]] = []
    used_images: set[int] = set()
    category_positions: dict[str, int] = defaultdict(int)

    while len(examples) < shots:
        added = False
        for category in ordered_categories:
            if len(examples) >= shots:
                break
            image_ids = category_to_images.get(category, [])
            pos = category_positions[category]
            while pos < len(image_ids) and image_ids[pos] in used_images:
                pos += 1
            category_positions[category] = pos
            if pos >= len(image_ids):
                continue
            image_id = image_ids[pos]
            category_positions[category] = pos + 1
            boxes = annotations_by_image.get(image_id)
            if not boxes:
                continue
            image_meta = image_meta_by_id.get(image_id)
            if not image_meta:
                continue
            file_name = image_meta.get("file_name")
            if not file_name:
                continue
            image_path = _resolve_coco_image_path(dataset_path, file_name, split=split)
            if not image_path.exists():
                continue
            try:
                image_bytes = image_path.read_bytes()
            except Exception:
                continue
            examples.append({"image": image_bytes, "boxes": list(boxes)})
            used_images.add(image_id)
            added = True
        if not added:
            break

    return examples


def detect_from_coco(  # noqa: PLR0913
    dataset_dir: str | Path,
    *,
    annotation_file: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    classes: Sequence[str] | None = None,
    stream: bool = False,
    shots: int = 0,
    **detect_kwargs: Any,
) -> list[CocoDetectResult]:
    """Run detection across a COCO-format dataset directory."""

    if stream:
        raise BadRequestError("detect_from_coco does not support streaming output.")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    ann_path_input = Path(annotation_file) if annotation_file is not None else None
    ann_path, payload = _load_coco_annotations(dataset_path, annotation_file=ann_path_input, split=split)

    images = payload.get("images") or []
    if not images:
        raise BadRequestError(f"Annotation file {ann_path} does not contain any images.")

    categories = payload.get("categories") or []
    category_by_id = {cat.get("id"): cat for cat in categories if cat.get("id") is not None}
    name_to_id = {cat.get("name"): cat_id for cat_id, cat in category_by_id.items() if cat.get("name")}

    if classes:
        missing = [name for name in classes if name not in name_to_id]
        if missing:
            raise BadRequestError("Classes not found in COCO categories: " + ", ".join(sorted(missing)))
        category_names = list(classes)
    else:
        category_names = _sorted_categories(categories)

    allowed_category_ids = [name_to_id[name] for name in category_names if name in name_to_id]
    if not allowed_category_ids:
        allowed_category_ids = list(category_by_id.keys())

    class_list = category_names or None

    ordered_images = sorted(images, key=lambda img: img.get("id", 0))
    if limit is not None:
        ordered_images = ordered_images[: max(limit, 0)]

    examples = _build_coco_examples(
        dataset_path,
        payload,
        allowed_category_ids=allowed_category_ids,
        category_names=category_names,
        split=split,
        shots=shots,
    )

    results: list[CocoDetectResult] = []
    for image_meta in ordered_images:
        file_name = image_meta.get("file_name")
        if not file_name:
            raise BadRequestError(f"Image entry missing file_name in {ann_path}")

        image_path = _resolve_coco_image_path(dataset_path, file_name, split=split)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file referenced in COCO annotations not found: {image_path}")

        image_bytes = image_path.read_bytes()
        result = detect(
            image_bytes,
            classes=class_list,
            stream=False,
            examples=examples if examples else None,
            **detect_kwargs,
        )
        results.append(CocoDetectResult(image_path=image_path, coco_image=dict(image_meta), result=result))

    return results
