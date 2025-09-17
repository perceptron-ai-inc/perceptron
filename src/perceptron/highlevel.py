from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .dsl.nodes import Sequence as SequenceNode, agent, image as image_node, system, text
from .dsl.perceive import perceive
from .errors import BadRequestError
from .pointing.parser import parse_text
from .annotations import serialize_annotations, annotate_image, canonicalize_text_collections
from .pointing.types import bbox


@dataclass
class _NormalizedExample:
    image: Any
    prompt: Optional[str]
    tags: str


@dataclass
class CocoDetectResult:
    image_path: Path
    coco_image: Dict[str, Any]
    result: Any


def _normalize_examples(examples: Sequence[Any], class_order: Sequence[str] | None) -> List[_NormalizedExample]:
    normalized: List[_NormalizedExample] = []
    order_lookup = {label: idx for idx, label in enumerate(class_order)} if class_order else None
    for example in examples:
        if isinstance(example, dict) and "image" in example:
            image_obj = example["image"]
            prompt = canonicalize_text_collections(example.get("prompt"))
            tags = serialize_annotations(
                example.get("boxes"),
                example.get("polygons"),
                example.get("points"),
                example.get("collections"),
                order_lookup,
            )
            if not tags:
                raise BadRequestError("Detection examples must include at least one annotation")
            normalized.append(_NormalizedExample(image=image_obj, prompt=prompt, tags=tags))
        else:
            raise BadRequestError(
                "Detection examples must be dicts with 'image' and annotation lists (boxes/polygons/points)"
            )
    return normalized


def _expectation_hint_text(expects: Optional[str]) -> Optional[str]:
    if expects in {"point", "box", "polygon"}:
        return f"<hint>{expects.upper()}</hint>"
    return None


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------


def _caption_sequence(image_obj: Any, style: str) -> SequenceNode:
    style_map = {
        "concise": "Provide a concise, human-friendly caption for the upcoming image.",
        "detailed": "Provide a detailed caption describing key objects, relationships, and context in the upcoming image.",
    }
    if style not in style_map:
        raise BadRequestError(f"Unsupported caption style: {style}")
    hint = _expectation_hint_text("box")
    nodes = []
    if hint:
        nodes.append(text(hint))
    nodes.append(image_node(image_obj))
    nodes.append(text(style_map[style]))
    return SequenceNode(nodes)


def caption(
    image_obj: Any,
    *,
    style: str = "concise",
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Generate a caption for an image using predefined best-practice prompts."""

    perceive_kwargs: dict[str, Any] = {"stream": stream, "expects": "box", "allow_multiple": True}
    perceive_kwargs.update(gen_kwargs)
    captioner = perceive(**perceive_kwargs)

    @captioner
    def _run():
        return _caption_sequence(image_obj, style)

    return _run()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _ocr_sequence(
    image_obj: Any,
    prompt: Optional[str],
) -> SequenceNode:
    base_instruction = prompt or "Transcribe every readable word in the image."
    system_instruction = "You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image."
    nodes = [system(system_instruction)]
    im = image_node(image_obj)
    nodes.extend([im, text(base_instruction)])
    return SequenceNode(nodes)


def ocr(
    image_obj: Any,
    *,
    prompt: Optional[str] = None,
    stream: bool = False,
    **gen_kwargs: Any,
):
    """Perform OCR on an image."""

    perceive_kwargs: dict[str, Any] = {"stream": stream, "expects": None, "allow_multiple": False}
    perceive_kwargs.update(gen_kwargs)
    reader = perceive(**perceive_kwargs)

    @reader
    def _run():
        return _ocr_sequence(image_obj, prompt)

    return _run()


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


def _detect_system_message(classes: Sequence[str] | None) -> SequenceNode:
    categories = ", ".join(str(c) for c in classes) if classes else "the objects in the scene"
    message = (
        "Your goal is to segment out the following categories: "
        f"{categories}. Always respond using <point_box> tags and include mention attributes when appropriate."
    )
    return SequenceNode([system(message)])


def _detect_sequence(
    image_obj: Any,
    *,
    classes: Sequence[str] | None,
    examples: Sequence[Any] | None,
) -> SequenceNode:
    sequence = _detect_system_message(classes)
    hint = _expectation_hint_text("box")
    if hint:
        sequence = sequence + text(hint)
    normalized_examples = _normalize_examples(examples, classes) if examples else []
    for ex in normalized_examples:
        sequence = sequence + image_node(ex.image)
        if ex.prompt:
            sequence = sequence + text(ex.prompt)
        sequence = sequence + agent(ex.tags)
    instruction = "Return canonical <point_box> tags for every detected object. Include a mention attribute with the class label when known."
    if classes:
        instruction += " Focus on: " + ", ".join(str(c) for c in classes) + "."
    else:
        instruction += " Include all salient objects in the scene."
    im = image_node(image_obj)
    sequence = sequence + im + text(instruction)
    return sequence


def detect(
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

    perceive_kwargs: dict[str, Any] = {
        "expects": "box",
        "stream": stream,
        "allow_multiple": True,
        "max_outputs": max_outputs,
    }
    if strict is not None:
        perceive_kwargs["strict"] = strict
    perceive_kwargs.update(gen_kwargs)

    detector = perceive(**perceive_kwargs)

    @detector
    def _run():
        return _detect_sequence(image_obj, classes=classes, examples=examples)

    return _run()


# ---------------------------------------------------------------------------
# COCO utilities
# ---------------------------------------------------------------------------


def _load_coco_annotations(
    dataset_dir: Path,
    *,
    annotation_file: Optional[Path],
    split: Optional[str],
) -> Tuple[Path, Dict[str, Any]]:
    def _is_coco_payload(payload: Dict[str, Any]) -> bool:
        return isinstance(payload, dict) and "images" in payload and "annotations" in payload

    if annotation_file:
        ann_path = dataset_dir / annotation_file if not annotation_file.is_absolute() else annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        if not _is_coco_payload(payload):
            raise BadRequestError(f"Annotation file does not appear to be COCO formatted: {ann_path}")
        return ann_path, payload

    candidates: List[Path] = []
    search_roots: List[Path] = []
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


def _candidate_image_roots(dataset_dir: Path, split: Optional[str]) -> Iterable[Path]:
    roots: List[Path] = []
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


def _resolve_coco_image_path(
    dataset_dir: Path, file_name: str, *, split: Optional[str]
) -> Path:
    image_name = Path(file_name)
    if image_name.is_absolute():
        return image_name

    for root in _candidate_image_roots(dataset_dir, split):
        candidate = root / image_name
        if candidate.exists():
            return candidate

    fallback = (dataset_dir / image_name).resolve()
    return fallback


def _sorted_categories(categories: Sequence[Dict[str, Any]]) -> List[str]:
    sorted_categories = sorted(categories, key=lambda cat: cat.get("id", 0))
    return [cat.get("name") for cat in sorted_categories if cat.get("name")]


def _build_coco_examples(
    dataset_path: Path,
    payload: Dict[str, Any],
    *,
    allowed_category_ids: Sequence[int],
    category_names: Sequence[str],
    split: Optional[str],
    shots: int,
) -> List[Dict[str, Any]]:
    if shots <= 0:
        return []

    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    if not images or not annotations:
        return []

    image_meta_by_id = {img.get("id"): img for img in images if img.get("id") is not None}
    category_by_id = {cat.get("id"): cat for cat in payload.get("categories", []) if cat.get("id") is not None}

    # Map image_id -> list of bounding boxes
    annotations_by_image: Dict[int, List[Any]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        if image_id not in image_meta_by_id or category_id not in allowed_category_ids:
            continue
        bbox_values = ann.get("bbox")
        if not bbox_values or len(bbox_values) < 4:
            continue
        x, y, width, height = bbox_values[:4]
        x2 = x + width
        y2 = y + height
        name = (category_by_id.get(category_id) or {}).get("name")
        if not name:
            continue
        box = bbox(
            int(round(x)),
            int(round(y)),
            int(round(x2)),
            int(round(y2)),
            mention=name,
        )
        annotations_by_image[image_id].append(box)

    if not annotations_by_image:
        return []

    category_to_images: Dict[str, List[int]] = defaultdict(list)
    for image_id, boxes in annotations_by_image.items():
        mentions = {box.mention for box in boxes if box.mention}
        for mention in mentions:
            category_to_images[mention].append(image_id)

    ordered_categories = list(category_names) if category_names else list({box.mention for boxes in annotations_by_image.values() for box in boxes if box.mention})
    if not ordered_categories:
        return []

    # Ensure deterministic order
    for cat in category_to_images.values():
        cat.sort()

    examples: List[Dict[str, Any]] = []
    used_images: set[int] = set()
    category_positions: Dict[str, int] = defaultdict(int)

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


def detect_from_coco(
    dataset_dir: str | Path,
    *,
    annotation_file: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    classes: Sequence[str] | None = None,
    stream: bool = False,
    shots: int = 0,
    **detect_kwargs: Any,
) -> List[CocoDetectResult]:
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
    name_to_id = {
        cat.get("name"): cat_id
        for cat_id, cat in category_by_id.items()
        if cat.get("name")
    }

    if classes:
        missing = [name for name in classes if name not in name_to_id]
        if missing:
            raise BadRequestError(
                "Classes not found in COCO categories: " + ", ".join(sorted(missing))
            )
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

    results: List[CocoDetectResult] = []
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
