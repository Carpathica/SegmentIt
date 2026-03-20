from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml
from PIL import Image

from .schemas import Segment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clean_classes(raw_classes: Iterable[str]) -> List[str]:
    classes: List[str] = []
    for item in raw_classes:
        label = str(item).strip()
        if label:
            classes.append(label)
    return classes


def load_classes(
    dataset_dir: Path,
    provided_classes: Iterable[str],
    classes_file: str | None = None,
) -> List[str]:
    classes = clean_classes(provided_classes)
    if classes:
        return classes

    if classes_file:
        classes_path = resolve_path(dataset_dir, classes_file)
        loaded = _load_classes_file(classes_path)
        if loaded:
            return loaded

    for candidate_name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        candidate = dataset_dir / candidate_name
        if candidate.exists():
            loaded = _load_classes_file(candidate)
            if loaded:
                return loaded

    txt_candidate = dataset_dir / "classes.txt"
    if txt_candidate.exists():
        loaded = _load_classes_file(txt_candidate)
        if loaded:
            return loaded

    return []


def _load_classes_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, list):
            return clean_classes(names)
        if isinstance(names, dict):
            indexed = []
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                indexed.append((idx, str(value)))
            indexed.sort(key=lambda item: item[0])
            return clean_classes(value for _, value in indexed)
        return []

    return clean_classes(path.read_text(encoding="utf-8").splitlines())


def discover_images(dataset_dir: Path) -> List[str]:
    images_root = dataset_dir / "images"
    if images_root.exists() and images_root.is_dir():
        root = images_root
        keep_prefix = True
    else:
        root = dataset_dir
        keep_prefix = False

    images: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_dir).as_posix() if keep_prefix else path.relative_to(root).as_posix()
        if rel.startswith("labels/"):
            continue
        images.append(rel)

    images.sort()
    return images


def resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def ensure_in_base(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("Path escapes dataset directory.") from exc
    return candidate


def image_absolute_path(dataset_dir: Path, image_rel_path: str) -> Path:
    image_path = (dataset_dir / image_rel_path).resolve()
    return ensure_in_base(dataset_dir, image_path)


def label_relative_path(dataset_dir: Path, image_rel_path: str) -> Path:
    image_rel = Path(image_rel_path)
    has_labels_dir = (dataset_dir / "labels").exists()
    has_images_dir = (dataset_dir / "images").exists()

    if image_rel.parts and image_rel.parts[0] == "images":
        return Path("labels", *image_rel.parts[1:]).with_suffix(".txt")
    if has_images_dir:
        return Path("labels", image_rel_path).with_suffix(".txt")
    if has_labels_dir:
        return Path("labels", image_rel_path).with_suffix(".txt")
    return image_rel.with_suffix(".txt")


def _strip_images_prefix(path: Path) -> Path:
    if path.parts and path.parts[0] == "images":
        return Path(*path.parts[1:])
    return path


def label_absolute_path(
    dataset_dir: Path,
    image_rel_path: str,
    labels_dir: str | None = None,
) -> Path:
    image_rel = _strip_images_prefix(Path(image_rel_path)).with_suffix(".txt")
    if labels_dir:
        labels_root = resolve_path(dataset_dir, labels_dir)
        return (labels_root / image_rel).resolve()

    label_rel = label_relative_path(dataset_dir, image_rel_path)
    return ensure_in_base(dataset_dir, (dataset_dir / label_rel).resolve())


def read_annotations(dataset_dir: Path, image_rel_path: str, labels_dir: str | None = None) -> List[Segment]:
    image_path = image_absolute_path(dataset_dir, image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_rel_path}")

    label_abs = label_absolute_path(dataset_dir, image_rel_path, labels_dir=labels_dir)

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    if not label_abs.exists():
        return []

    segments: List[Segment] = []
    for raw_line in label_abs.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 6:
            continue
        if len(coords) % 2 == 1:
            coords = coords[:-1]
        points: List[List[float]] = []
        for idx in range(0, len(coords), 2):
            x = coords[idx] * image_width
            y = coords[idx + 1] * image_height
            points.append([x, y])
        if len(points) < 3:
            continue
        segments.append(Segment(class_id=max(class_id, 0), points=points))
    return segments


def save_annotations(
    dataset_dir: Path,
    image_rel_path: str,
    segments: Iterable[Segment],
    labels_dir: str | None = None,
) -> Path:
    image_path = image_absolute_path(dataset_dir, image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_rel_path}")

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    label_abs = label_absolute_path(dataset_dir, image_rel_path, labels_dir=labels_dir)
    label_abs.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for segment in segments:
        points = segment.points
        if not points or len(points) < 3:
            continue
        coords: List[float] = []
        for point in points:
            if len(point) < 2:
                continue
            x = max(0.0, min(float(point[0]), image_width))
            y = max(0.0, min(float(point[1]), image_height))
            coords.append(x / image_width)
            coords.append(y / image_height)
        if len(coords) < 6:
            continue
        line = f"{int(segment.class_id)} " + " ".join(f"{value:.6f}" for value in coords)
        lines.append(line)

    content = "\n".join(lines)
    if content:
        content += "\n"
    label_abs.write_text(content, encoding="utf-8")
    return label_abs
