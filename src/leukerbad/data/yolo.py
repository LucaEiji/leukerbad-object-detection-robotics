"""Shared helpers for Ultralytics-style dataset exports."""

from __future__ import annotations

from pathlib import Path

import yaml


def read_classes(dataset_dir: Path) -> list[str]:
    """Read class names from a dataset's ``data.yaml`` file."""
    config_path = dataset_dir / "data.yaml"
    if not config_path.exists():
        return []
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names = config.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def label_path(dataset_dir: Path, relative_image: str) -> Path | None:
    """Return an image's sibling YOLO label file when it exists."""
    image_path = dataset_dir / relative_image
    labels = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return labels if labels.exists() else None
