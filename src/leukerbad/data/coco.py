"""Small helpers for COCO-style global annotation files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_coco(raw_dir: Path) -> tuple[dict[str, Any], Path] | None:
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if all(key in data for key in ("images", "annotations", "categories")):
            return data, path
    return None


def index_coco(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    image_names = {item["id"]: Path(item["file_name"]).name for item in data["images"]}
    indexed: dict[str, list[dict[str, Any]]] = {name: [] for name in image_names.values()}
    categories = {item["id"]: item["name"] for item in data["categories"]}
    for annotation in data["annotations"]:
        name = image_names.get(annotation["image_id"])
        if name is not None:
            indexed[name].append({**annotation, "category_name": categories.get(annotation["category_id"], "unknown")})
    return indexed
