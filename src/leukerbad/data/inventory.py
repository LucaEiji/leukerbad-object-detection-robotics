"""Inventory downloaded images and source annotations into JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .coco import index_coco, load_coco
from .common import data_root, load_sources, sha256_file, utc_now

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIXES = {".json", ".txt", ".xml"}


def _image_info(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            return image.width, image.height, image.format
    except (OSError, ValueError):
        return None, None, None


def _annotation_candidates(image_path: Path) -> list[str]:
    candidates = [
        str(candidate.relative_to(image_path.parent))
        for candidate in sorted(image_path.parent.glob(f"{image_path.stem}.*"))
        if candidate.suffix.lower() in ANNOTATION_SUFFIXES
    ]
    # Roboflow YOLO exports keep images and labels in sibling directories.
    sibling_labels = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if sibling_labels.exists():
        candidates.append(str(sibling_labels.relative_to(image_path.parent.parent.parent)))
    return sorted(set(candidates))


def inventory_source(source: dict, root: Path) -> tuple[list[dict], dict]:
    source_dir = root.parent / source["raw_dir"]
    images = []
    annotation_files = []
    coco_index = {}
    coco_file = None
    if source_dir.exists():
        annotation_files = [
            path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES
        ]
        coco_result = load_coco(source_dir)
        if coco_result:
            coco_index = index_coco(coco_result[0])
            coco_file = str(coco_result[1].relative_to(source_dir))
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            width, height, image_format = _image_info(path)
            annotation_candidates = _annotation_candidates(path)
            if path.name in coco_index and coco_file:
                annotation_candidates.append(coco_file)
            images.append(
                {
                    "source_id": source["id"],
                    "relative_path": str(path.relative_to(source_dir)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "annotation_candidates": sorted(set(annotation_candidates)),
                    "annotation_status": "candidate_present"
                    if annotation_candidates
                    else "unknown_empty_or_global",
                }
            )
    summary = {
        "source_id": source["id"],
        "raw_dir": source["raw_dir"],
        "image_files": len(images),
        "annotation_files": len(annotation_files),
        "readable_images": sum(item["width"] is not None for item in images),
    }
    return images, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", help="Inventory only one configured source")
    args = parser.parse_args()

    selected = load_sources()
    if args.source_id:
        selected = [source for source in selected if source["id"] == args.source_id]
        if not selected:
            parser.error(f"Unknown source ID: {args.source_id}")

    manifest_dir = data_root() / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source in selected:
        records, summary = inventory_source(source, data_root())
        manifest_path = manifest_dir / f"{source['id']}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        summaries.append(summary)

    summary_path = manifest_dir / "inventory_summary.json"
    summary_path.write_text(
        json.dumps({"generated_at": utc_now(), "sources": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_path)
