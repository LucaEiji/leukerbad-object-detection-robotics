"""Create deterministic annotation overlays for manual review."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .coco import index_coco, load_coco
from .common import data_root, load_sources
from .curate import _label_path, _read_classes

TARGETS = {
    "positive": 30,
    "hard_negative": 30,
    "cluttered_or_overlapping": 20,
    "suspected_annotation_error": 20,
}


def overlay(source: dict, record: dict, output: Path) -> None:
    raw_dir = data_root().parent / source["raw_dir"]
    image_path = raw_dir / record["relative_path"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    classes = _read_classes(raw_dir)
    label_path = _label_path(raw_dir, record["relative_path"])
    if label_path:
        for row in label_path.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            label = classes[class_id] if class_id < len(classes) else "unknown"
            if len(values) == 4:
                x, y, width, height = values
                points = [
                    ((x - width / 2) * image.width, (y - height / 2) * image.height),
                    ((x + width / 2) * image.width, (y - height / 2) * image.height),
                    ((x + width / 2) * image.width, (y + height / 2) * image.height),
                    ((x - width / 2) * image.width, (y + height / 2) * image.height),
                ]
            elif len(values) >= 6 and len(values) % 2 == 0:
                points = [(values[index] * image.width, values[index + 1] * image.height) for index in range(0, len(values), 2)]
            else:
                continue
            draw.line(points + [points[0]], fill="red", width=3)
            draw.text((points[0][0], max(0, points[0][1] - 16)), label, fill="red")
    elif (coco_result := load_coco(raw_dir)):
        for annotation in index_coco(coco_result[0]).get(Path(record["relative_path"]).name, []):
            x, y, width, height = annotation.get("bbox", [0, 0, 0, 0])
            points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
            draw.line(points + [points[0]], fill="red", width=3)
            draw.text((x, max(0, y - 16)), annotation["category_name"], fill="red")
    image.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    sources = {source["id"]: source for source in load_sources()}
    records = []
    for source in sources.values():
        path = data_root() / "manifests" / f"curated_{source['id']}.jsonl"
        if path.exists():
            records.extend((source, json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines())
    audit_errors = set()
    audit_path = data_root() / "reports" / "annotation_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_errors = {
            (source["id"], item["image"])
            for source_summary in audit["sources"]
            for source in sources.values()
            if source_summary["source_id"] == source["id"]
            for item in source_summary["errors"]
            if isinstance(item, dict) and "image" in item
        }
    rng = random.Random(args.seed)
    output_root = data_root() / "reports" / "review_gallery"
    output_root.mkdir(parents=True, exist_ok=True)
    review_lines = ["# Review Gallery", "", f"Random seed: `{args.seed}`", ""]
    for category, target in TARGETS.items():
        candidates = []
        for source, record in records:
            raw_dir = data_root().parent / source["raw_dir"]
            label_path = _label_path(raw_dir, record["relative_path"])
            object_count = (
                sum(bool(line.strip()) for line in label_path.read_text(encoding="utf-8").splitlines())
                if label_path
                else 0
            )
            labeled_candidate = (
                category == "positive" and record.get("curation_decision") == "positive"
                or category == "hard_negative" and record.get("curation_decision") == "hard_negative"
            )
            clutter_candidate = category == "cluttered_or_overlapping" and object_count >= 2
            error_candidate = (
                category == "suspected_annotation_error"
                and (source["id"], record["relative_path"]) in audit_errors
            )
            if labeled_candidate or clutter_candidate or error_candidate:
                candidates.append((source, record))
        rng.shuffle(candidates)
        category_dir = output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for previous in category_dir.glob("*.jpg"):
            previous.unlink()
        for index, (source, record) in enumerate(candidates[:target]):
            output = category_dir / f"{index:03d}_{source['id']}_{Path(record['relative_path']).stem}.jpg"
            overlay(source, record, output)
        (category_dir / "README.txt").write_text(
            f"Requested {target}; generated {min(target, len(candidates))}. Seed: {args.seed}\n",
            encoding="utf-8",
        )
        review_lines.append(f"- `{category}`: {min(target, len(candidates))} generated (requested {target})")
    review_lines.extend(
        [
            "",
            (
                "Suspected annotation errors are generated only from automated audit findings. "
                "An empty category means the audit found no such examples."
            ),
        ]
    )
    (data_root() / "reports" / "review_status.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )
