"""Audit YOLO annotations while preserving the original files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .coco import index_coco, load_coco
from .common import data_root, load_sources, utc_now
from .yolo import label_path, read_classes


def audit_source(source: dict[str, Any]) -> dict[str, Any]:
    raw_dir = data_root().parent / source["raw_dir"]
    classes = read_classes(raw_dir)
    coco_index = {}
    coco_result = load_coco(raw_dir)
    if coco_result:
        coco_index = index_coco(coco_result[0])
    inventory_path = data_root() / "manifests" / f"{source['id']}.jsonl"
    summary = {
        "source_id": source["id"],
        "images": 0,
        "instances": 0,
        "valid_annotations": 0,
        "unannotated_images": 0,
        "class_instances": {},
        "errors": [],
    }
    if not inventory_path.exists():
        summary["errors"].append("inventory_missing")
        return summary
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        image = json.loads(line)
        summary["images"] += 1
        annotation_path = label_path(raw_dir, image["relative_path"])
        if not annotation_path and Path(image["relative_path"]).name in coco_index:
            image_errors = []
            for annotation in coco_index[Path(image["relative_path"]).name]:
                bbox = annotation.get("bbox", [])
                if len(bbox) != 4:
                    image_errors.append({"error": "invalid_coco_bbox"})
                    continue
                x, y, width, height = bbox
                valid = (
                    width > 0
                    and height > 0
                    and x >= 0
                    and y >= 0
                    and x + width <= image["width"]
                    and y + height <= image["height"]
                )
                if not valid:
                    image_errors.append({"error": "out_of_bounds_coco_bbox", "bbox": bbox})
                    continue
                summary["instances"] += 1
                name = annotation["category_name"]
                summary["class_instances"][name] = summary["class_instances"].get(name, 0) + 1
            if image_errors:
                summary["errors"].append({"image": image["relative_path"], "errors": image_errors})
            else:
                summary["valid_annotations"] += 1
            continue
        if not annotation_path:
            if coco_result:
                summary["unannotated_images"] += 1
            else:
                summary["errors"].append({"image": image["relative_path"], "error": "missing_label"})
            continue
        image_errors = []
        for line_number, row in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = row.split()
            if len(fields) != 5 and (len(fields) < 7 or (len(fields) - 1) % 2):
                image_errors.append({"line": line_number, "error": "expected_five_yolo_fields"})
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                image_errors.append({"line": line_number, "error": "non_numeric_yolo_value"})
                continue
            if class_id < 0 or class_id >= len(classes):
                image_errors.append({"line": line_number, "error": "invalid_class_id", "class_id": class_id})
            if len(values) == 4:
                valid_geometry = all(0 <= value <= 1 for value in values) and values[2] > 0 and values[3] > 0
            else:
                valid_geometry = len(values) >= 6 and all(0 <= value <= 1 for value in values)
            if not valid_geometry:
                image_errors.append({"line": line_number, "error": "out_of_bounds_or_invalid_geometry"})
            else:
                summary["instances"] += 1
                class_name = classes[class_id] if 0 <= class_id < len(classes) else "unknown"
                summary["class_instances"][class_name] = summary["class_instances"].get(class_name, 0) + 1
        if image_errors:
            summary["errors"].append({"image": image["relative_path"], "errors": image_errors})
        else:
            summary["valid_annotations"] += 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", help="Audit only one source")
    args = parser.parse_args()
    sources = load_sources()
    if args.source_id:
        sources = [source for source in sources if source["id"] == args.source_id]
        if not sources:
            parser.error(f"Unknown source ID: {args.source_id}")
    summaries = [audit_source(source) for source in sources]
    report = {"generated_at": utc_now(), "sources": summaries}
    report_path = data_root() / "reports" / "annotation_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (data_root() / "reports" / "annotation_audit.md").write_text(
        "# Annotation Audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
