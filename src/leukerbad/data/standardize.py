"""Build an unsplit, single-class YOLO dataset with source-level provenance."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from .coco import index_coco
from .common import project_root, sha256_file, utc_now, write_json
from .inventory import IMAGE_SUFFIXES

TARGETS = {"Vape", "vape", "ecigarette"}
EXCLUDED_KAGGLE = {
    "1dcf18f5__902a7158-image_130.jpg",
    "8e65bb67__196012fb-image_16.jpg",
}
YOLO_SOURCES = {
    "roboflow_primary": "public_data/raw/roboflow_primary",
    "roboflow_additional": (
        "public_data/raw/roboflow_additional/infocommprojecttft_vape-dataset-gknul"
    ),
    "manual_photos": "manual_data/raw/manual_photos.yolo26",
    "kaggle_test_manual": "public_data/raw/kaggle/test_annotated.yolo26",
}


def validate_box(box: list[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError(f"Invalid YOLO box: {box}")
    x, y, w, h = box
    if w <= 0 or h <= 0 or min(x - w / 2, y - h / 2) < -1e-6:
        raise ValueError(f"Nonpositive or out-of-bounds YOLO box: {box}")
    if max(x + w / 2, y + h / 2) > 1 + 1e-6:
        raise ValueError(f"Out-of-bounds YOLO box: {box}")


def filter_yolo(text: str, names: list[str]) -> tuple[list[list[float]], Counter]:
    boxes = []
    removed: Counter = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        class_id = int(fields[0])
        if not 0 <= class_id < len(names):
            raise ValueError(f"Unknown class ID: {class_id}")
        name = names[class_id]
        if name not in TARGETS:
            removed[name] += 1
            continue
        box = list(map(float, fields[1:]))
        if len(box) != 4:
            if len(box) < 6 or len(box) % 2 or not all(
                math.isfinite(v) and 0 <= v <= 1 for v in box
            ):
                raise ValueError(f"Invalid YOLO polygon: {box}")
            xs, ys = box[::2], box[1::2]
            box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                   max(xs) - min(xs), max(ys) - min(ys)]
        validate_box(box)
        boxes.append(box)
    return boxes, removed


def coco_boxes(annotations: list[dict], width: int, height: int) -> tuple[list, Counter]:
    boxes = []
    removed: Counter = Counter()
    for annotation in annotations:
        name = annotation["category_name"]
        if name not in TARGETS:
            removed[name] += 1
            continue
        x, y, w, h = annotation["bbox"]
        box = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
        validate_box(box)
        boxes.append(box)
    return boxes, removed


def images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def build(root: Path, output: Path) -> dict:
    # A fresh directory prevents mixing old labels with a new export.
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    records = []
    exclusions = []
    counts: dict[str, Counter] = defaultdict(Counter)
    removed: dict[str, Counter] = defaultdict(Counter)
    converted_polygons: Counter = Counter()
    manual_hashes: dict[str, str] = {}
    kaggle_test_names = {p.stem for p in images(root / "public_data/raw/kaggle/extracted/test")}

    def add(source: str, image: Path, annotation: Path | None, boxes: list, dropped: Counter,
            source_split: str, group: str) -> None:
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        digest = sha256_file(image)
        relative = str(image.relative_to(root))
        name = f"{source}__{image.stem}{image.suffix.lower()}"
        records.append({
            "source_id": source, "source_image": relative,
            "source_annotation": str(annotation.relative_to(root)) if annotation else None,
            "source_split": source_split, "split": None, "origin_group": group,
            "sha256": digest, "width": width, "height": height,
            "image": f"images/{name}", "label": f"labels/{Path(name).stem}.txt",
            "target_boxes": boxes, "target_instances": len(boxes),
            "negative": not boxes, "removed_annotations": dict(dropped),
        })
        counts[source].update(images=1, positive_images=bool(boxes),
                              negative_images=not boxes, target_instances=len(boxes))
        removed[source].update(dropped)
        if source == "manual_photos":
            manual_hashes[digest] = relative

    for source, directory in YOLO_SOURCES.items():
        source_root = root / directory
        config = yaml.safe_load((source_root / "data.yaml").read_text())
        names = config["names"]
        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]
        for image in images(source_root):
            split = image.relative_to(source_root).parts[0]
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            if not label.exists():
                raise ValueError(f"Missing label file (not assumed negative): {image}")
            origin = image.name.split(".rf.")[0]
            if source == "kaggle_test_manual":
                original = origin.removesuffix("_jpg")
                if original not in kaggle_test_names:
                    duplicate = manual_hashes.get(sha256_file(image))
                    if not duplicate:
                        raise ValueError(f"Unmatched extra image in Kaggle test export: {image}")
                    exclusions.append({"source_image": str(image.relative_to(root)),
                                       "reason": "exact_duplicate_of_manual_photo",
                                       "retained_source_image": duplicate})
                    continue
                group = f"kaggle_test/{original}"
            else:
                group = f"{source}/{origin}"
            text = label.read_text()
            converted_polygons[source] += sum(
                len(line.split()) != 5 and names[int(line.split()[0])] in TARGETS
                for line in text.splitlines() if line.strip()
            )
            boxes, dropped = filter_yolo(text, names)
            add(source, image, label, boxes, dropped, split, group)
    matched = {r["origin_group"].split("/", 1)[1] for r in records
               if r["source_id"] == "kaggle_test_manual"}
    if matched != kaggle_test_names:
        raise ValueError("Manual Kaggle export does not cover the original test image identities")

    kaggle = root / "public_data/raw/kaggle/extracted"
    annotation = kaggle / "train_annotations.json"
    indexed = index_coco(json.loads(annotation.read_text()))
    for image in images(kaggle / "train"):
        if image.name in EXCLUDED_KAGGLE:
            exclusions.append({"source_image": str(image.relative_to(root)),
                               "reason": "user_requested_out_of_bounds_coco_bbox_exclusion",
                               "source_annotation": str(annotation.relative_to(root)),
                               "annotations": indexed[image.name]})
            continue
        if image.name not in indexed:
            raise ValueError(f"Missing COCO image record: {image}")
        with Image.open(image) as im:
            boxes, dropped = coco_boxes(indexed[image.name], *im.size)
        add("kaggle_train", image, annotation, boxes, dropped, "train",
            f"kaggle_train/{image.stem}")

    negatives = root / "public_data/raw/open_images_v7_negatives"
    negative_categories: Counter = Counter()
    for image in images(negatives / "images"):
        negative_categories[image.parent.name] += 1
        add("open_images_negatives", image, None, [], Counter(), "validation_source",
            f"open_images/{image.stem}")

    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hashes[record["sha256"]].append(record["source_image"])
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    summary = {
        "generated_at": utc_now(), "format": "YOLO detection", "names": {0: "vape"},
        "retained_source_classes": sorted(TARGETS), "split_status": "unassigned",
        "sources": {s: {**dict(c), "removed_annotations": dict(removed[s]),
                        "target_polygons_converted_to_boxes": converted_polygons[s]}
                    for s, c in counts.items()},
        "totals": dict(sum(counts.values(), Counter())),
        "exclusions": exclusions, "open_images_categories": dict(negative_categories),
        "origin_groups": len({r["origin_group"] for r in records}),
        "remaining_exact_duplicate_groups": duplicate_groups,
        "original_kaggle_test_images_omitted": len(kaggle_test_names),
    }
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    for record in records:
        shutil.copy2(root / record["source_image"], output / record["image"])
        text = "".join("0 " + " ".join(format(v, ".12g") for v in box) + "\n"
                       for box in record["target_boxes"])
        (output / record["label"]).write_text(text)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    write_json(output / "summary.json", summary)
    (output / "classes.yaml").write_text("nc: 1\nnames:\n  0: vape\n")
    report = ["# Standardized Vape Dataset", "", f"Generated: {summary['generated_at']}", "",
              "Single-class YOLO detection: `0: vape`. Final splits are unassigned.", "",
              "| Source | Images | Positive | Negative | Vape boxes |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for source, c in counts.items():
        report.append(f"| {source} | {c['images']} | {c['positive_images']} | "
                      f"{c['negative_images']} | {c['target_instances']} |")
    report += ["", "## Policy and lineage", "",
               "- Only `Vape`, `vape`, and `ecigarette` boxes survive; all map to `0: vape`.",
               "- All negative images are retained with empty label files.",
               "- Original unlabeled Kaggle test copies are omitted; manual labels take precedence.",
               "- Two Kaggle train images are excluded for the previously flagged out-of-bounds boxes.",
               "- The extra Kaggle-export photo is omitted because it is byte-identical to a manual photo.",
               "- Source files are unchanged. Exclusions are from this export, not raw deletion.",
               "- No resizing, new augmentation, or global split assignment is performed.",
               "- Retained YOLO polygons are converted to enclosing axis-aligned detection boxes.",
               "- Source splits and filename-based origin groups are retained in manifest.jsonl.",
               "- Existing augmentation variants are retained; origin groups are not unique-scene counts.",
               "- classes.yaml defines classes only; training data.yaml awaits the split decision.",
               "- summary.json records removed class counts, exclusions, and exact duplicates.",
               "- Image readability, label pairing, target geometry, and Kaggle test coverage are checked.",
               "", "## Exact summary", "", "```json", json.dumps(summary, indent=2), "```", ""]
    (output / "README.md").write_text("\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="public_data/exports/vape_yolo")
    args = parser.parse_args()
    root = project_root()
    summary = build(root, root / args.output)
    report_dir = root / "public_data/reports"
    write_json(report_dir / "standardization_summary.json", summary)
    shutil.copy2(root / args.output / "README.md", report_dir / "standardization.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
