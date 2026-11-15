"""Build an unsplit, single-class YOLO dataset with source-level provenance."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from .coco import index_coco
from .common import project_root, sha256_file, utc_now, write_json
from .files import images
from .manual_filter import (
    EXPLICIT_ADDITIONAL,
    REFERENCE_IMAGES,
    VAPE_AREA_THRESHOLD,
    manual_exclusion_reason,
)

TARGETS = {"Vape", "vape", "ecigarette"}
EXCLUDED_KAGGLE = {
    "1dcf18f5__902a7158-image_130.jpg",
    "8e65bb67__196012fb-image_16.jpg",
}
YOLO_SOURCES = {
    "roboflow_primary": "public_data/raw/roboflow_primary",
    "roboflow_additional": "public_data/raw/roboflow_additional/infocommprojecttft_vape-dataset-gknul",
    "manual_photos": "manual_data/raw/manual_photos.yolo26",
    "kaggle_test_manual": "public_data/raw/kaggle/test_annotated.yolo26",
}


def validate_box(box: list[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError(f"Invalid YOLO box: {box}")
    x, y, width, height = box
    if width <= 0 or height <= 0 or min(x - width / 2, y - height / 2) < -1e-6:
        raise ValueError(f"Nonpositive or out-of-bounds YOLO box: {box}")
    if max(x + width / 2, y + height / 2) > 1 + 1e-6:
        raise ValueError(f"Out-of-bounds YOLO box: {box}")


def filter_yolo(text: str, names: list[str]) -> tuple[list[list[float]], Counter]:
    boxes: list[list[float]] = []
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
        values = list(map(float, fields[1:]))
        if len(values) != 4:
            if len(values) < 6 or len(values) % 2 or not all(
                math.isfinite(value) and 0 <= value <= 1 for value in values
            ):
                raise ValueError(f"Invalid YOLO polygon: {values}")
            xs, ys = values[::2], values[1::2]
            values = [
                (min(xs) + max(xs)) / 2,
                (min(ys) + max(ys)) / 2,
                max(xs) - min(xs),
                max(ys) - min(ys),
            ]
        validate_box(values)
        boxes.append(values)
    return boxes, removed


def coco_boxes(annotations: list[dict], width: int, height: int) -> tuple[list[list[float]], Counter]:
    boxes: list[list[float]] = []
    removed: Counter = Counter()
    for annotation in annotations:
        name = annotation["category_name"]
        if name not in TARGETS:
            removed[name] += 1
            continue
        x, y, box_width, box_height = annotation["bbox"]
        box = [
            (x + box_width / 2) / width,
            (y + box_height / 2) / height,
            box_width / width,
            box_height / height,
        ]
        validate_box(box)
        boxes.append(box)
    return boxes, removed


@dataclass
class BuildState:
    root: Path
    records: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    removed: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    converted_polygons: Counter = field(default_factory=Counter)
    manual_hashes: dict[str, str] = field(default_factory=dict)
    kaggle_test_names: set[str] = field(default_factory=set)

    def add(
        self,
        source: str,
        image: Path,
        annotation: Path | None,
        boxes: list[list[float]],
        dropped: Counter,
        source_split: str,
        group: str,
    ) -> None:
        with Image.open(image) as decoded:
            width, height = decoded.size
            decoded.verify()
        digest = sha256_file(image)
        relative = str(image.relative_to(self.root))
        filename = f"{source}__{image.stem}{image.suffix.lower()}"
        self.records.append(
            {
                "source_id": source,
                "source_image": relative,
                "source_annotation": str(annotation.relative_to(self.root)) if annotation else None,
                "source_split": source_split,
                "split": None,
                "origin_group": group,
                "sha256": digest,
                "width": width,
                "height": height,
                "image": f"images/{filename}",
                "label": f"labels/{Path(filename).stem}.txt",
                "target_boxes": boxes,
                "target_instances": len(boxes),
                "negative": not boxes,
                "removed_annotations": dict(dropped),
            }
        )
        self.counts[source].update(
            images=1,
            positive_images=bool(boxes),
            negative_images=not boxes,
            target_instances=len(boxes),
        )
        self.removed[source].update(dropped)
        if source == "manual_photos":
            self.manual_hashes[digest] = relative


def _class_names(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config["names"]
    return [str(names[index]) for index in range(len(names))] if isinstance(names, dict) else names


def _exclude(state: BuildState, image: Path, label: Path, reason: str) -> None:
    record = {
        "source_image": str(image.relative_to(state.root)),
        "reason": reason,
    }
    if reason == "exact_duplicate_of_manual_photo":
        record["retained_source_image"] = state.manual_hashes.get(sha256_file(image))
    else:
        record["source_annotation"] = str(label.relative_to(state.root))
    state.exclusions.append(record)


def _collect_yolo(state: BuildState, source: str, source_directory: Path) -> None:
    classes = _class_names(source_directory / "data.yaml")
    for image in images(source_directory):
        split = image.relative_to(source_directory).parts[0]
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        reason = manual_exclusion_reason(source, image.name, "", classes)
        if reason:
            _exclude(state, image, label, reason)
            continue
        if not label.exists():
            raise ValueError(f"Missing label file (not assumed negative): {image}")

        text = label.read_text(encoding="utf-8")
        reason = manual_exclusion_reason(source, image.name, text, classes)
        if reason:
            _exclude(state, image, label, reason)
            continue

        origin = image.name.split(".rf.")[0]
        if source == "kaggle_test_manual":
            original = origin.removesuffix("_jpg")
            if original not in state.kaggle_test_names:
                duplicate = state.manual_hashes.get(sha256_file(image))
                if duplicate is None:
                    raise ValueError(f"Unmatched extra image in Kaggle test export: {image}")
                _exclude(state, image, label, "exact_duplicate_of_manual_photo")
                continue
            group = f"kaggle_test/{original}"
        else:
            group = f"{source}/{origin}"

        state.converted_polygons[source] += sum(
            len(line.split()) != 5 and classes[int(line.split()[0])] in TARGETS
            for line in text.splitlines()
            if line.strip()
        )
        boxes, dropped = filter_yolo(text, classes)
        state.add(source, image, label, boxes, dropped, split, group)


def _collect_kaggle_train(state: BuildState) -> None:
    kaggle = state.root / "public_data/raw/kaggle/extracted"
    annotation = kaggle / "train_annotations.json"
    indexed = index_coco(json.loads(annotation.read_text(encoding="utf-8")))
    for image in images(kaggle / "train"):
        if image.name in EXCLUDED_KAGGLE:
            state.exclusions.append(
                {
                    "source_image": str(image.relative_to(state.root)),
                    "reason": "user_requested_out_of_bounds_coco_bbox_exclusion",
                    "source_annotation": str(annotation.relative_to(state.root)),
                    "annotations": indexed[image.name],
                }
            )
            continue
        if image.name not in indexed:
            raise ValueError(f"Missing COCO image record: {image}")
        with Image.open(image) as decoded:
            boxes, dropped = coco_boxes(indexed[image.name], *decoded.size)
        state.add("kaggle_train", image, annotation, boxes, dropped, "train", f"kaggle_train/{image.stem}")


def _collect_open_images(state: BuildState) -> Counter:
    counts: Counter = Counter()
    image_root = state.root / "public_data/raw/open_images_v7_negatives/images"
    provenance_path = state.root / "public_data/manifests/open_images_negatives.jsonl"
    provenance = {}
    if provenance_path.exists():
        provenance = {
            item["open_images_id"]: item
            for item in map(json.loads, provenance_path.read_text(encoding="utf-8").splitlines())
        }
    for image in images(image_root):
        counts[image.parent.name] += 1
        source_split = provenance.get(image.stem, {}).get("image_split", "unknown")
        state.add(
            "open_images_negatives", image, None, [], Counter(),
            source_split, f"open_images/{image.stem}",
        )
    return counts


def _summary(state: BuildState, open_images_categories: Counter) -> dict:
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in state.records:
        hashes[record["sha256"]].append(record["source_image"])
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "generated_at": utc_now(),
        "format": "YOLO detection",
        "names": {0: "vape"},
        "retained_source_classes": sorted(TARGETS),
        "split_status": "unassigned",
        "sources": {
            source: {
                **dict(count),
                "removed_annotations": dict(state.removed[source]),
                "target_polygons_converted_to_boxes": state.converted_polygons[source],
            }
            for source, count in state.counts.items()
        },
        "totals": dict(sum(state.counts.values(), Counter())),
        "exclusions": state.exclusions,
        "open_images_categories": dict(open_images_categories),
        "origin_groups": len({record["origin_group"] for record in state.records}),
        "remaining_exact_duplicate_groups": duplicate_groups,
        "original_kaggle_test_images_omitted": len(state.kaggle_test_names),
        "manual_removal_policy": {
            "implementation": "leukerbad.data.manual_filter.manual_exclusion_reason",
            "requires_local_exclusion_ledger": False,
            "primary_cigarette_only": True,
            "primary_vape_area_threshold_inclusive": VAPE_AREA_THRESHOLD,
            "threshold_rationale": (
                "Manual dataset-quality decision: the user judged three reference pictures unsuitable. "
                "Their largest normalized vape box area defines the inclusive cutoff; it is not "
                "statistically fitted or model-derived."
            ),
            "area_metric": "normalized box area or polygon shoelace area",
            "reference_images": REFERENCE_IMAGES,
            "explicit_additional_filenames": EXPLICIT_ADDITIONAL,
            "excluded_kaggle_train_filenames": sorted(EXCLUDED_KAGGLE),
            "applied_this_build_by_reason": dict(Counter(item["reason"] for item in state.exclusions)),
        },
    }


def _write_export(state: BuildState, output: Path, summary: dict) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    image_directory = output / "images"
    label_directory = output / "labels"
    image_directory.mkdir(parents=True)
    label_directory.mkdir()
    for record in state.records:
        shutil.copy2(state.root / record["source_image"], output / record["image"])
        labels = "".join(
            "0 " + " ".join(format(value, ".12g") for value in box) + "\n"
            for box in record["target_boxes"]
        )
        (output / record["label"]).write_text(labels, encoding="utf-8")

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in state.records:
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(output / "summary.json", summary)
    (output / "classes.yaml").write_text("nc: 1\nnames:\n  0: vape\n", encoding="utf-8")
    report = [
        "# Standardized Vape Dataset",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "Single-class YOLO detection: `0: vape`. Splits remain unassigned.",
        "",
        "| Source | Images | Positive | Negative | Vape boxes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, counts in summary["sources"].items():
        report.append(
            f"| {source} | {counts['images']} | {counts['positive_images']} | "
            f"{counts['negative_images']} | {counts['target_instances']} |"
        )
    report.extend(
        [
            "",
            (
                "Only `Vape`, `vape`, and `ecigarette` annotations are retained. Negative images "
                "receive empty label files. Source files are unchanged; split assignment is deferred."
            ),
            (
                "Historical source exclusions, class counts, duplicate groups, and record-level "
                "provenance are in `summary.json` and `manifest.jsonl`."
            ),
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(report), encoding="utf-8")


def build(root: Path, output: Path) -> dict:
    """Build a fresh YOLO export without modifying source files."""
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    state = BuildState(root=root)
    kaggle_test = root / "public_data/raw/kaggle/extracted/test"
    state.kaggle_test_names = {image.stem for image in images(kaggle_test)}

    for source, relative_directory in YOLO_SOURCES.items():
        _collect_yolo(state, source, root / relative_directory)
    matched = {
        record["origin_group"].split("/", 1)[1]
        for record in state.records
        if record["source_id"] == "kaggle_test_manual"
    }
    if matched != state.kaggle_test_names:
        raise ValueError("Manual Kaggle export does not cover the original test image identities")

    _collect_kaggle_train(state)
    open_images_categories = _collect_open_images(state)
    summary = _summary(state, open_images_categories)
    _write_export(state, output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="public_data/exports/vape_yolo")
    args = parser.parse_args()
    root = project_root()
    summary = build(root, root / args.output)
    reports = root / "public_data/reports"
    write_json(reports / "standardization_summary.json", summary)
    shutil.copy2(root / args.output / "README.md", reports / "standardization.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
