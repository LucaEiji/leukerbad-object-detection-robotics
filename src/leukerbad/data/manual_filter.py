"""Apply the requested manual cleanup rules to raw source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import data_root, utc_now
from .curate import _read_classes

REFERENCE_IMAGES = [
    "Screenshot-2024-04-03-190615_jpg.rf.43c331dafb47c594d85571a0c83a821a.jpg",
    "Screenshot-2024-04-03-190859_jpg.rf.6d7ae93453d906307f33d6e8ddc2be60.jpg",
    "Screenshot-2024-04-03-191628_jpg.rf.3c5d9507e3f816304ca463ab3fd25a38.jpg",
]
# Manual quality decision: these reference pictures were judged unsuitable by the user.
# The largest reference box area defines an inclusive cutoff, not a model-derived threshold.
REFERENCE_AREAS = [0.046875 * 0.0640625, 0.0484375 * 0.059375, 0.0359375 * 0.028125]
VAPE_AREA_THRESHOLD = max(REFERENCE_AREAS)
EXPLICIT_ADDITIONAL = [
    "b9176c46-1fb5-4558-b52f-cdb60eedb853-JORDANVAPEPENS_BTM_0007_frame_0_webp.rf.2f89a2b0e40a440ad7c333b441d8d383.jpg",
    "b9176c46-1fb5-4558-b52f-cdb60eedb853-JORDANVAPEPENS_BTM_0007_frame_0_webp.rf.6d5c0833c40f88becdb8a47017311b6e.jpg",
    "b9176c46-1fb5-4558-b52f-cdb60eedb853-JORDANVAPEPENS_BTM_0007_frame_0_webp.rf.22197846a73e52ee31c46fce56e693d1.jpg",
]


def _annotation_area(fields: list[str]) -> float | None:
    if len(fields) == 5:
        return float(fields[3]) * float(fields[4])
    if len(fields) >= 7 and (len(fields) - 1) % 2 == 0:
        points = [(float(fields[index]), float(fields[index + 1])) for index in range(1, len(fields), 2)]
        return abs(
            sum(points[index][0] * points[(index + 1) % len(points)][1] for index in range(len(points)))
            - sum(points[index][1] * points[(index + 1) % len(points)][0] for index in range(len(points)))
        ) / 2
    return None


def manual_exclusion_reason(
    source: str, filename: str, text: str, classes: list[str]
) -> str | None:
    """Reproduce historical raw cleanup without requiring a local exclusion ledger.

    These source-specific image removals precede target-label filtering. Other
    non-target-only images remain eligible as intentional negative examples.
    """
    if source == "roboflow_additional" and filename in EXPLICIT_ADDITIONAL:
        return "user_explicitly_marked_misannotated"
    if source != "roboflow_primary":
        return None
    annotations = [line.split() for line in text.splitlines() if line.strip()]
    names = [classes[int(fields[0])].lower() for fields in annotations]
    if names and all(name == "cigarette" for name in names):
        return "primary_cigarette_only"
    for fields, name in zip(annotations, names):
        if name == "vape":
            area = _annotation_area(fields)
            if area is not None and area <= VAPE_AREA_THRESHOLD:
                return "primary_vape_area_at_or_below_reference_threshold"
    return None


def _remove_pair(image_path: Path, reason: str, records: list[dict]) -> None:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    record = {
        "timestamp": utc_now(),
        "source_image": str(image_path.relative_to(data_root().parent)),
        "source_label": str(label_path.relative_to(data_root().parent)) if label_path.exists() else None,
        "reason": reason,
    }
    image_path.unlink()
    if label_path.exists():
        label_path.unlink()
    records.append(record)


def filter_primary(records: list[dict]) -> None:
    raw_dir = data_root() / "raw" / "roboflow_primary"
    classes = _read_classes(raw_dir)
    for image_path in sorted(raw_dir.glob("*/images/*")):
        if not image_path.is_file():
            continue
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        reason = manual_exclusion_reason(
            "roboflow_primary", image_path.name,
            label_path.read_text(encoding="utf-8"), classes,
        )
        if reason:
            _remove_pair(image_path, reason, records)


def filter_explicit_additional(records: list[dict]) -> None:
    raw_dir = data_root() / "raw" / "roboflow_additional" / "infocommprojecttft_vape-dataset-gknul" / "train" / "images"
    for filename in EXPLICIT_ADDITIONAL:
        image_path = raw_dir / filename
        if image_path.exists():
            reason = manual_exclusion_reason("roboflow_additional", filename, "", [])
            if reason:
                _remove_pair(image_path, reason, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    exclusion_log = data_root() / "manifests" / "exclusions.jsonl"
    records = (
        [json.loads(line) for line in exclusion_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if exclusion_log.exists()
        else []
    )
    filter_primary(records)
    filter_explicit_additional(records)
    with exclusion_log.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "generated_at": utc_now(),
        "threshold_metric": "normalized annotation area",
        "vape_area_threshold_inclusive": VAPE_AREA_THRESHOLD,
        "reference_images": REFERENCE_IMAGES,
        "removed_count": sum(record["reason"] != "user_requested_file_not_found" for record in records),
        "records": len(records),
    }
    (data_root() / "reports" / "manual_exclusions.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Removed {summary['removed_count']} image/label pairs")
