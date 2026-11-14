"""Find exact and candidate near-duplicate images across source inventories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import data_root, load_sources, utc_now


def average_hash(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    except OSError:
        return None
    coefficients = cv2.dct(pixels)[:8, :8]
    average = float(np.median(coefficients[1:, 1:]))
    return sum((1 << index) for index, value in enumerate(coefficients.flat) if value >= average)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    exact: dict[str, list[dict]] = defaultdict(list)
    hashed: list[tuple[int, dict, Path]] = []
    for source in load_sources():
        raw_dir = data_root().parent / source["raw_dir"]
        manifest = data_root() / "manifests" / f"{source['id']}.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            exact[record["sha256"]].append(record)
            image_path = raw_dir / record["relative_path"]
            image_hash = average_hash(image_path)
            if image_hash is not None:
                hashed.append((image_hash, record, image_path))

    exact_groups = [records for records in exact.values() if len(records) > 1]
    near_groups = []
    buckets: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for image_hash, record, _ in hashed:
        buckets[image_hash >> 48].append((image_hash, record))
    for candidates in buckets.values():
        for index, (left_hash, left_record) in enumerate(candidates):
            for right_hash, right_record in candidates[index + 1 :]:
                distance = hamming(left_hash, right_hash)
                if 0 < distance <= 8:
                    near_groups.append(
                        {"distance": distance, "records": [left_record, right_record], "status": "review"}
                    )
    report = {
        "generated_at": utc_now(),
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_candidates": near_groups,
        "note": "Near-duplicate candidates are flagged for review and are not automatically deleted.",
    }
    (data_root() / "reports" / "duplicates.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (data_root() / "reports" / "duplicates.md").write_text(
        f"# Duplicate Review\n\nExact groups: {len(exact_groups)}\n\n"
        f"Near-duplicate candidates: {len(near_groups)}\n\n"
        "All near-duplicate candidates require review.\n",
        encoding="utf-8",
    )
