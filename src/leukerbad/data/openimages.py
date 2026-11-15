"""Download a balanced, attributed negative-image subset from Open Images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from .common import data_root, load_sources, sha256_file, utc_now, write_json

CLASSES = ["eraser", "pen", "perfume", "spice rack", "tin can", "torch"]
SELECTION_ORDER = ["eraser", "spice rack", "torch", "perfume", "pen", "tin can"]
PER_CLASS = 100
CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
VALIDATION_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
TRAIN_LABEL_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv"
TRAIN_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
ALL_TRAIN_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-with-labels-with-rotation.csv"
S3_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
LANDING_URL = "https://storage.googleapis.com/openimages/web/index.html"
FALLBACK_MULTIPLIER = 5


def _download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length", 0))
        print(f"Downloading {path.name} ({expected / (1024 * 1024):.0f} MiB).", flush=True)
        received = 0
        next_report = 100 * 1024 * 1024
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    received += len(chunk)
                    if received >= next_report:
                        print(f"  {received / (1024 * 1024):.0f} MiB received.", flush=True)
                        next_report += 100 * 1024 * 1024
    temporary.replace(path)


def _load_classes(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as source:
        return {row["DisplayName"].strip().lower(): row["LabelName"] for row in csv.DictReader(source)}


def _record(source_id: str, class_name: str, image_id: str, relative: Path, metadata: dict, path: Path) -> dict:
    pixel_url = metadata.get("PixelURL") or S3_IMAGE_URL.format(
        split=metadata.get("Subset", "unknown"), image_id=image_id
    )
    return {
        "source_id": source_id,
        "negative_class": class_name,
        "open_images_id": image_id,
        "relative_path": str(relative),
        "sha256": sha256_file(path),
        "license": metadata.get("License", "unknown"),
        "author": metadata.get("Author", "unknown"),
        "title": metadata.get("Title", "unknown"),
        "original_url": metadata.get("OriginalURL", "unknown"),
        "landing_url": metadata.get("OriginalLandingURL", LANDING_URL),
        "thumbnail_url": metadata.get("Thumbnail300KURL", "unknown"),
        "image_url": pixel_url,
        "image_split": metadata.get("Subset", "unknown"),
        "metadata_source": metadata.get("MetadataSource", "unknown"),
    }


def _collect_candidates(
    label_url: str,
    class_ids: dict[str, str],
    available_ids: set[str],
    existing_ids: set[str],
    deficits: dict[str, int],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    reverse = {mid: name for name, mid in class_ids.items()}
    candidates = {name: set() for name in CLASSES}
    fallbacks = {name: set() for name in CLASSES}
    selected_ids: set[str] = set()
    rows_seen = 0
    started = time.monotonic()
    with requests.get(label_url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        lines = (line.decode("utf-8") for line in response.iter_lines() if line)
        for row in csv.DictReader(lines):
            rows_seen += 1
            name = reverse.get(row["LabelName"])
            image_id = row["ImageID"]
            if (
                name
                and row.get("Confidence") == "1"
                and image_id not in existing_ids
                and image_id not in selected_ids
            ):
                if image_id in available_ids and len(candidates[name]) < deficits[name]:
                    candidates[name].add(image_id)
                    selected_ids.add(image_id)
                elif (
                    image_id not in available_ids
                    and len(fallbacks[name])
                    < max(0, deficits[name] - len(candidates[name])) * FALLBACK_MULTIPLIER
                ):
                    fallbacks[name].add(image_id)
            if rows_seen % 1_000_000 == 0:
                counts = ", ".join(
                    f"{key} bucket {len(candidates[key])}/{deficits[key]}, "
                    f"fallbacks {len(fallbacks[key])}"
                    for key in CLASSES
                )
                print(f"Scanned {rows_seen:,} labels ({(time.monotonic() - started) / 60:.1f} min): {counts}", flush=True)
            enough_candidates = all(
                len(candidates[name]) >= deficits[name]
                or len(fallbacks[name])
                >= (deficits[name] - len(candidates[name])) * FALLBACK_MULTIPLIER
                for name in CLASSES
            )
            if enough_candidates:
                break
    return candidates, fallbacks


def _metadata_dict(row: dict, metadata_source: str) -> dict:
    return {
        "Subset": row.get("Subset", "unknown"),
        "OriginalURL": row.get("OriginalURL", "unknown"),
        "OriginalLandingURL": row.get("OriginalLandingURL", "unknown"),
        "License": row.get("License", "unknown"),
        "Author": row.get("Author", "unknown"),
        "Title": row.get("Title", "unknown"),
        "Thumbnail300KURL": row.get("Thumbnail300KURL", "unknown"),
        "MetadataSource": metadata_source,
    }


def _download_image(url: str, path: Path) -> None:
    temporary = path.with_suffix(".jpg.part")
    error: Exception | None = None
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return
        except Exception as caught:  # noqa: BLE001 - preserve provider-specific failures after retries.
            error = caught
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    if error is not None:
        raise error


def download_open_images(source: dict) -> dict:
    destination = data_root().parent / source["raw_dir"]
    metadata_dir = destination / "metadata"
    class_path = metadata_dir / "oidv7-class-descriptions-boxable.csv"
    validation_metadata_path = metadata_dir / "validation-images-with-rotation.csv"
    train_metadata_path = metadata_dir / "train-images-boxable-with-rotation.csv"
    _download_file(CLASS_URL, class_path)
    _download_file(VALIDATION_METADATA_URL, validation_metadata_path)
    _download_file(TRAIN_METADATA_URL, train_metadata_path)

    class_ids_all = _load_classes(class_path)
    class_ids = {name: class_ids_all[name] for name in CLASSES}
    existing: dict[str, list[Path]] = {name: [] for name in CLASSES}
    known_files: set[str] = set()
    for name in CLASSES:
        for path in sorted((destination / "images" / name.replace(" ", "_")).glob("*.jpg")):
            existing[name].append(path)
            known_files.add(path.stem)

    counts = {name: len(existing[name]) for name in CLASSES}
    deficits = {name: max(0, PER_CLASS - counts[name]) for name in CLASSES}
    print("Indexing downloadable train image IDs...", flush=True)
    with train_metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        available_ids = {row["ImageID"] for row in csv.DictReader(metadata_file)}
    print(f"Indexed {len(available_ids):,} train IDs; remaining quotas: {deficits}", flush=True)

    candidates, fallback_candidates = (
        _collect_candidates(TRAIN_LABEL_URL, class_ids, available_ids, known_files, deficits)
        if any(deficits.values())
        else ({name: set() for name in CLASSES}, {name: set() for name in CLASSES})
    )
    selected_ids: dict[str, str] = {}
    for name in SELECTION_ORDER:
        for image_id in sorted(candidates[name]):
            if deficits[name] > 0 and image_id not in selected_ids:
                selected_ids[image_id] = name
                deficits[name] -= 1

    fallback_to_classes: dict[str, list[str]] = {}
    for name, image_ids in fallback_candidates.items():
        for image_id in image_ids:
            if image_id not in selected_ids:
                fallback_to_classes.setdefault(image_id, []).append(name)

    requested_ids = known_files | set(selected_ids)
    metadata_by_id: dict[str, dict] = {}
    print(f"Resolving attribution for {len(requested_ids):,} images...", flush=True)
    with train_metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        for row in csv.DictReader(metadata_file):
            if row["ImageID"] in requested_ids:
                metadata_by_id[row["ImageID"]] = _metadata_dict(row, TRAIN_METADATA_URL)
    with validation_metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        for row in csv.DictReader(metadata_file):
            if row["ImageID"] in known_files:
                metadata_by_id[row["ImageID"]] = _metadata_dict(row, VALIDATION_METADATA_URL)

    if any(deficits.values()):
        print(f"Resolving remaining IDs from Open Images image-level metadata: {deficits}", flush=True)
        with requests.get(ALL_TRAIN_METADATA_URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            lines = (line.decode("utf-8") for line in response.iter_lines() if line)
            rows_seen = 0
            for rows_seen, row in enumerate(csv.DictReader(lines), start=1):
                image_id = row["ImageID"]
                possible_classes = fallback_to_classes.get(image_id, [])
                thumbnail_url = row.get("Thumbnail300KURL", "")
                for name in SELECTION_ORDER:
                    if (
                        name in possible_classes
                        and deficits[name] > 0
                        and image_id not in selected_ids
                        and thumbnail_url
                    ):
                        selected_ids[image_id] = name
                        metadata_by_id[image_id] = _metadata_dict(row, ALL_TRAIN_METADATA_URL)
                        metadata_by_id[image_id]["PixelURL"] = thumbnail_url
                        deficits[name] -= 1
                        break
                if rows_seen % 500_000 == 0:
                    print(f"Matched {len(selected_ids):,} images after {rows_seen:,} metadata rows: {deficits}", flush=True)
                if not any(deficits.values()):
                    break

    if any(deficits.values()):
        raise RuntimeError(f"Available labeled Open Images candidates fall short: {deficits}")
    selected = {image_id: (name, metadata_by_id[image_id]) for image_id, name in selected_ids.items()}
    already_attributed = metadata_by_id

    records = []
    failures = []
    jobs = list(selected.items())

    def fetch(job: tuple[str, tuple[str, dict]]) -> dict:
        image_id, (name, metadata) = job
        relative = Path("images") / name.replace(" ", "_") / f"{image_id}.jpg"
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size == 0:
            image_url = metadata.get("PixelURL") or S3_IMAGE_URL.format(
                split=metadata["Subset"], image_id=image_id
            )
            metadata["PixelURL"] = image_url
            _download_image(image_url, path)
        return _record(source["id"], name, image_id, relative, metadata, path)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except Exception as error:  # noqa: BLE001 - keep per-image failure details.
                image_id, (name, _) = futures[future]
                failures.append({"class": name, "image_id": image_id, "error": str(error)})
            if completed % 25 == 0 or completed == len(futures):
                print(f"Downloaded {completed}/{len(futures)} newly selected images.", flush=True)

    for name in CLASSES:
        for path in existing[name]:
            metadata = already_attributed.get(path.stem, {})
            records.append(_record(source["id"], name, path.stem, path.relative_to(destination), metadata, path))
    records.sort(key=lambda row: (row["negative_class"], row["open_images_id"]))

    manifest = data_root() / "manifests" / "open_images_negatives.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    final_counts = {name: sum(record["negative_class"] == name for record in records) for name in CLASSES}
    result = {
        "source_id": source["id"],
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "status": "downloaded" if not failures and all(value == PER_CLASS for value in final_counts.values()) else "partial",
        "requested": PER_CLASS * len(CLASSES),
        "downloaded": len(records),
        "per_class_counts": final_counts,
        "failures": failures,
        "classes": CLASSES,
        "per_class": PER_CLASS,
        "license": source["license"],
        "metadata_sources": [
            str(validation_metadata_path.relative_to(destination)),
            str(train_metadata_path.relative_to(destination)),
            ALL_TRAIN_METADATA_URL,
        ],
    }
    write_json(data_root() / "manifests" / "download_open_images_v7_negatives.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sources = {source["id"]: source for source in load_sources()}
    result = download_open_images(sources["open_images_v7_negatives"])
    print(f"{result['status']}: {result['downloaded']}/{result['requested']} images")
    if result["status"] != "downloaded":
        raise SystemExit(1)
