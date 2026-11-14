"""Download configured public datasets without altering their raw contents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZipFile

from .common import data_root, load_sources, project_root, utc_now, write_json


def _download_roboflow(source: dict, destination: Path, version: int, export_format: str) -> None:
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RuntimeError("Install the data dependencies with `uv sync --extra data`.") from error

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")

    client = Roboflow(api_key=api_key)
    project = client.workspace(source["workspace"]).project(source["project"])
    has_images = any(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for path in destination.rglob("*"))
    project.version(version).download(
        model_format=export_format,
        location=str(destination),
        overwrite=not has_images,
    )


def _download_kaggle(source: dict, destination: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError("Install the data dependencies with `uv sync --extra data`.") from error

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:  # Kaggle exposes several auth exception types.
        raise RuntimeError(
            "Kaggle authentication failed. Configure ~/.kaggle/kaggle.json and accept the "
            "competition rules."
        ) from error

    destination.mkdir(parents=True, exist_ok=True)
    api.competition_download_files(
        source["competition"],
        path=str(destination),
        quiet=False,
        force=False,
    )
    archives = sorted(destination.glob("*.zip"))
    if not archives:
        raise RuntimeError("Kaggle download completed but no ZIP archive was found.")
    extraction_dir = destination / "extracted"
    extraction_dir.mkdir(exist_ok=True)
    with ZipFile(archives[-1]) as archive:
        for member in archive.infolist():
            target = (extraction_dir / member.filename).resolve()
            if extraction_dir.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe path in Kaggle archive: {member.filename}")
        archive.extractall(extraction_dir)


def download_source(
    source: dict,
    *,
    version: int | None,
    export_format: str,
) -> dict:
    destination = data_root().parent / source["raw_dir"]
    record = {
        "source_id": source["id"],
        "started_at": utc_now(),
        "raw_dir": source["raw_dir"],
        "status": "failed",
    }
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if source["kind"] == "roboflow":
            if source.get("license") in {None, "unknown"}:
                raise RuntimeError(
                    "Dataset license or reuse permission is unknown; verify it before downloading."
                )
            if version is None:
                raise RuntimeError(
                    f"An explicit Roboflow version is required for {source['id']}."
                )
            _download_roboflow(source, destination, version, export_format)
            record["version"] = version
            record["format"] = export_format
        elif source["kind"] == "kaggle":
            _download_kaggle(source, destination)
        else:
            raise RuntimeError(f"Unsupported source kind: {source['kind']}")
        record["status"] = "downloaded"
    except Exception as error:  # noqa: BLE001 - preserve provider-specific download errors.
        record["error"] = str(error)
    record["finished_at"] = utc_now()
    return record


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("Install the data dependencies with `uv sync --extra data`.") from error
    load_dotenv(project_root() / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="Configured source ID to download")
    parser.add_argument("--roboflow-version", type=int, help="Exact Roboflow version number")
    parser.add_argument("--format", default="yolov8", help="Roboflow export format")
    args = parser.parse_args()

    sources = {source["id"]: source for source in load_sources()}
    if args.source_id not in sources:
        parser.error(f"Unknown source ID: {args.source_id}")

    result = download_source(
        sources[args.source_id],
        version=args.roboflow_version or sources[args.source_id].get("version"),
        export_format=args.format,
    )
    manifest_path = data_root() / "manifests" / f"download_{args.source_id}.json"
    write_json(manifest_path, result)
    print(f"{result['status']}: {args.source_id}")
    if result["status"] == "failed":
        print(f"error: {result['error']}")
        raise SystemExit(1)
