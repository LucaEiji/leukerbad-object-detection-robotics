"""Shared paths and serialization helpers for the data pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root based on this package's location."""
    current_directory = Path.cwd()
    if (current_directory / "public_data" / "dataset_sources.json").exists():
        return current_directory
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return project_root() / "public_data"


def load_sources(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the declarative dataset source configuration."""
    source_path = path or data_root() / "dataset_sources.json"
    with source_path.open(encoding="utf-8") as source_file:
        return json.load(source_file)["sources"]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
