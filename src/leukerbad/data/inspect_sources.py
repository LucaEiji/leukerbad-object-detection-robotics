"""Create a source and access report from the checked-in dataset registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import data_root, load_sources, utc_now


def render_report(sources: list[dict], inspected_at: str) -> str:
    lines = [
        "# VapeGuard Dataset Sources",
        "",
        f"Last generated: `{inspected_at}`",
        "",
        (
            "Metadata is recorded only when verified from the source or its export. `unknown` "
            "means it has not been verified and is not an estimate."
        ),
        "",
        "## Source Summary",
        "",
        "| Source | URL | Images | Instances | Classes | Format | License | Access status |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    audit_path = data_root() / "reports" / "annotation_audit.json"
    audit_by_source = {}
    if audit_path.exists():
        audit_by_source = {
            item["source_id"]: item for item in json.loads(audit_path.read_text(encoding="utf-8"))["sources"]
        }

    downloads = {}
    for source in sources:
        path = data_root() / "manifests" / f"download_{source['id']}.json"
        downloads[source["id"]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for source in sources:
        classes = source.get("original_classes", "unknown")
        if isinstance(classes, list):
            classes = ", ".join(classes)
        status = "requires credentials/terms verification"
        if source["kind"] == "roboflow" and source["id"] == "roboflow_primary":
            status = "public page; API key and exact version required for export"
        elif source["kind"] == "roboflow":
            status = "metadata verified; reuse permission unknown"
        download = downloads[source["id"]]
        if download.get("status") == "downloaded":
            status = "downloaded; terms recorded"
        elif source["kind"] == "open_images" and download:
            status = f"{download.get('status', 'unknown')}: {download.get('downloaded', 0)}/{download.get('requested', source.get('target_images', '?'))} images"
        version = source.get("version") or download.get("version") or "unknown"
        instances = (
            "n/a (image labels used for selection)"
            if source["kind"] == "open_images"
            else audit_by_source.get(source["id"], {}).get("instances", "unknown")
        )
        image_count = source.get("source_images", "unknown")
        if source["kind"] == "open_images":
            image_count = (
                f"{download.get('downloaded', 0)}/{download.get('requested', source.get('target_images', '?'))} target"
                if download
                else f"0/{source.get('target_images', '?')} target"
            )
        lines.append(
            f"| `{source['id']}` | [{source['url']}]({source['url']}) | "
            f"{image_count} | {instances} | {classes} | "
            f"{source.get('annotation_format', 'unknown')} | {source.get('license', 'unknown')} | {status} |"
        )

    lines.extend(["", "## Detailed Records", ""])
    for source in sources:
        classes = source.get("original_classes", "unknown")
        if isinstance(classes, list):
            classes = ", ".join(classes)
        download = downloads[source["id"]]
        version = source.get("version") or download.get("version") or "unknown"
        lines.extend(
            [
                f"### `{source['id']}`",
                "",
                f"- Source URL: {source['url']}",
                f"- Raw destination: `{source['raw_dir']}`",
                f"- Version: {version}",
                f"- Download date: {download.get('finished_at', 'unknown')}",
                f"- License or terms: {source.get('license', 'unknown')}",
                f"- Original classes: {classes}",
                f"- Source-image count: {source.get('source_images', 'unknown')}",
                f"- Annotation format: {source.get('annotation_format', 'unknown')}",
                f"- Notes: {source.get('notes', 'unknown')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Reuse Constraints",
            "",
            (
                "- Roboflow's primary Universe page lists CC BY 4.0. Attribution and a link to "
                "the license must be retained, and underlying image rights should not be assumed "
                "beyond the stated dataset license."
            ),
            (
                "- Kaggle competition data is restricted by the accepted competition terms. Raw "
                "files must remain private and must not be committed to this repository or redistributed."
            ),
            (
                "- The additional Roboflow dataset is listed as CC BY 4.0 and requires attribution "
                "and a link to the license."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=data_root() / "reports" / "dataset_sources.md")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(load_sources(), utc_now()), encoding="utf-8")
    print(args.output)
