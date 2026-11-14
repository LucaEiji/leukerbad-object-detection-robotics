"""Apply conservative class decisions to source annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .coco import index_coco, load_coco
from .common import data_root, load_sources, utc_now

POSITIVE = {"vape", "ecigarette", "e-cigarette", "e_cigarette"}
NEGATIVE = {"cigarette", "nonvape", "non-vape", "notvape", "not-vape"}
AMBIGUOUS = {"possiblyvape", "possibly-vape", "cartridge", "battery"}


def decision_for_class(name: str) -> tuple[str, str]:
    normalized = name.strip().lower().replace(" ", "")
    if normalized in POSITIVE:
        return "positive", "Confirmed vape synonym mapped to vape."
    if normalized in NEGATIVE:
        return "hard_negative", "Explicit non-vape or traditional-cigarette class kept separate."
    if normalized in AMBIGUOUS:
        return "manual_review", "Potential vape, component, or ambiguous class not promoted to vape."
    return "manual_review", "Unknown source class requires manual review."


def _read_classes(raw_dir: Path) -> list[str]:
    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (ImportError, OSError, ValueError):
        return []
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return [str(name) for name in names]


def _label_path(raw_dir: Path, relative_image: str) -> Path | None:
    image_path = raw_dir / relative_image
    split_dir = image_path.parent.parent
    candidate = split_dir / "labels" / f"{image_path.stem}.txt"
    return candidate if candidate.exists() else None


def curate_source(source: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    raw_dir = data_root().parent / source["raw_dir"]
    classes = _read_classes(raw_dir)
    coco_index = {}
    coco_result = load_coco(raw_dir)
    if coco_result:
        coco_index = index_coco(coco_result[0])
    inventory_path = data_root() / "manifests" / f"{source['id']}.jsonl"
    records: list[dict] = []
    counts: dict[str, int] = {}
    if not inventory_path.exists():
        return records, {"source_id": source["id"], "status": "inventory_missing"}

    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        image = json.loads(line)
        label_path = _label_path(raw_dir, image["relative_path"])
        class_names = []
        if label_path:
            for row in label_path.read_text(encoding="utf-8").splitlines():
                fields = row.split()
                if fields and fields[0].isdigit():
                    class_id = int(fields[0])
                    class_names.append(classes[class_id] if class_id < len(classes) else "unknown")
        elif Path(image["relative_path"]).name in coco_index:
            class_names = [item["category_name"] for item in coco_index[Path(image["relative_path"]).name]]
        decisions = [decision_for_class(name) for name in class_names]
        labels = {decision[0] for decision in decisions}
        if "positive" in labels:
            decision = "positive"
        elif "manual_review" in labels or not class_names:
            decision = "manual_review" if class_names else "unknown_empty_annotation"
        else:
            decision = "hard_negative"
        counts[decision] = counts.get(decision, 0) + 1
        records.append(
            {
                **image,
                "source_classes": sorted(set(class_names)),
                "curation_decision": decision,
                "mask_status": "pending_manual_annotation",
                "provenance": {
                    "source_id": source["id"],
                    "source_url": source["url"],
                    "raw_relative_path": image["relative_path"],
                },
            }
        )
    return records, {"source_id": source["id"], "class_names": classes, "decisions": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", help="Curate only one source")
    args = parser.parse_args()
    sources = load_sources()
    if args.source_id:
        sources = [source for source in sources if source["id"] == args.source_id]
        if not sources:
            parser.error(f"Unknown source ID: {args.source_id}")

    manifest_dir = data_root() / "manifests"
    summaries = []
    for source in sources:
        records, summary = curate_source(source)
        output = manifest_dir / f"curated_{source['id']}.jsonl"
        with output.open("w", encoding="utf-8") as file_handle:
            for record in records:
                file_handle.write(json.dumps(record, sort_keys=True) + "\n")
        summaries.append(summary)
    report_dir = data_root() / "reports"
    (report_dir / "curation_summary.json").write_text(
        json.dumps({"generated_at": utc_now(), "sources": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "class_mapping.md").write_text(
        "# Class Mapping\n\n"
        "| Source class | Decision | Rationale |\n"
        "| --- | --- | --- |\n"
        "| `vape`, `Vape`, `ecigarette`, `e-cigarette` | `vape` positive | Confirmed vape synonym. |\n"
        "| `cigarette` | hard negative | Traditional cigarette is not a vape. |\n"
        "| `nonVape`, `notVape` | hard negative | Explicit negative class remains negative. |\n"
        "| `possiblyVape` | manual review | Ambiguous class is not promoted automatically. |\n"
        "| `Cartridge`, `Battery` | manual review | Component is not assumed to be a complete device. |\n"
        "| unknown classes | manual review | No unverified class is promoted to positive. |\n\n"
        "All curated records retain source class names and raw relative paths. Box-only records "
        "are marked `mask_pending_manual_annotation`; no masks are synthesized.\n",
        encoding="utf-8",
    )
