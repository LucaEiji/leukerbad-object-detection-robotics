# Leukerbad Object Detection Robotics

Base project for combining computer-vision object detection with a simple robotic arm.

## Project layout

```text
src/leukerbad/
  detection/    Detection interfaces and model adapters
  robotics/     Robotic-arm interfaces and motion commands
  cli.py        Minimal command-line entry point
tests/          Tests
```

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --extra dev
uv run leukerbad-detect
```

Optional integrations can be installed with:

```bash
uv sync --extra ml --extra robotics --extra dev
```

The current implementation provides interfaces and safe placeholders. Camera, model, and
robot-specific wiring can be added without changing the project layout.

## Development

```bash
uv run ruff check .
uv run pytest
```

## VapeGuard Data Pipeline

The initial data pipeline covers source registration, raw downloads, source reporting, and
image/annotation inventory. It does not train a model, create masks, curate labels, or control
the robot.

Install the data tools:

```bash
uv sync --extra data --extra dev
```

Configure credentials locally. Do not commit either credential or raw data:

```bash
export ROBOFLOW_API_KEY="your-key"
# Kaggle uses ~/.kaggle/kaggle.json after competition rules are accepted.
```

Generate the source report:

```bash
uv run vapeguard-inspect
```

Download a specific Roboflow version only after verifying its license and metadata:

```bash
uv run vapeguard-download \
  --source-id roboflow_primary \
  --roboflow-version 7 \
  --format yolov8
```

Download the Kaggle competition after accepting its rules:

```bash
uv run vapeguard-download \
  --source-id kaggle_ecigarettes_vs_batteries
```

Inventory all downloaded sources:

```bash
uv run vapeguard-inventory
```

Continue the local audit pipeline:

```bash
uv run vapeguard-curate --source-id roboflow_primary
uv run vapeguard-audit --source-id roboflow_primary
uv run vapeguard-duplicates
uv run vapeguard-gallery
```

Apply the documented manual cleanup rules:

```bash
uv run vapeguard-filter
```

This removes the requested raw image/label pairs and records every removal in
`public_data/manifests/exclusions.jsonl`. The primary vape threshold is inclusive and based on
normalized annotation area, with the exact reference images and threshold recorded in
`public_data/reports/manual_exclusions.json`.

Outputs are written under `public_data/manifests/` and `public_data/reports/`. Raw,
curated, annotation, and export contents are intentionally ignored by Git.

Data lineage and cleanup rules are documented in
`public_data/reports/transformations.md`. The current raw directories reflect the requested
manual cleanup; the exclusion ledger is at `public_data/manifests/exclusions.jsonl`.

## Standardized single-class YOLO data

Build a fresh, unsplit detection export with:

```bash
uv run --extra data vapeguard-standardize
```

The default output is `public_data/exports/vape_yolo/`. Use `--output` to select a fresh
directory for subsequent builds; existing exports are never overwritten. The export retains
only `Vape`, `vape`, and `ecigarette` annotations, all mapped to class `0: vape`.
Every negative image is retained with an empty label file. Retained source polygons become
enclosing detection boxes, and Kaggle COCO boxes are converted to normalized YOLO boxes.
Original source files remain unchanged.

The export also reproduces the historical raw cleanup directly from code, even on a fresh
download with no local exclusion ledger. The rules shared with `vapeguard-filter` exclude
primary-source cigarette-only images, primary images containing a vape whose normalized
annotation area is at or below `0.0030029296875`, and the three explicitly marked additional-source
images. Area uses width times height for boxes and shoelace area for polygons, before conversion
to enclosing boxes. The cutoff is a manual dataset-quality decision: the user judged three
reference pictures unsuitable, and the largest vape box area among those pictures was chosen
as the inclusive threshold. It is not a statistically fitted or model-derived quality threshold.
These source-specific image exclusions precede label filtering; all other
images without retained target boxes remain intentional negatives. The two Kaggle train
exclusions remain active. Export summaries record the policy and exclusions applied to the
input files present on that run; previously deleted raw files are not counted as new exclusions.

Only the manually labeled Kaggle test images enter the export. The two previously flagged
Kaggle train images and the exact duplicate manual photo in the Kaggle test export are excluded.
Final train/validation/test assignments are deferred; original split metadata and origin groups
remain in `manifest.jsonl`. `classes.yaml` defines the class mapping, not training splits.

Exact counts, conversion details, and exclusions are documented in
`public_data/reports/standardization.md` and `standardization_summary.json`.
Older inventory/audit reports describe their original source snapshots, not this unified export.

## License

MIT
