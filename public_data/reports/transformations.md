# VapeGuard Data Transformations

This document is the lineage record from the downloaded public sources to the current
VapeGuard manifests. Raw files are not transformed in place when the normal pipeline runs.

## Important Raw-Directory Exception

The user-requested cleanup was applied directly to the local raw directories before this
ledger was added. The removed image/label pairs are therefore no longer present locally. Every
known removal is recorded in `public_data/manifests/exclusions.jsonl`, including its original
relative path and reason. The current raw directories are filtered working copies, not pristine
archives of the original downloads.

## Source Inventories

| Source | Original image count | Current image count | Original instances | Current instances |
| --- | ---: | ---: | ---: | ---: |
| Roboflow primary v10 | 5,774 | 2,585 | 7,470 | 3,907 |
| Roboflow additional v14 | 2,209 | 2,206 | 4,342 | 4,336 |
| Kaggle competition | 323 | 323 | 631 | 631 |

Original counts are recorded from the inventory generated before the requested cleanup. Current
counts come from the latest inventory and annotation audit.

## Image Exclusions

### Primary Roboflow dataset

- Removed every image whose annotations contained only `cigarette`.
- Removed every image containing a `vape` annotation whose normalized area was less than or equal to `0.0030029296875`.
- For a YOLO box, normalized area is `width * height`.
- For a YOLO polygon, normalized area is its shoelace area, before conversion to an enclosing box.
- The threshold is the largest normalized box area among the three user-provided reference images.
- This value reflects a manual dataset-quality decision: the user considered those reference
  pictures unsuitable and selected them as examples to exclude. The cutoff reproduces that
  judgment; it is not a statistically fitted or model-derived quality threshold.
- The threshold was inclusive so all three reference images were excluded.
- The image and matching label file were removed together.
- Removed: 3,054 cigarette-only pairs and 135 small-vape pairs.

### Additional Roboflow dataset

- Removed the three explicitly user-marked misannotated image/label pairs.
- No other additional-dataset images were removed by the manual filter.

### Kaggle dataset

- No raw images or source annotations were removed.
- The source ZIP was retained and extracted under `public_data/raw/kaggle/extracted/`.

## Class Transformations

| Source class | Curated decision |
| --- | --- |
| `vape`, `Vape`, `ecigarette`, `e-cigarette` | `vape` positive |
| `cigarette` | hard negative when retained; primary cigarette-only images were excluded |
| `nonVape`, `notVape` | hard negative |
| `possiblyVape` | manual review |
| `Battery`, `battery`, `Cartridge` | manual review; never mapped to `vape` |

Kaggle `ecigarette` annotations are mapped to the VapeGuard positive class `vape` in the
curated manifest. The original COCO category remains unchanged in the downloaded source.

## Annotation Transformations

- Roboflow YOLO files remain unchanged for retained images.
- Kaggle `train_annotations.json` remains unchanged in the raw extraction.
- YOLO and COCO annotations are parsed into common JSONL manifests for inventory, curation, and audit.
- Source class names and raw relative paths remain in every curated record.
- No synthetic segmentation masks are generated.
- Retained box-only records are marked `mask_pending_manual_annotation`.

## Derived Outputs

- `public_data/manifests/*jsonl`: inventory and curated provenance records
- `public_data/manifests/exclusions.jsonl`: manual exclusion ledger
- `public_data/reports/manual_exclusions.json`: threshold and removal summary
- `public_data/reports/class_mapping.md`: class decisions
- `public_data/reports/annotation_audit.md`: annotation validation
- `public_data/reports/duplicates.md`: duplicate-review summary
- `public_data/reports/review_status.md`: review-gallery counts

Duplicate candidates were flagged only; no duplicate-based deletion was performed.

## Reproducible standardized-export cleanup

`vapeguard-standardize` applies the historical primary and additional-source image exclusions
using the same `manual_exclusion_reason` code as `vapeguard-filter`. It works on fresh downloads
without `manifests/exclusions.jsonl`; filenames, reference threshold, and class-based rules are
part of the code. It also retains the two explicit Kaggle train image exclusions implemented
in `standardize.py`. Standardization excludes images from the export without deleting raw files.

The historical 3,192 raw removals (3,054 cigarette-only, 135 small-vape, three marked additional)
are a prior-run count. New export summaries report only exclusions found in their current input,
alongside the reproducible rule definitions. The historical primary cigarette-only removal is
a source-specific exception to negative retention. Other non-target-only and empty-label images
are retained as negatives; non-target annotation rows are discarded rather than reviewed.
