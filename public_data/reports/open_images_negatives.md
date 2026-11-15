# Open Images Confuser Subset

## Status

The requested target is 600 images: 100 per class. The local dataset currently has 371 readable
image files (snapshot: 2026-11-15). It is **partial**, not a completed 600-image download.

| Class | Current | Target | Remaining |
| --- | ---: | ---: | ---: |
| Eraser | 41 | 100 | 59 |
| Pen | 78 | 100 | 22 |
| Perfume | 70 | 100 | 30 |
| Spice rack | 60 | 100 | 40 |
| Tin can | 80 | 100 | 20 |
| Torch | 42 | 100 | 58 |
| **Total** | **371** | **600** | **229** |

Current files span both Open Images validation and train splits: 103 validation images and 268
train images. The class values above are the local folder assignments and are not training labels.

## Selection and Download

The selector uses the Open Images V7 boxable class descriptions to resolve class IDs and the Open
Images V6 human-verified train image-level labels (`Confidence=1`) to find candidate images. It
intersects those IDs with the Open Images boxable train image index before trying the public image
bucket. IDs outside that downloadable subset can be resolved through the broader train image
metadata and its `Thumbnail300KURL`.

Official source files and documentation:

- [Open Images downloads and data format](https://storage.googleapis.com/openimages/web/download.html)
- [V7 boxable class descriptions](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)
- [V6 human-verified train image labels](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv)
- [Train images with boxable metadata](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv)
- [All train image metadata](https://storage.googleapis.com/openimages/2018_04/train/train-images-with-labels-with-rotation.csv)

The large upstream label and metadata files are streamed, with progress printed during scans.
This scan can take a substantial time. Images already present under the class folders are kept and
count toward their class quota on a subsequent run:

```bash
uv sync --extra data --extra dev
uv run vapeguard-openimages
```

Images are stored under `public_data/raw/open_images_v7_negatives/images/<class>/`. Selection and
download summaries go to `public_data/manifests/download_open_images_v7_negatives.json`; per-image
records go to `public_data/manifests/open_images_negatives.jsonl`.

## Provenance and Interpretation

Each completed record should preserve the Open Images ID, assigned confuser class, local path,
SHA-256, source split, pixel URL, original URL and landing page, thumbnail URL, author, title,
license URL, and metadata source. The original source attribution is image-specific; keep those
fields with the image when copying or redistributing it.

The selected Open Images class is a **positive class label**, not proof that the image contains no
vape. These images are intended as potential confuser/background examples and require review
before being treated as confirmed vape-free negatives. They have no VapeGuard boxes; the
standardized YOLO export gives them empty label files.

The current partial manifest predates full metadata resolution: some existing records still have
unknown author and original-URL fields. Do not treat those fields as verified attribution. A
successful metadata-enrichment run should replace them with the corresponding source rows.
