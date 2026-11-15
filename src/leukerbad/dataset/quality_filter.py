"""Dataset image quality filter and shape-balance utilities for YOLO-seg training splits.

Detects corrupted offline augmentations (pure-white overexposures, pitch-black frames,
and severe motion blur) while strictly refusing to modify any `test` split.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
_ROBOFLOW_HASH_RE = re.compile(r"\.rf\.[0-9a-zA-Z]+$")


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Photometric and sharpness metrics for a single dataset image."""

    mean_luma: float
    p5_luma: float
    p95_luma: float
    dynamic_range: float
    laplacian_var: float
    is_whiteout: bool
    is_blackout: bool
    is_blurry: bool
    quality_score: float
    is_usable: bool


@dataclass(frozen=True)
class QualityFilterReport:
    """Summary of quality-based filtering on a non-test dataset split."""

    total_checked: int
    kept_count: int
    removed_count: int
    rescued_group_best_count: int
    removed_whiteout: int
    removed_blackout: int
    removed_blurry: int
    removed_files: tuple[str, ...]


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2D or 3D BGR/BGRA image array into a 2D `uint8` grayscale image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray = image[:, :, 0]
        else:
            raise ValueError(f"Unsupported channel count {channels} for quality check.")
    else:
        raise ValueError(f"Expected 2D or 3D image array, got ndim={image.ndim}.")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def assess_image_quality(
    image: np.ndarray,
    *,
    max_mean_luma: float = 218.0,
    min_mean_luma: float = 38.0,
    min_dynamic_range: float = 48.0,
    min_laplacian_var: float = 15.0,
) -> ImageQualityMetrics:
    """Assess exposure, dynamic range, and focus sharpness of a BGR or grayscale image."""
    gray = _to_grayscale_uint8(image)

    mean_luma = float(np.mean(gray))
    p5_raw, p95_raw = np.percentile(gray, [5.0, 95.0])
    p5_luma = float(p5_raw)
    p95_luma = float(p95_raw)
    dynamic_range = float(p95_luma - p5_luma)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_whiteout = bool(
        mean_luma > max_mean_luma or (p5_luma > 180.0 and dynamic_range < min_dynamic_range)
    )
    is_blackout = bool(
        mean_luma < min_mean_luma or (p95_luma < 65.0 and dynamic_range < min_dynamic_range)
    )
    is_blurry = bool(laplacian_var < min_laplacian_var)
    is_usable = not (is_whiteout or is_blackout or is_blurry)

    exposure_score = max(0.0, min(1.0, 1.0 - abs(mean_luma - 128.0) / 128.0))
    contrast_score = max(0.0, min(1.0, dynamic_range / 140.0))
    sharpness_score = max(0.0, min(1.0, math.log1p(max(0.0, laplacian_var)) / 6.0))
    raw_score = 0.35 * exposure_score + 0.30 * contrast_score + 0.35 * sharpness_score
    quality_score = float(max(0.0, min(1.0, raw_score if is_usable else raw_score * 0.35)))

    return ImageQualityMetrics(
        mean_luma=mean_luma,
        p5_luma=p5_luma,
        p95_luma=p95_luma,
        dynamic_range=dynamic_range,
        laplacian_var=laplacian_var,
        is_whiteout=is_whiteout,
        is_blackout=is_blackout,
        is_blurry=is_blurry,
        quality_score=quality_score,
        is_usable=is_usable,
    )


def _is_test_split_dir(path: Path) -> bool:
    name = path.name.lower()
    return name == "test" or (
        name in ("images", "labels") and path.parent.name.lower() == "test"
    )


def _fallback_origin_group(stem: str) -> str:
    """Derive a grouping key from an image stem when manifest provenance is unavailable."""
    stripped = _ROBOFLOW_HASH_RE.sub("", stem)
    return stripped or stem


def _build_origin_group_lookup(
    manifest_entries: list[dict[str, Any]] | None,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not manifest_entries:
        return lookup

    for entry in manifest_entries:
        if not isinstance(entry, dict):
            continue
        ogroup = entry.get("origin_group")
        if not isinstance(ogroup, str) or not ogroup.strip():
            continue
        group_id = ogroup.strip()
        for field_name in ("image", "file_name", "filename", "image_path", "export_image"):
            val = entry.get(field_name)
            if isinstance(val, str) and val:
                p = Path(val)
                lookup[val] = group_id
                lookup[p.name] = group_id
                lookup[p.stem] = group_id
    return lookup


def filter_dataset_split_by_quality(
    images_dir: Path,
    labels_dir: Path,
    manifest_entries: list[dict[str, Any]] | None = None,
    *,
    keep_best_per_group: bool = True,
    dry_run: bool = False,
    **quality_kwargs: Any,
) -> QualityFilterReport:
    """Filter unusable images from a `train` or `val` split while rescuing all-bad groups."""
    img_root = Path(images_dir)
    lbl_root = Path(labels_dir)
    if _is_test_split_dir(img_root) or _is_test_split_dir(lbl_root):
        raise ValueError(
            f"Refusing to filter test split directory ({img_root}, {lbl_root}); "
            "held-out test splits must remain untouched."
        )

    if not img_root.is_dir():
        return QualityFilterReport(
            total_checked=0,
            kept_count=0,
            removed_count=0,
            rescued_group_best_count=0,
            removed_whiteout=0,
            removed_blackout=0,
            removed_blurry=0,
            removed_files=(),
        )

    group_lookup = _build_origin_group_lookup(manifest_entries)
    image_paths = sorted(
        p for p in img_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    assessed: list[tuple[Path, str, ImageQualityMetrics]] = []
    grouped_indices: dict[str, list[int]] = {}

    for img_path in image_paths:
        ogroup = (
            group_lookup.get(img_path.name)
            or group_lookup.get(img_path.stem)
            or _fallback_origin_group(img_path.stem)
        )
        decoded = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if decoded is None:
            metrics = ImageQualityMetrics(
                mean_luma=0.0,
                p5_luma=0.0,
                p95_luma=0.0,
                dynamic_range=0.0,
                laplacian_var=0.0,
                is_whiteout=False,
                is_blackout=True,
                is_blurry=True,
                quality_score=0.0,
                is_usable=False,
            )
        else:
            metrics = assess_image_quality(decoded, **quality_kwargs)

        idx = len(assessed)
        assessed.append((img_path, ogroup, metrics))
        grouped_indices.setdefault(ogroup, []).append(idx)

    keep_flags = [item[2].is_usable for item in assessed]
    rescued_count = 0

    if keep_best_per_group:
        for _ogroup, indices in sorted(grouped_indices.items()):
            if not any(keep_flags[i] for i in indices):
                best_idx = max(
                    indices,
                    key=lambda i: (assessed[i][2].quality_score, -i),
                )
                keep_flags[best_idx] = True
                rescued_count += 1

    removed_whiteout = 0
    removed_blackout = 0
    removed_blurry = 0
    removed_names: list[str] = []

    for idx, (img_path, _ogroup, metrics) in enumerate(assessed):
        if keep_flags[idx]:
            continue
        removed_names.append(img_path.name)
        if metrics.is_whiteout:
            removed_whiteout += 1
        if metrics.is_blackout:
            removed_blackout += 1
        if metrics.is_blurry:
            removed_blurry += 1

        if not dry_run:
            img_path.unlink(missing_ok=True)
            lbl_path = lbl_root / f"{img_path.stem}.txt"
            lbl_path.unlink(missing_ok=True)

    total_checked = len(assessed)
    removed_count = len(removed_names)
    kept_count = total_checked - removed_count
    return QualityFilterReport(
        total_checked=total_checked,
        kept_count=kept_count,
        removed_count=removed_count,
        rescued_group_best_count=rescued_count,
        removed_whiteout=removed_whiteout,
        removed_blackout=removed_blackout,
        removed_blurry=removed_blurry,
        removed_files=tuple(removed_names),
    )


def compute_polygon_elongation(coords: list[float] | np.ndarray) -> float:
    """Compute oriented aspect ratio `major / minor` for a YOLO polygon or 4-value bbox."""
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 4:
            cx, cy, w, h = (float(v) for v in arr)
            hw, hh = abs(w) * 0.5, abs(h) * 0.5
            pts = np.array(
                [
                    [cx - hw, cy - hh],
                    [cx + hw, cy - hh],
                    [cx + hw, cy + hh],
                    [cx - hw, cy + hh],
                ],
                dtype=np.float32,
            )
        elif arr.size >= 6 and arr.size % 2 == 0:
            pts = arr.reshape(-1, 2)
        else:
            raise ValueError(
                f"Expected 4 bbox coords or even >=6 polygon coords, got {arr.size}."
            )
    else:
        pts = arr.reshape(-1, 2)
        if pts.shape[0] < 3:
            raise ValueError(f"Polygon requires at least 3 vertices, got {pts.shape[0]}.")

    _, (side1, side2), _ = cv2.minAreaRect(pts)
    major = max(float(side1), float(side2))
    if major < 1e-6:
        return 1.0
    minor = max(1e-6, min(float(side1), float(side2)))
    return major / minor


def find_underrepresented_shape_stems(
    labels_dir: Path,
    *,
    max_elongation: float = 2.2,
    min_instances_for_clutter: int = 2,
) -> list[str]:
    """Return sorted stems of positive labels with compact/upright shapes or multi-vape clutter."""
    lbl_root = Path(labels_dir)
    if not lbl_root.is_dir():
        return []

    matched_stems: list[str] = []
    for lbl_path in sorted(p for p in lbl_root.glob("*.txt") if p.is_file()):
        elongations: list[float] = []
        for raw_line in lbl_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 5:
                continue
            try:
                int(tokens[0])
                coords = [float(tok) for tok in tokens[1:]]
            except ValueError:
                continue
            if len(coords) != 4 and (len(coords) < 6 or len(coords) % 2 != 0):
                continue
            elongations.append(compute_polygon_elongation(coords))

        if not elongations:
            continue
        has_compact = any(el <= max_elongation for el in elongations)
        has_clutter = len(elongations) >= min_instances_for_clutter
        if has_compact or has_clutter:
            matched_stems.append(lbl_path.stem)

    return matched_stems
