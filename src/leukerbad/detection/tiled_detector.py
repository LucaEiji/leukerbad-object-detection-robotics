"""Multi-scale overlapping tiled wrapper + polygon-union NMS for overhead YOLO-seg inference.

Magnifies small tabletop objects (`1920x1080` -> overlapping tiles) before YOLO `640`
downscaling, then merges tile-seam fragments and full-frame duplicates via local ROI mask union.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector
from leukerbad.detection.geometry import compute_grasp_from_polygon, estimate_gripper_opening_mm


def apply_clahe_bgr(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE strictly to the LAB luminance (`L`) channel, preserving hue and saturation."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    work = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)

    if work.ndim == 2:
        return clahe.apply(work)

    if work.ndim == 3:
        channels = work.shape[2]
        if channels == 1:
            return clahe.apply(work[:, :, 0])[:, :, np.newaxis]
        if channels == 3:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = clahe.apply(l_ch)
            return cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        if channels == 4:
            lab = cv2.cvtColor(work[:, :, :3], cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = clahe.apply(l_ch)
            bgr_eq = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
            return np.concatenate((bgr_eq, work[:, :, 3:]), axis=2)

    raise ValueError(f"Unsupported frame shape {frame.shape} for CLAHE.")


def _axis_windows(length: int, count: int, overlap: float) -> list[tuple[int, int]]:
    if count == 1:
        return [(0, length)]
    denom = 1.0 + (count - 1) * (1.0 - overlap)
    win = min(length, max(1, math.ceil(length / denom)))
    if win >= length:
        return [(0, length)]
    max_start = length - win
    starts = [round(i * max_start / (count - 1)) for i in range(count)]
    return [(s, s + win) for s in starts]


def generate_tile_windows(
    height: int,
    width: int,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Compute overlapping `(x1, y1, x2, y2)` tile windows covering `[0..width, 0..height]`."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Frame dimensions must be positive, got height={height}, width={width}.")
    rows, cols = int(grid[0]), int(grid[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"grid dimensions must be >= 1, got {grid}.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}.")

    y_spans = _axis_windows(height, rows, overlap)
    x_spans = _axis_windows(width, cols, overlap)
    windows: list[tuple[int, int, int, int]] = []
    for y1, y2 in y_spans:
        for x1, x2 in x_spans:
            windows.append((x1, y1, x2, y2))
    return windows


def _bbox_to_polygon(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _ensure_polygon(det: Detection) -> np.ndarray:
    if det.polygon is not None:
        pts = np.asarray(det.polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] >= 3:
            return pts
    return _bbox_to_polygon(det.bounding_box)


@dataclass(frozen=True)
class _Candidate:
    detection: Detection
    polygon: np.ndarray
    bbox: tuple[int, int, int, int]
    norm_label: str
    source: str
    sort_score: float


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _polygon_overlap_stats(
    cand_a: _Candidate,
    cand_b: _Candidate,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """Compute `(iou, containment)` between two polygons inside a tight local ROI mask."""
    if not _bboxes_overlap(cand_a.bbox, cand_b.bbox):
        return 0.0, 0.0

    rx1 = max(0, min(cand_a.bbox[0], cand_b.bbox[0]) - 2)
    ry1 = max(0, min(cand_a.bbox[1], cand_b.bbox[1]) - 2)
    rx2 = min(frame_w, max(cand_a.bbox[2], cand_b.bbox[2]) + 2)
    ry2 = min(frame_h, max(cand_a.bbox[3], cand_b.bbox[3]) + 2)
    rw = rx2 - rx1
    rh = ry2 - ry1
    if rw <= 0 or rh <= 0:
        return 0.0, 0.0

    mask_a = np.zeros((rh, rw), dtype=np.uint8)
    mask_b = np.zeros((rh, rw), dtype=np.uint8)
    pts_a = np.round(cand_a.polygon - np.array([[rx1, ry1]], dtype=np.float32)).astype(np.int32)
    pts_b = np.round(cand_b.polygon - np.array([[rx1, ry1]], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask_a, [pts_a], 1)
    cv2.fillPoly(mask_b, [pts_b], 1)

    area_a = float(np.count_nonzero(mask_a))
    area_b = float(np.count_nonzero(mask_b))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0, 0.0

    inter = float(np.count_nonzero(mask_a & mask_b))
    if inter <= 0.0:
        return 0.0, 0.0
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    containment = inter / min(area_a, area_b)
    return iou, containment


class TiledYoloSegDetector:
    """Overlapping tiled + full-frame detector with polygon-union seam & duplicate merging."""

    def __init__(
        self,
        base_detector: Detector,
        grid: tuple[int, int] = (2, 2),
        overlap: float = 0.25,
        include_full_frame: bool = True,
        iou_merge_threshold: float = 0.45,
        containment_threshold: float = 0.68,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        min_tile_trigger_px: int = 800,
    ) -> None:
        self.base_detector = base_detector
        self.grid = grid
        self.overlap = float(overlap)
        self.include_full_frame = bool(include_full_frame)
        self.iou_merge_threshold = float(iou_merge_threshold)
        self.containment_threshold = float(containment_threshold)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.min_tile_trigger_px = int(min_tile_trigger_px)
        self.mm_per_px: float | None = getattr(base_detector, "mm_per_px", None)

    def _build_detection(
        self,
        label: str,
        confidence: float,
        polygon: np.ndarray,
        bbox: tuple[int, int, int, int],
        base_metadata: dict[str, Any],
        tiled_meta: dict[str, Any],
    ) -> Detection:
        grasp = compute_grasp_from_polygon(polygon)
        if grasp is None:
            fallback_poly = _bbox_to_polygon(bbox)
            grasp = compute_grasp_from_polygon(fallback_poly)

        meta = dict(base_metadata)
        if self.mm_per_px is not None and grasp is not None:
            width_mm, opening_mm = estimate_gripper_opening_mm(
                grasp.width_px,
                mm_per_px=self.mm_per_px,
            )
            if width_mm is not None and opening_mm is not None:
                meta["width_mm"] = width_mm
                meta["gripper_opening_mm"] = opening_mm
        meta["tiled"] = tiled_meta

        return Detection(
            label=label,
            confidence=confidence,
            bounding_box=bbox,
            polygon=polygon,
            grasp=grasp,
            metadata=meta,
        )

    def _merge_cluster(
        self,
        cluster: list[_Candidate],
        frame_w: int,
        frame_h: int,
    ) -> Detection:
        best = max(cluster, key=lambda c: c.detection.confidence)
        sources = [c.source for c in cluster]

        if len(cluster) == 1:
            only = cluster[0]
            return self._build_detection(
                label=only.detection.label,
                confidence=only.detection.confidence,
                polygon=only.polygon,
                bbox=only.bbox,
                base_metadata=only.detection.metadata,
                tiled_meta={"merged_count": 1, "sources": sources},
            )

        rx1 = max(0, min(c.bbox[0] for c in cluster) - 4)
        ry1 = max(0, min(c.bbox[1] for c in cluster) - 4)
        rx2 = min(frame_w, max(c.bbox[2] for c in cluster) + 4)
        ry2 = min(frame_h, max(c.bbox[3] for c in cluster) + 4)
        rw = max(1, rx2 - rx1)
        rh = max(1, ry2 - ry1)

        union_mask = np.zeros((rh, rw), dtype=np.uint8)
        offset = np.array([[rx1, ry1]], dtype=np.float32)
        for cand in cluster:
            local_pts = np.round(cand.polygon - offset).astype(np.int32)
            cv2.fillPoly(union_mask, [local_pts], 1)

        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_poly: np.ndarray | None = None
        if contours:
            significant = [c for c in contours if cv2.contourArea(c) >= 4.0]
            chosen_contours = significant if significant else contours
            if len(chosen_contours) == 1:
                pts_local = chosen_contours[0].reshape(-1, 2).astype(np.float32)
            else:
                stacked = np.vstack(chosen_contours)
                pts_local = cv2.convexHull(stacked).reshape(-1, 2).astype(np.float32)

            if pts_local.shape[0] >= 3:
                pts_global = pts_local + offset
                pts_global[:, 0] = np.clip(pts_global[:, 0], 0.0, float(frame_w))
                pts_global[:, 1] = np.clip(pts_global[:, 1], 0.0, float(frame_h))
                merged_poly = pts_global

        if merged_poly is not None:
            bx1 = round(float(np.min(merged_poly[:, 0])))
            by1 = round(float(np.min(merged_poly[:, 1])))
            bx2 = round(float(np.max(merged_poly[:, 0])))
            by2 = round(float(np.max(merged_poly[:, 1])))
            merged_bbox = (bx1, by1, bx2, by2)
        else:
            merged_bbox = (
                min(c.bbox[0] for c in cluster),
                min(c.bbox[1] for c in cluster),
                max(c.bbox[2] for c in cluster),
                max(c.bbox[3] for c in cluster),
            )
            merged_poly = _bbox_to_polygon(merged_bbox)

        return self._build_detection(
            label=best.detection.label,
            confidence=best.detection.confidence,
            polygon=merged_poly,
            bbox=merged_bbox,
            base_metadata=best.detection.metadata,
            tiled_meta={"merged_count": len(cluster), "sources": sources},
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run multi-scale overlapping tiled + full-frame segmentation and merge detections."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            return []

        proc_frame = (
            apply_clahe_bgr(frame, clip_limit=self.clahe_clip_limit) if self.use_clahe else frame
        )
        height, width = int(proc_frame.shape[0]), int(proc_frame.shape[1])
        frame_area = float(height * width)
        run_tiles = max(height, width) >= self.min_tile_trigger_px

        candidates: list[_Candidate] = []

        if self.include_full_frame or not run_tiles:
            for det in self.base_detector.detect(proc_frame):
                poly = _ensure_polygon(det).copy()
                poly[:, 0] = np.clip(poly[:, 0], 0.0, float(width))
                poly[:, 1] = np.clip(poly[:, 1], 0.0, float(height))
                x1, y1, x2, y2 = det.bounding_box
                clamped_bbox = (
                    max(0, min(width, int(x1))),
                    max(0, min(height, int(y1))),
                    max(0, min(width, int(x2))),
                    max(0, min(height, int(y2))),
                )
                candidates.append(
                    _Candidate(
                        detection=det,
                        polygon=poly,
                        bbox=clamped_bbox,
                        norm_label=det.label.strip().lower(),
                        source="full",
                        sort_score=float(det.confidence),
                    )
                )

        if run_tiles:
            windows = generate_tile_windows(height, width, grid=self.grid, overlap=self.overlap)
            for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(windows):
                if (tx1, ty1, tx2, ty2) == (0, 0, width, height) and self.include_full_frame:
                    continue
                tile = proc_frame[ty1:ty2, tx1:tx2]
                for det in self.base_detector.detect(tile):
                    poly = _ensure_polygon(det).copy()
                    poly[:, 0] = np.clip(poly[:, 0] + float(tx1), 0.0, float(width))
                    poly[:, 1] = np.clip(poly[:, 1] + float(ty1), 0.0, float(height))
                    bx1, by1, bx2, by2 = det.bounding_box
                    global_bbox = (
                        max(0, min(width, int(bx1) + tx1)),
                        max(0, min(height, int(by1) + ty1)),
                        max(0, min(width, int(bx2) + tx1)),
                        max(0, min(height, int(by2) + ty1)),
                    )
                    bbox_area = float(
                        max(0, global_bbox[2] - global_bbox[0])
                        * max(0, global_bbox[3] - global_bbox[1])
                    )
                    bonus = 0.03 if bbox_area < 0.08 * frame_area else 0.0
                    candidates.append(
                        _Candidate(
                            detection=det,
                            polygon=poly,
                            bbox=global_bbox,
                            norm_label=det.label.strip().lower(),
                            source=f"tile_{tile_idx}",
                            sort_score=float(det.confidence) + bonus,
                        )
                    )

        if not candidates:
            return []

        candidates.sort(key=lambda c: c.sort_score, reverse=True)

        clusters: list[list[_Candidate]] = []
        for cand in candidates:
            matched_cluster: list[_Candidate] | None = None
            for cluster in clusters:
                if cluster[0].norm_label != cand.norm_label:
                    continue
                seed_iou, seed_cont = _polygon_overlap_stats(cand, cluster[0], width, height)
                if (
                    seed_iou >= self.iou_merge_threshold
                    or seed_cont >= self.containment_threshold
                ):
                    matched_cluster = cluster
                    break
                if any(
                    _polygon_overlap_stats(cand, member, width, height)[0]
                    >= self.iou_merge_threshold
                    for member in cluster[1:]
                ):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(cand)
            else:
                clusters.append([cand])

        merged_detections = [self._merge_cluster(cluster, width, height) for cluster in clusters]
        merged_detections.sort(key=lambda det: det.confidence, reverse=True)
        return merged_detections
