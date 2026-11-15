"""Open-world VLM scene discovery, ROI contour grasp extraction, and SO-101 pick-order ranking."""

import importlib.util
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector, GraspCandidate
from leukerbad.detection.geometry import compute_grasp_from_polygon, estimate_gripper_opening_mm
from leukerbad.detection.vlm_verifier import (
    VlmVerifiedDetector,
    _make_genai_image_part,
    _make_genai_json_config,
    _strip_json_fences,
)

_DISCOVERY_SYSTEM_INSTRUCTION = (
    "You are an open-world hazmat robotics scene inspector for a dual-arm SO-101 waste-sorting "
    "tabletop cell. Green boxes labeled 'FOUND' mark vapes already detected by Stage 1.\n"
    "Inspect the scene for any UNMARKED physical 3D disposable vapes, pod vapes, chunky "
    "metallic/matte rounded-square box-mod puffs (e.g., Lost Mary, Geek Bar, Vozol with offset "
    "duckbill mouthpiece), upright standing end-on vapes, or partially occluded vapes (e.g., a "
    "vape with a pen, cable, or receipt lying across it).\n"
    "STRICT EXCLUSIONS — DO NOT report:\n"
    "1. Any object already inside a green 'FOUND' box.\n"
    "2. 2D flat printed product photos/graphics on cardboard retail packaging (e.g., webcam bar "
    "printed on a teal AUKEY box, vape printed on a flat ElfBar carton — check cardboard box "
    "corners, flat print glare, barcodes, and lack of 3D relief).\n"
    "3. Pens, multi-color click pens (Bic), fountain pens, markers, highlighters (Stabilo).\n"
    "4. AA/AAA/18650 cylindrical batteries (Duracell +/- terminals), tactical LED flashlights "
    "(CREE), medical asthma inhalers (Salbutamol), AirPods cases, lipsticks, USB sticks, or "
    "BIC lighters.\n"
    "Respond ONLY with a JSON array (empty [] if none found) matching:\n"
    '[{"box_2d": [ymin, xmin, ymax, xmax], "detected_object": "<snake_case_name>", '
    '"confidence": <float 0.0..1.0>, "is_occluded": <bool>, '
    '"occlusion_note": "<what occludes it or empty>", "hazard_note": "<concise hazard note>"}]\n'
    "Coordinates in box_2d MUST use normalized 0..1000 integers [ymin, xmin, ymax, xmax]."
)


@dataclass(frozen=True)
class SceneDiscoveryCandidate:
    """Open-world VLM-discovered vape candidate in pixel coordinates."""

    bounding_box: tuple[int, int, int, int]
    detected_object: str
    confidence: float
    is_occluded: bool = False
    occlusion_note: str = ""
    hazard_note: str = ""


def _has_google_genai() -> bool:
    """Return True if ``google.genai`` is importable without importing it eagerly."""
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _is_confirmed_vape(det: Detection) -> bool:
    """Return True if ``det`` represents a confirmed target vape rather than a non-vape."""
    norm = det.label.strip().lower()
    return "vape" in norm and not norm.startswith(("non_vape", "non-vape"))


def _bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = (
        min(box_a[0], box_a[2]),
        min(box_a[1], box_a[3]),
        max(box_a[0], box_a[2]),
        max(box_a[1], box_a[3]),
    )
    bx1, by1, bx2, by2 = (
        min(box_b[0], box_b[2]),
        min(box_b[1], box_b[3]),
        max(box_b[0], box_b[2]),
        max(box_b[1], box_b[3]),
    )
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _point_in_bbox(px: float, py: float, bbox: tuple[int, int, int, int]) -> bool:
    x1, x2 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
    y1, y2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
    return x1 <= px <= x2 and y1 <= py <= y2


def _attach_scale_metadata(
    meta: dict[str, Any],
    grasp: GraspCandidate | None,
    mm_per_px: float | None,
) -> dict[str, Any]:
    if mm_per_px is not None and grasp is not None:
        width_mm, opening_mm = estimate_gripper_opening_mm(
            grasp.width_px,
            mm_per_px=mm_per_px,
        )
        if width_mm is not None and opening_mm is not None:
            meta["width_mm"] = width_mm
            meta["gripper_opening_mm"] = opening_mm
    return meta


def _bbox_grasp_fallback(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    mm_per_px: float | None,
) -> tuple[np.ndarray, GraspCandidate | None, dict[str, Any]]:
    polygon = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    grasp = compute_grasp_from_polygon(polygon)
    meta = _attach_scale_metadata({"contour_method": "bbox_fallback"}, grasp, mm_per_px)
    return polygon, grasp, meta


def extract_contour_grasp_in_roi(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    mm_per_px: float | None = None,
) -> tuple[np.ndarray, GraspCandidate | None, dict[str, Any]]:
    """Extract a foreground contour polygon and analytical 2D grasp inside a VLM ROI."""
    if frame is None or frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must be a non-empty 2D or 3D image array")

    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
    frame_h, frame_w = bgr.shape[:2]

    raw_x1, raw_y1, raw_x2, raw_y2 = bounding_box
    x1 = max(0, min(frame_w - 1, min(raw_x1, raw_x2)))
    y1 = max(0, min(frame_h - 1, min(raw_y1, raw_y2)))
    x2 = max(x1 + 1, min(frame_w, max(raw_x1, raw_x2)))
    y2 = max(y1 + 1, min(frame_h, max(raw_y1, raw_y2)))

    rw = x2 - x1
    rh = y2 - y1
    if rw < 8 or rh < 8:
        return _bbox_grasp_fallback(x1, y1, x2, y2, mm_per_px)

    ring_pad = 6
    ox1 = max(0, x1 - ring_pad)
    oy1 = max(0, y1 - ring_pad)
    ox2 = min(frame_w, x2 + ring_pad)
    oy2 = min(frame_h, y2 + ring_pad)

    outer_bgr = bgr[oy1:oy2, ox1:ox2]
    outer_lab = cv2.cvtColor(outer_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    ix1, iy1 = x1 - ox1, y1 - oy1
    ix2, iy2 = ix1 + rw, iy1 + rh
    roi_lab = outer_lab[iy1:iy2, ix1:ix2]
    roi_bgr = outer_bgr[iy1:iy2, ix1:ix2]

    ring_mask = np.ones(outer_lab.shape[:2], dtype=bool)
    ring_mask[iy1:iy2, ix1:ix2] = False
    if int(np.count_nonzero(ring_mask)) >= 16:
        bg_pixels = outer_lab[ring_mask]
    else:
        perim_mask = np.zeros((rh, rw), dtype=bool)
        border = min(2, max(1, min(rh, rw) // 4))
        perim_mask[:border, :] = True
        perim_mask[-border:, :] = True
        perim_mask[:, :border] = True
        perim_mask[:, -border:] = True
        bg_pixels = roi_lab[perim_mask]

    bg_median_lab = np.median(bg_pixels, axis=0)
    lab_dist = np.linalg.norm(roi_lab - bg_median_lab, axis=2)

    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    canny = cv2.Canny(gray_roi, 40, 120).astype(np.float32)

    lab_span = float(np.percentile(lab_dist, 95) - np.percentile(lab_dist, 5))
    grad_peak = float(grad_mag.max())
    if lab_span < 4.5 and grad_peak < 15.0:
        return _bbox_grasp_fallback(x1, y1, x2, y2, mm_per_px)

    lab_norm = cv2.normalize(lab_dist, None, 0.0, 255.0, cv2.NORM_MINMAX)
    grad_norm = (
        cv2.normalize(grad_mag, None, 0.0, 255.0, cv2.NORM_MINMAX)
        if grad_peak > 1e-3
        else np.zeros_like(lab_norm)
    )
    salience_u8 = np.clip(0.84 * lab_norm + 0.12 * grad_norm + 0.04 * canny, 0.0, 255.0).astype(
        np.uint8
    )

    _, binary = cv2.threshold(salience_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(rw * rh)
    roi_cx, roi_cy = rw * 0.5, rh * 0.5
    half_diag = max(1.0, 0.5 * math.hypot(rw, rh))

    best_cnt: np.ndarray | None = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.12 * roi_area or area > 0.96 * roi_area:
            continue
        moments = cv2.moments(cnt)
        if abs(moments["m00"]) > 1e-5:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + 0.5 * bw
            cy = by + 0.5 * bh

        norm_dist = math.hypot(cx - roi_cx, cy - roi_cy) / half_diag
        if norm_dist > 0.80:
            continue
        score = (area / roi_area) * max(0.15, 1.0 - 0.45 * norm_dist)
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return _bbox_grasp_fallback(x1, y1, x2, y2, mm_per_px)

    perimeter = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.008 * perimeter, True)
    pts_local = approx.reshape(-1, 2)
    if pts_local.shape[0] < 3:
        pts_local = best_cnt.reshape(-1, 2)
    if pts_local.shape[0] < 3:
        return _bbox_grasp_fallback(x1, y1, x2, y2, mm_per_px)

    polygon = pts_local.astype(np.float32) + np.array([float(x1), float(y1)], dtype=np.float32)
    grasp = compute_grasp_from_polygon(polygon)
    if grasp is None:
        return _bbox_grasp_fallback(x1, y1, x2, y2, mm_per_px)

    meta = _attach_scale_metadata({"contour_method": "lab_foreground"}, grasp, mm_per_px)
    return polygon, grasp, meta


def _jaw_tips_collide_with_neighbor(
    det_idx: int,
    grasp: GraspCandidate,
    all_detections: list[Detection],
) -> bool:
    cx, cy = grasp.center_px
    half_span = grasp.width_px * 0.65 + 12.0
    rad = math.radians(grasp.gripper_angle_deg)
    dx = math.cos(rad) * half_span
    dy = math.sin(rad) * half_span
    jaw_tips = ((cx - dx, cy - dy), (cx + dx, cy + dy))

    for other_idx, other in enumerate(all_detections):
        if other_idx == det_idx:
            continue
        poly_pts: np.ndarray | None = None
        if other.polygon is not None:
            reshaped = np.asarray(other.polygon, dtype=np.float32).reshape(-1, 2)
            if reshaped.shape[0] >= 3:
                poly_pts = reshaped

        for jx, jy in jaw_tips:
            if poly_pts is not None:
                if cv2.pointPolygonTest(poly_pts, (float(jx), float(jy)), False) >= 0.0:
                    return True
            elif _point_in_bbox(jx, jy, other.bounding_box):
                return True
    return False


def rank_grasp_candidates(
    detections: list[Detection],
    frame_shape: tuple[int, int] | None = None,
) -> list[Detection]:
    """Score jaw clearance, occlusion, and urgency to assign SO-101 ``pick_rank`` order."""
    scored_vapes: list[tuple[bool, float, float, Detection, list[str]]] = []
    non_vapes: list[Detection] = []

    for idx, det in enumerate(detections):
        if not _is_confirmed_vape(det):
            non_vapes.append(
                replace(
                    det,
                    metadata={
                        **det.metadata,
                        "robotics": {
                            "pick_rank": None,
                            "grasp_priority": 0.0,
                            "clearance_ok": False,
                            "blocking_reasons": ["non_target_class"],
                        },
                    },
                )
            )
            continue

        score = 0.55 * float(det.confidence) + 0.45
        clearance_ok = True
        reasons: list[str] = []

        vlm_info = det.metadata.get("vlm") if isinstance(det.metadata.get("vlm"), dict) else {}
        if vlm_info.get("is_occluded") is True:
            score -= 0.30
            clearance_ok = False
            reasons.append("occluded_by_object")

        if det.grasp is not None and _jaw_tips_collide_with_neighbor(idx, det.grasp, detections):
            score -= 0.25
            clearance_ok = False
            reasons.append("jaw_collision_neighbor")

        if frame_shape is not None and len(frame_shape) >= 2:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
            bx1, by1, bx2, by2 = det.bounding_box
            x1, x2 = min(bx1, bx2), max(bx1, bx2)
            y1, y2 = min(by1, by2), max(by1, by2)
            if x1 <= 6 or y1 <= 6 or x2 >= frame_w - 6 or y2 >= frame_h - 6:
                score -= 0.12
                reasons.append("near_frame_border")

        hazard_note = str(vlm_info.get("hazard_note") or "").lower()
        if any(tok in hazard_note for tok in ("swollen", "damaged", "leaking")):
            score += 0.08

        grasp_priority = max(0.0, min(1.0, score))
        scored_vapes.append((clearance_ok, grasp_priority, float(det.confidence), det, reasons))

    scored_vapes.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    ranked: list[Detection] = []
    for rank, (clearance_ok, priority, _conf, det, reasons) in enumerate(scored_vapes, start=1):
        ranked.append(
            replace(
                det,
                metadata={
                    **det.metadata,
                    "robotics": {
                        "pick_rank": rank,
                        "grasp_priority": round(priority, 4),
                        "clearance_ok": clearance_ok,
                        "blocking_reasons": reasons,
                    },
                },
            )
        )

    ranked.extend(non_vapes)
    return ranked


def _render_discovery_canvas(
    frame: np.ndarray,
    confirmed_detections: list[Detection],
    max_dim: int = 1024,
) -> np.ndarray:
    """Downscale ``frame`` to ``<= max_dim`` and draw green ``FOUND`` boxes around known vapes."""
    frame_h, frame_w = frame.shape[:2]
    peak = max(frame_h, frame_w)
    if peak > max_dim:
        scale = max_dim / float(peak)
        new_w = max(1, round(frame_w * scale))
        new_h = max(1, round(frame_h * scale))
        canvas = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        canvas = frame.copy()

    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    ch, cw = canvas.shape[:2]
    found_bgr = (0, 255, 0)
    for det in confirmed_detections:
        rx1, ry1, rx2, ry2 = det.bounding_box
        x1 = max(0, min(cw - 1, round(min(rx1, rx2) * scale)))
        y1 = max(0, min(ch - 1, round(min(ry1, ry2) * scale)))
        x2 = max(x1 + 1, min(cw, round(max(rx1, rx2) * scale)))
        y2 = max(y1 + 1, min(ch, round(max(ry1, ry2) * scale)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), found_bgr, 2, lineType=cv2.LINE_AA)
        ty = y1 - 5 if y1 >= 18 else min(ch - 4, y2 + 14)
        cv2.putText(
            canvas,
            "FOUND",
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            found_bgr,
            2,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _parse_discovery_candidates_json(
    raw_text: str,
    frame_shape: tuple[int, int],
) -> list[SceneDiscoveryCandidate]:
    """Parse Gemini normalized ``0..1000`` ``box_2d`` JSON array into pixel candidates."""
    cleaned = _strip_json_fences(raw_text)
    payload = json.loads(cleaned)

    items: list[Any] | None = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        list_val = (
            payload.get("candidates")
            or payload.get("detections")
            or payload.get("items")
            or payload.get("vapes")
        )
        if isinstance(list_val, list):
            items = list_val
        elif "box_2d" in payload:
            items = [payload]

    if items is None:
        raise ValueError("Discovery JSON response must be a JSON array")

    frame_h, frame_w = frame_shape
    parsed: list[SceneDiscoveryCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_box = item.get("box_2d")
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
            continue
        try:
            ymin_n, xmin_n, ymax_n, xmax_n = (float(v) for v in raw_box[:4])
        except (TypeError, ValueError):
            continue

        y_lo = max(0.0, min(1000.0, min(ymin_n, ymax_n)))
        y_hi = max(0.0, min(1000.0, max(ymin_n, ymax_n)))
        x_lo = max(0.0, min(1000.0, min(xmin_n, xmax_n)))
        x_hi = max(0.0, min(1000.0, max(xmin_n, xmax_n)))

        x1 = max(0, min(frame_w - 1, round(x_lo / 1000.0 * frame_w)))
        y1 = max(0, min(frame_h - 1, round(y_lo / 1000.0 * frame_h)))
        x2 = max(x1 + 1, min(frame_w, round(x_hi / 1000.0 * frame_w)))
        y2 = max(y1 + 1, min(frame_h, round(y_hi / 1000.0 * frame_h)))
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        raw_obj = (
            str(item.get("detected_object") or "disposable_vape").strip().lower().replace(" ", "_")
        )
        if not raw_obj:
            raw_obj = "disposable_vape"

        try:
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.75))))
        except (TypeError, ValueError):
            conf = 0.75

        raw_occ = item.get("is_occluded", False)
        if isinstance(raw_occ, bool):
            is_occluded = raw_occ
        elif isinstance(raw_occ, str):
            is_occluded = raw_occ.strip().lower() == "true"
        else:
            is_occluded = bool(raw_occ)

        occlusion_note = str(item.get("occlusion_note") or "").strip()
        default_hazard = "Embedded Li-ion cell + nicotine e-liquid fire hazard"
        hazard_note = str(item.get("hazard_note") or default_hazard).strip() or default_hazard

        parsed.append(
            SceneDiscoveryCandidate(
                bounding_box=(x1, y1, x2, y2),
                detected_object=raw_obj,
                confidence=conf,
                is_occluded=is_occluded,
                occlusion_note=occlusion_note,
                hazard_note=hazard_note,
            )
        )

    return parsed


class HybridVlmDetector:
    """Four-stage YOLO + batched VLM verifier + open-world scene discoverer + SO-101 ranker."""

    def __init__(
        self,
        base_detector: Detector,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        min_conf_to_verify: float = 0.12,
        max_conf_to_verify: float = 0.88,
        padding_ratio: float = 0.22,
        reject_non_vapes: bool = False,
        enable_scene_discovery: bool = True,
        discovery_trigger: Literal["always", "when_empty", "never"] = "always",
        discovery_iou_dedup: float = 0.30,
        mm_per_px: float | None = None,
        client: Any | None = None,
    ) -> None:
        if discovery_trigger not in {"always", "when_empty", "never"}:
            raise ValueError("discovery_trigger must be one of 'always', 'when_empty', or 'never'")
        if not (0.0 <= min_conf_to_verify <= max_conf_to_verify <= 1.0):
            raise ValueError(
                "Confidence bounds must satisfy "
                "0.0 <= min_conf_to_verify <= max_conf_to_verify <= 1.0"
            )
        if not (0.0 <= discovery_iou_dedup <= 1.0):
            raise ValueError("discovery_iou_dedup must satisfy 0.0 <= discovery_iou_dedup <= 1.0")
        if client is None and not _has_google_genai():
            raise ImportError(
                "google-genai is required for HybridVlmDetector. Install with: uv sync --extra vlm"
            )

        self.base_detector = base_detector
        self.model_name = model_name
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.min_conf_to_verify = min_conf_to_verify
        self.max_conf_to_verify = max_conf_to_verify
        self.padding_ratio = padding_ratio
        self.reject_non_vapes = reject_non_vapes
        self.enable_scene_discovery = enable_scene_discovery
        self.discovery_trigger: Literal["always", "when_empty", "never"] = discovery_trigger
        self.discovery_iou_dedup = discovery_iou_dedup
        self.mm_per_px = mm_per_px
        self._client = client

        self._verifier = VlmVerifiedDetector(
            base_detector=base_detector,
            model_name=model_name,
            api_key=self.api_key,
            min_conf_to_verify=min_conf_to_verify,
            max_conf_to_verify=max_conf_to_verify,
            padding_ratio=padding_ratio,
            reject_non_vapes=False,
            batch_verify=True,
            client=client,
            _skip_genai_check=True,
        )

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = self._verifier._get_or_create_client()
        return self._client

    def discover_missed_vapes(
        self,
        frame: np.ndarray,
        confirmed_detections: list[Detection],
        *,
        existing_detections: list[Detection] | None = None,
    ) -> list[SceneDiscoveryCandidate]:
        """Prompt Gemini on a ``FOUND``-annotated scene to localize any missed physical vapes."""
        if frame is None or frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return []

        client = self._get_or_create_client()
        canvas = _render_discovery_canvas(frame, confirmed_detections, max_dim=1024)
        encoded_ok, encoded_buf = cv2.imencode(".jpg", canvas)
        if not encoded_ok:
            raise RuntimeError("cv2.imencode failed for scene discovery image")

        image_part = _make_genai_image_part(encoded_buf.tobytes())
        config = _make_genai_json_config(_DISCOVERY_SYSTEM_INSTRUCTION)
        user_prompt = (
            f"{len(confirmed_detections)} confirmed vape(s) are already outlined in green "
            "with 'FOUND'. Return a JSON array of any UNMARKED physical 3D vapes, chunky "
            "box-mod puffs, upright standing vapes, or partially occluded vapes using "
            "box_2d=[ymin, xmin, ymax, xmax] in 0..1000 coordinates. Ignore all 'FOUND' "
            "boxes, 2D cardboard packaging graphics (AUKEY/ElfBar boxes), pens, batteries, "
            "flashlights, and inhalers."
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=[image_part, user_prompt],
            config=config,
        )
        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Empty response text from VLM scene discovery")

        raw_candidates = _parse_discovery_candidates_json(raw_text, frame.shape[:2])
        all_existing = (
            existing_detections if existing_detections is not None else confirmed_detections
        )

        deduped: list[SceneDiscoveryCandidate] = []
        for cand in sorted(raw_candidates, key=lambda c: c.confidence, reverse=True):
            cx = 0.5 * (cand.bounding_box[0] + cand.bounding_box[2])
            cy = 0.5 * (cand.bounding_box[1] + cand.bounding_box[3])

            overlaps_existing = any(
                _bbox_iou(cand.bounding_box, det.bounding_box) >= self.discovery_iou_dedup
                for det in all_existing
            )
            center_in_confirmed = any(
                _point_in_bbox(cx, cy, det.bounding_box) for det in confirmed_detections
            )
            overlaps_kept = any(
                _bbox_iou(cand.bounding_box, kept.bounding_box) >= self.discovery_iou_dedup
                for kept in deduped
            )
            if overlaps_existing or center_in_confirmed or overlaps_kept:
                continue
            deduped.append(cand)

        return deduped

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Execute batched crop verification, open-world scene discovery, and pick ranking."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            return []

        self._verifier.base_detector = self.base_detector
        stage1_dets = self._verifier.detect(frame)
        confirmed_vapes = [det for det in stage1_dets if _is_confirmed_vape(det)]

        has_credentials = self._client is not None or bool(self.api_key)
        should_discover = (
            self.enable_scene_discovery
            and has_credentials
            and (
                self.discovery_trigger == "always"
                or (self.discovery_trigger == "when_empty" and len(confirmed_vapes) == 0)
            )
        )

        discovered_dets: list[Detection] = []
        if should_discover:
            try:
                candidates = self.discover_missed_vapes(
                    frame,
                    confirmed_vapes,
                    existing_detections=stage1_dets,
                )
            except ImportError:
                raise
            except Exception:  # noqa: BLE001
                candidates = []

            eff_mm_per_px = (
                self.mm_per_px
                if self.mm_per_px is not None
                else getattr(self.base_detector, "mm_per_px", None)
            )
            for cand in candidates:
                poly, grasp, roi_meta = extract_contour_grasp_in_roi(
                    frame,
                    cand.bounding_box,
                    mm_per_px=eff_mm_per_px,
                )
                discovered_dets.append(
                    Detection(
                        label="vape",
                        confidence=round(cand.confidence, 4),
                        bounding_box=cand.bounding_box,
                        polygon=poly,
                        grasp=grasp,
                        metadata={
                            **roi_meta,
                            "source": "vlm_scene_discovery",
                            "vlm": {
                                "verified": True,
                                "discovered": True,
                                "model": self.model_name,
                                "is_vape": True,
                                "detected_object": cand.detected_object,
                                "confidence": cand.confidence,
                                "is_occluded": cand.is_occluded,
                                "occlusion_note": cand.occlusion_note,
                                "hazard_note": cand.hazard_note,
                            },
                        },
                    )
                )

        combined = stage1_dets + discovered_dets
        if self.reject_non_vapes:
            combined = [
                det
                for det in combined
                if not det.label.strip().lower().startswith(("non_vape", "non-vape"))
            ]

        return rank_grasp_candidates(combined, frame_shape=frame.shape[:2])
