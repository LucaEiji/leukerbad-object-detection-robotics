"""Visual overlay renderer for segmentation masks, oriented boxes, and SO-101 grasp poses."""

import math
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, GraspCandidate

VAPE_COLOR_BGR: tuple[int, int, int] = (0, 60, 255)
GENERAL_COLOR_BGR: tuple[int, int, int] = (80, 200, 60)
GRIPPER_COLOR_BGR: tuple[int, int, int] = (255, 230, 0)
CROSSHAIR_COLOR_BGR: tuple[int, int, int] = (255, 255, 255)
DARK_OUTLINE_BGR: tuple[int, int, int] = (20, 20, 20)


def _select_color_bgr(label: str) -> tuple[int, int, int]:
    """Return red/orange for vape classes and green/teal for non-vape / general waste."""
    normalized = label.strip().lower()
    if not normalized.startswith(("non_vape", "non-vape")) and "vape" in normalized:
        return VAPE_COLOR_BGR
    return GENERAL_COLOR_BGR


def _format_vlm_tag(vlm_info: Any) -> str:
    """Build a compact status/hazard badge from `detection.metadata['vlm']`."""
    if isinstance(vlm_info, dict):
        suffix = "+OCCLUDED" if vlm_info.get("is_occluded") is True else ""
        hazard_flag = bool(
            vlm_info.get("battery_hazard")
            or vlm_info.get("damaged")
            or vlm_info.get("is_damaged")
            or vlm_info.get("hazard") is True
        )
        hazard_level = vlm_info.get("hazard_level")
        if hazard_level:
            return f"VLM:{hazard_level}{suffix}"
        if hazard_flag:
            return f"VLM:HAZARD{suffix}"
        if vlm_info.get("discovered") is True:
            return f"VLM:disc{suffix}"
        if vlm_info.get("status"):
            return f"VLM:{vlm_info['status']}{suffix}"
        if vlm_info.get("category"):
            return f"VLM:{vlm_info['category']}{suffix}"
        if "is_vape" in vlm_info:
            base = "VLM:vape" if vlm_info["is_vape"] else "VLM:non_vape"
            return f"{base}{suffix}"
        return f"VLM:verified{suffix}"
    if isinstance(vlm_info, str) and vlm_info.strip():
        return f"VLM:{vlm_info.strip()}"
    return "VLM:verified"


def _build_label_text(detection: Detection) -> str:
    """Format the readable text pill string for a single detection."""
    robotics_info = detection.metadata.get("robotics")
    pick_rank = (
        robotics_info.get("pick_rank")
        if isinstance(robotics_info, dict)
        else None
    )
    prefix = f"#{pick_rank} " if isinstance(pick_rank, int) else ""
    parts = [f"{prefix}{detection.label} {detection.confidence:.2f}"]
    if detection.grasp is not None:
        parts.append(f"wrist={detection.grasp.gripper_angle_deg:.1f}deg")
        parts.append(f"w={detection.grasp.width_px:.0f}px")
    if "vlm" in detection.metadata and detection.metadata["vlm"] is not None:
        parts.append(_format_vlm_tag(detection.metadata["vlm"]))
    return " | ".join(parts)


def _draw_grasp_geometry(
    canvas: np.ndarray,
    grasp: GraspCandidate,
    color: tuple[int, int, int],
) -> None:
    """Draw oriented bounding box, antipodal approach line with jaw ticks, and crosshair."""
    if len(grasp.box_points_px) >= 4:
        box_pts = (
            np.round(np.asarray(grasp.box_points_px, dtype=np.float32))
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
        cv2.polylines(
            canvas, [box_pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA
        )

    cx, cy = grasp.center_px
    half_span = grasp.width_px * 0.65 + 12.0
    rad = math.radians(grasp.gripper_angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)

    j1x, j1y = cx - ux * half_span, cy - uy * half_span
    j2x, j2y = cx + ux * half_span, cy + uy * half_span
    p1 = (round(j1x), round(j1y))
    p2 = (round(j2x), round(j2y))

    cv2.line(canvas, p1, p2, DARK_OUTLINE_BGR, thickness=4, lineType=cv2.LINE_AA)
    cv2.line(canvas, p1, p2, GRIPPER_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)

    tick_half = max(6.0, min(14.0, grasp.length_px * 0.12))
    px_perp, py_perp = -uy * tick_half, ux * tick_half
    for jx, jy in ((j1x, j1y), (j2x, j2y)):
        t1 = (round(jx - px_perp), round(jy - py_perp))
        t2 = (round(jx + px_perp), round(jy + py_perp))
        cv2.line(canvas, t1, t2, DARK_OUTLINE_BGR, thickness=4, lineType=cv2.LINE_AA)
        cv2.line(canvas, t1, t2, GRIPPER_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)

    icx, icy = round(cx), round(cy)
    arm = 6
    cv2.line(canvas, (icx - arm, icy), (icx + arm, icy), DARK_OUTLINE_BGR, 3, lineType=cv2.LINE_AA)
    cv2.line(canvas, (icx, icy - arm), (icx, icy + arm), DARK_OUTLINE_BGR, 3, lineType=cv2.LINE_AA)
    cv2.line(
        canvas, (icx - arm, icy), (icx + arm, icy), CROSSHAIR_COLOR_BGR, 1, lineType=cv2.LINE_AA
    )
    cv2.line(
        canvas, (icx, icy - arm), (icx, icy + arm), CROSSHAIR_COLOR_BGR, 1, lineType=cv2.LINE_AA
    )
    cv2.circle(canvas, (icx, icy), 2, color, -1, lineType=cv2.LINE_AA)


def _draw_text_pill(
    canvas: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Render a high-contrast rounded-style rectangular text pill clamped within the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x, pad_y = 6, 5
    pill_w = text_w + 2 * pad_x
    pill_h = text_h + baseline + 2 * pad_y

    img_h, img_w = canvas.shape[:2]
    x0 = max(2, min(anchor[0], max(2, img_w - pill_w - 2)))
    y0 = anchor[1] - pill_h - 4
    if y0 < 2:
        y0 = min(max(2, anchor[1] + 4), max(2, img_h - pill_h - 2))

    x1 = min(img_w - 1, x0 + pill_w)
    y1 = min(img_h - 1, y0 + pill_h)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), DARK_OUTLINE_BGR, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
    text_org = (x0 + pad_x, y0 + pad_y + text_h)
    cv2.putText(
        canvas,
        text,
        text_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    alpha: float = 0.35,
) -> np.ndarray:
    """Render masks, oriented grasp boxes, SO-101 gripper axes, and telemetry pills on a copy."""
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("`frame` must be a valid NumPy ndarray.")

    canvas = frame.copy()
    if canvas.size == 0 or not detections:
        return canvas

    clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
    mask_layer = canvas.copy()
    has_mask = False

    for detection in detections:
        if detection.polygon is None:
            continue
        pts = np.asarray(detection.polygon, dtype=np.float32)
        if pts.size < 6:
            continue
        pts_i32 = np.round(pts.reshape(-1, 1, 2)).astype(np.int32)
        if pts_i32.shape[0] < 3:
            continue
        color = _select_color_bgr(detection.label)
        cv2.fillPoly(mask_layer, [pts_i32], color, lineType=cv2.LINE_AA)
        has_mask = True

    if has_mask and clamped_alpha > 0.0:
        cv2.addWeighted(mask_layer, clamped_alpha, canvas, 1.0 - clamped_alpha, 0.0, dst=canvas)

    for detection in detections:
        color = _select_color_bgr(detection.label)
        if detection.polygon is not None:
            pts = np.asarray(detection.polygon, dtype=np.float32)
            if pts.size >= 6:
                pts_i32 = np.round(pts.reshape(-1, 1, 2)).astype(np.int32)
                if pts_i32.shape[0] >= 3:
                    cv2.polylines(
                        canvas,
                        [pts_i32],
                        isClosed=True,
                        color=color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        x1, y1, x2, y2 = detection.bounding_box
        if detection.grasp is not None:
            _draw_grasp_geometry(canvas, detection.grasp, color)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        pill_text = _build_label_text(detection)
        _draw_text_pill(canvas, pill_text, anchor=(x1, y1), color=color)

    return canvas
