"""Command-line entry point for Leukerbad vape & waste detection and grasp visualization."""

import argparse
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector
from leukerbad.detection.visualize import draw_detections
from leukerbad.detection.yolo_seg import YoloSegDetector


@dataclass(frozen=True)
class _SyntheticBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass(frozen=True)
class _SyntheticMasks:
    xy: list[np.ndarray]


@dataclass(frozen=True)
class _SyntheticResult:
    boxes: _SyntheticBoxes
    masks: _SyntheticMasks
    names: dict[int, str]


class _SyntheticYoloModel:
    """Offline Ultralytics-compatible stub model backing `--synthetic-demo`."""

    def __init__(
        self,
        polygons: list[np.ndarray],
        confidences: list[float],
        class_ids: list[int],
        names: dict[int, str],
    ) -> None:
        self._polygons = polygons
        self._confidences = confidences
        self._class_ids = class_ids
        self._names = names

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
        verbose: bool = False,
    ) -> list[_SyntheticResult]:
        del source, iou, device, verbose
        kept_xyxy: list[list[float]] = []
        kept_conf: list[float] = []
        kept_cls: list[int] = []
        kept_polys: list[np.ndarray] = []

        for poly, score, cls_id in zip(
            self._polygons, self._confidences, self._class_ids, strict=True
        ):
            if score < conf:
                continue
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            kept_xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
            kept_conf.append(score)
            kept_cls.append(cls_id)
            kept_polys.append(poly)

        xyxy_arr = (
            np.asarray(kept_xyxy, dtype=np.float32)
            if kept_xyxy
            else np.zeros((0, 4), dtype=np.float32)
        )
        boxes = _SyntheticBoxes(
            xyxy=xyxy_arr,
            conf=np.asarray(kept_conf, dtype=np.float32),
            cls=np.asarray(kept_cls, dtype=np.int64),
        )
        masks = _SyntheticMasks(xy=kept_polys)
        return [_SyntheticResult(boxes=boxes, masks=masks, names=self._names)]


def _rotated_capsule_polygon(
    center: tuple[float, float],
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """Create a 12-vertex rounded-end cylinder polygon rotated by `angle_deg`."""
    hl = length / 2.0
    hw = width / 2.0
    chamfer = min(hw * 0.45, hl * 0.15)
    local_pts = np.array(
        [
            [-hl + chamfer, -hw],
            [hl - chamfer, -hw],
            [hl, -hw * 0.55],
            [hl, hw * 0.55],
            [hl - chamfer, hw],
            [-hl + chamfer, hw],
            [-hl, hw * 0.55],
            [-hl, -hw * 0.55],
        ],
        dtype=np.float32,
    )
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return local_pts @ rot.T + np.asarray(center, dtype=np.float32)


def _create_synthetic_demo_scene() -> tuple[np.ndarray, _SyntheticYoloModel]:
    """Generate an offline synthetic conveyor tabletop frame and matching segmentation model."""
    height, width = 480, 640
    frame = np.full((height, width, 3), (54, 58, 62), dtype=np.uint8)

    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (62, 66, 70), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (62, 66, 70), 1)

    specs: list[
        tuple[tuple[float, float], float, float, float, tuple[int, int, int], float, int]
    ] = [
        ((195.0, 205.0), 148.0, 30.0, -32.0, (190, 95, 45), 0.94, 0),
        ((445.0, 175.0), 100.0, 26.0, 52.0, (90, 175, 235), 0.89, 1),
        ((335.0, 365.0), 156.0, 16.0, 14.0, (175, 180, 175), 0.82, 2),
    ]
    names = {0: "disposable_vape", 1: "vape_pod", 2: "non_vape_pen"}

    polygons: list[np.ndarray] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for center, length, obj_w, angle_deg, body_bgr, score, cls_id in specs:
        poly = _rotated_capsule_polygon(center, length, obj_w, angle_deg)
        shadow = np.round(poly + np.array([4.0, 5.0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(frame, [shadow.reshape(-1, 1, 2)], (34, 36, 38), lineType=cv2.LINE_AA)

        pts_i32 = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [pts_i32], body_bgr, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_i32], isClosed=True, color=(30, 30, 30), thickness=1)

        rad = math.radians(angle_deg)
        tip_cx = round(center[0] + math.cos(rad) * (length * 0.42))
        tip_cy = round(center[1] + math.sin(rad) * (length * 0.42))
        cv2.circle(frame, (tip_cx, tip_cy), max(3, int(obj_w * 0.22)), (40, 40, 45), -1)

        polygons.append(poly)
        confidences.append(score)
        class_ids.append(cls_id)

    model = _SyntheticYoloModel(
        polygons=polygons,
        confidences=confidences,
        class_ids=class_ids,
        names=names,
    )
    return frame, model


def build_parser() -> argparse.ArgumentParser:
    """Create the `leukerbad-detect` command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="leukerbad-detect",
        description="Detect discarded vapes/waste and render SO-101 2D grasp overlays.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", type=Path, help="Path to an input image file.")
    source_group.add_argument("--camera", type=int, help="OpenCV webcam device index (e.g. 0).")
    source_group.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run offline on a generated synthetic tabletop scene with rotated vapes.",
    )

    parser.add_argument("--save", type=Path, help="Save the annotated output frame to this path.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s-seg.pt",
        help="Ultralytics segmentation checkpoint path or name (default: yolo11s-seg.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated whitelist of class labels.",
    )
    parser.add_argument(
        "--mm-per-px",
        type=float,
        default=None,
        help="Optional tabletop scale (mm per pixel) to compute SO-101 jaw opening in mm.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        default=False,
        help=(
            "Enable overlapping 2x2 + full-frame tiled segmentation for high-res (1080p) "
            "overhead tabletop scenes."
        ),
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        default=False,
        help="Apply LAB luminance CLAHE contrast normalization before detection.",
    )
    parser.add_argument(
        "--verifier",
        choices=["none", "gemini", "hybrid"],
        default="none",
        help=(
            "Optional VLM backend: 'none', 'gemini' (crop verifier), or 'hybrid' "
            "(batched crop verifier + missed-vape scene discovery + SO-101 pick ranker)."
        ),
    )
    parser.add_argument(
        "--discovery-trigger",
        choices=["always", "when_empty", "never"],
        default="always",
        help=(
            "With --verifier hybrid, when to run full-scene Gemini missed-vape discovery "
            "(default: always)."
        ),
    )
    parser.add_argument(
        "--only-vapes",
        action="store_true",
        default=False,
        help="Filter out non-vape detections.",
    )
    parser.add_argument(
        "--verify-min",
        type=float,
        default=0.20,
        help="Minimum confidence to trigger VLM verification (default: 0.20).",
    )
    parser.add_argument(
        "--verify-max",
        type=float,
        default=0.85,
        help="Maximum confidence to trigger VLM verification (default: 0.85).",
    )
    parser.add_argument(
        "--verify-every",
        type=int,
        default=30,
        help=(
            "In --camera mode with VLM verifier, run VLM every N frames "
            "(0 = only when 'v' is pressed; default: 30)."
        ),
    )
    return parser


def _parse_allowed_labels(raw_labels: str | None) -> set[str] | None:
    if raw_labels is None:
        return None
    parsed = {item.strip() for item in raw_labels.split(",") if item.strip()}
    return parsed or None


def _filter_only_vapes(detections: list[Detection]) -> list[Detection]:
    return [
        det
        for det in detections
        if "vape" in det.label.lower()
        and not det.label.lower().startswith(("non_vape", "non-vape"))
    ]


def _print_detection_summary(detections: list[Detection]) -> None:
    print(f"Detected {len(detections)} object(s):")
    for idx, det in enumerate(detections, start=1):
        robotics_meta = det.metadata.get("robotics")
        pick_rank = (
            robotics_meta.get("pick_rank") if isinstance(robotics_meta, dict) else None
        )
        rank_tag = f"pick=#{pick_rank}, " if isinstance(pick_rank, int) else ""
        if det.grasp is not None:
            cx, cy = det.grasp.center_px
            extra = (
                f"{rank_tag}center=({cx:.1f}, {cy:.1f})px, "
                f"wrist={det.grasp.gripper_angle_deg:.1f}deg, "
                f"width={det.grasp.width_px:.1f}px"
            )
            if "gripper_opening_mm" in det.metadata:
                extra += f", opening={det.metadata['gripper_opening_mm']:.1f}mm"
        else:
            extra = f"{rank_tag}bbox={det.bounding_box}"
        print(f"  [{idx}] {det.label} ({det.confidence:.2f}) | {extra}")


def _save_frame(output_path: Path, frame: np.ndarray) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(output_path), frame))
    if ok:
        print(f"Saved annotated image to {output_path}")
    else:
        print(f"Error: failed to write image to {output_path}", file=sys.stderr)
    return ok


def _format_camera_snapshot_path(save_path: Path | None, snapshot_index: int) -> Path:
    """Return a non-overwriting numbered path for interactive camera snapshots."""
    if save_path is None:
        return Path(f"leukerbad_snapshot_{snapshot_index:03d}.png")
    suffix = save_path.suffix or ".png"
    return save_path.with_name(f"{save_path.stem}_{snapshot_index:03d}{suffix}")


@dataclass(frozen=True)
class _CachedVlmEntry:
    bounding_box: tuple[int, int, int, int]
    label: str
    confidence: float
    vlm_meta: dict[str, Any] | None
    raw_yolo_label: str | None
    robotics_meta: dict[str, Any] | None = None
    discovered_det: Detection | None = None


def _bbox_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_cached_verdict(
    bbox: tuple[int, int, int, int],
    cache: list[_CachedVlmEntry],
) -> _CachedVlmEntry | None:
    if not cache:
        return None
    ax = 0.5 * (bbox[0] + bbox[2])
    ay = 0.5 * (bbox[1] + bbox[3])
    best_entry: _CachedVlmEntry | None = None
    best_score = -1.0

    for entry in cache:
        cb = entry.bounding_box
        iou = _bbox_iou(bbox, cb)
        bx = 0.5 * (cb[0] + cb[2])
        by = 0.5 * (cb[1] + cb[3])
        diag = max(1.0, math.hypot(cb[2] - cb[0], cb[3] - cb[1]))
        norm_dist = math.hypot(ax - bx, ay - by) / diag
        if iou < 0.30 and norm_dist > 0.50:
            continue
        score = iou + max(0.0, 1.0 - norm_dist) * 0.25
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry


def _apply_cached_verdicts(
    raw_detections: list[Detection],
    cache: list[_CachedVlmEntry],
) -> list[Detection]:
    if not cache:
        return raw_detections
    merged: list[Detection] = []
    for det in raw_detections:
        matched = _match_cached_verdict(det.bounding_box, cache)
        if matched is None:
            merged.append(det)
            continue
        vlm_info = matched.vlm_meta
        if vlm_info is not None and not vlm_info.get("verified", False):
            merged.append(replace(det, metadata={**det.metadata, "vlm": vlm_info}))
            continue
        new_meta: dict[str, Any] = {**det.metadata}
        if matched.raw_yolo_label is not None:
            new_meta["raw_yolo_label"] = matched.raw_yolo_label
        elif matched.label != det.label:
            new_meta["raw_yolo_label"] = det.label
        if vlm_info is not None:
            new_meta["vlm"] = {**vlm_info, "cached": True}
        if matched.robotics_meta is not None:
            new_meta["robotics"] = dict(matched.robotics_meta)
        merged.append(
            replace(
                det,
                label=matched.label,
                confidence=matched.confidence,
                metadata=new_meta,
            )
        )
    for entry in cache:
        if entry.discovered_det is None:
            continue
        if any(_bbox_iou(entry.bounding_box, d.bounding_box) >= 0.30 for d in merged):
            continue
        merged.append(entry.discovered_det)
    return merged


def _run_vlm_pass_and_build_cache(
    detector: Detector,
    frame: np.ndarray,
) -> tuple[list[Detection], list[_CachedVlmEntry]]:
    had_reject_attr = hasattr(detector, "reject_non_vapes")
    orig_reject = bool(getattr(detector, "reject_non_vapes", False))
    if had_reject_attr and orig_reject:
        try:
            detector.reject_non_vapes = False  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            had_reject_attr = False
    try:
        verified = detector.detect(frame)
    finally:
        if had_reject_attr and orig_reject:
            detector.reject_non_vapes = orig_reject  # type: ignore[attr-defined]

    cache = [
        _CachedVlmEntry(
            bounding_box=det.bounding_box,
            label=det.label,
            confidence=det.confidence,
            vlm_meta=(
                dict(det.metadata["vlm"])
                if isinstance(det.metadata.get("vlm"), dict)
                else None
            ),
            raw_yolo_label=(
                str(det.metadata["raw_yolo_label"])
                if "raw_yolo_label" in det.metadata
                else None
            ),
            robotics_meta=(
                dict(det.metadata["robotics"])
                if isinstance(det.metadata.get("robotics"), dict)
                else None
            ),
            discovered_det=(
                det if det.metadata.get("source") == "vlm_scene_discovery" else None
            ),
        )
        for det in verified
    ]
    if orig_reject:
        verified = [
            det
            for det in verified
            if not (
                isinstance(det.metadata.get("vlm"), dict)
                and det.metadata["vlm"].get("verified") is True
                and det.metadata["vlm"].get("is_vape") is False
            )
        ]
    return verified, cache


def _build_detector(
    args: argparse.Namespace,
    injected_model: Any | None = None,
) -> Detector:
    allowed_labels = _parse_allowed_labels(args.labels)
    detector: Detector = YoloSegDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        allowed_labels=allowed_labels,
        mm_per_px=args.mm_per_px,
        model=injected_model,
    )
    if getattr(args, "tiled", False) or getattr(args, "clahe", False):
        from leukerbad.detection.tiled_detector import TiledYoloSegDetector

        use_tiled = bool(getattr(args, "tiled", False))
        detector = TiledYoloSegDetector(
            detector,
            use_clahe=bool(getattr(args, "clahe", False)),
            min_tile_trigger_px=800 if use_tiled else 1_000_000,
        )
    if args.verifier == "gemini":
        from leukerbad.detection.vlm_verifier import VlmVerifiedDetector

        detector = VlmVerifiedDetector(
            detector,
            min_conf_to_verify=args.verify_min,
            max_conf_to_verify=args.verify_max,
            reject_non_vapes=args.only_vapes,
        )
    elif args.verifier == "hybrid":
        from leukerbad.detection.vlm_scene import HybridVlmDetector

        detector = HybridVlmDetector(
            detector,
            min_conf_to_verify=args.verify_min,
            max_conf_to_verify=args.verify_max,
            reject_non_vapes=args.only_vapes,
            discovery_trigger=getattr(args, "discovery_trigger", "always"),
            mm_per_px=args.mm_per_px,
        )
    return detector


def _run_camera_loop(detector: Detector, args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: unable to open camera index {args.camera}.", file=sys.stderr)
        return 1

    print(
        "Live camera started — press 'q' to quit, 's' to save snapshot, "
        "'v' to force VLM verify."
    )
    snapshot_count = 0
    frame_idx = 0
    force_verify = False
    vlm_cache: list[_CachedVlmEntry] = []
    base_detector: Detector | None = (
        getattr(detector, "base_detector", None)
        if getattr(args, "verifier", "none") != "none"
        else None
    )
    verify_every = max(0, int(getattr(args, "verify_every", 30)))
    last_annotated: np.ndarray | None = None
    saved_manually = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            ran_vlm = False
            if base_detector is None:
                detections = detector.detect(frame)
            else:
                should_verify = force_verify or (
                    verify_every > 0 and frame_idx % verify_every == 0
                )
                if should_verify:
                    detections, vlm_cache = _run_vlm_pass_and_build_cache(detector, frame)
                    force_verify = False
                    ran_vlm = True
                else:
                    raw_detections = base_detector.detect(frame)
                    detections = _apply_cached_verdicts(raw_detections, vlm_cache)

            if args.only_vapes:
                detections = _filter_only_vapes(detections)

            annotated = draw_detections(frame, detections)
            last_annotated = annotated
            cv2.imshow("Leukerbad Vape & Waste Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v") and base_detector is not None:
                if not ran_vlm:
                    detections, vlm_cache = _run_vlm_pass_and_build_cache(detector, frame)
                    if args.only_vapes:
                        detections = _filter_only_vapes(detections)
                    annotated = draw_detections(frame, detections)
                    last_annotated = annotated
                    cv2.imshow("Leukerbad Vape & Waste Detection", annotated)
                else:
                    force_verify = True
            if key == ord("s"):
                snapshot_count += 1
                target = _format_camera_snapshot_path(args.save, snapshot_count)
                _save_frame(target, annotated)
                saved_manually = True

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if (
        args.save is not None
        and last_annotated is not None
        and not saved_manually
        and not _save_frame(args.save, last_annotated)
    ):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Leukerbad detection and grasp visualization CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image is None and args.camera is None and not args.synthetic_demo:
        print(
            "Leukerbad object detection and robotics CLI ready. "
            "Pass --synthetic-demo, --image <path>, or --camera <index> (see --help)."
        )
        return 0

    if args.verify_every < 0:
        print("Error: --verify-every must be >= 0.", file=sys.stderr)
        return 1

    if args.verifier in {"gemini", "hybrid"} and not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        print(
            f"Error: --verifier {args.verifier} requires GEMINI_API_KEY or GOOGLE_API_KEY "
            "to be set in the environment.",
            file=sys.stderr,
        )
        return 1

    try:
        if args.camera is not None:
            detector = _build_detector(args)
            return _run_camera_loop(detector, args)

        if args.synthetic_demo:
            frame, synthetic_model = _create_synthetic_demo_scene()
            injected = None if Path(args.weights).is_file() else synthetic_model
            detector = _build_detector(args, injected_model=injected)
        else:
            if not args.image.is_file():
                print(f"Error: image file not found: {args.image}", file=sys.stderr)
                return 1
            loaded = cv2.imread(str(args.image))
            if loaded is None:
                print(f"Error: failed to decode image: {args.image}", file=sys.stderr)
                return 1
            frame = loaded
            detector = _build_detector(args)
    except (ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    detections = detector.detect(frame)
    if args.only_vapes:
        detections = _filter_only_vapes(detections)

    _print_detection_summary(detections)
    annotated = draw_detections(frame, detections)

    if args.save is not None and not _save_frame(args.save, annotated):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
