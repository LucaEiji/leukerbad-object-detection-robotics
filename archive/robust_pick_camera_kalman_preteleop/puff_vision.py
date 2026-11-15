"""Multi-cue puff detection & self-calibrated Eye-in-Hand visual servoing.

Works out-of-the-box on dark, colored, or white-on-white elongated puffs without
GPU training, auto-discovers the right USB camera index on macOS, and supports
plugging in an external Data-Science detector (`custom_puff_detector.py`).
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from so101_ik import EXPECTED_CALIBRATION_SHA256, ROOT, SO101CalibratedIK, fk_camera_frame
from wrist_object import detect_vape as _legacy_detect_vape

DEFAULT_SERVO_CALIB_PATH = ROOT / "real_transfer/wrist_servo_calibration.json"
DEFAULT_CUSTOM_DETECTOR_PATH = ROOT / "custom_puff_detector.py"
DEFAULT_HOVER_Z_MM = 80.0


class TableTargetKalman:
    """Constant-velocity Kalman tracker for a puff in the fixed table frame.

    Measurements are table-plane millimetres reconstructed from the wrist-camera
    Jacobian. A short image dropout can therefore be bridged while the robot is
    still at the observation height. Uncertainty grows during prediction; callers
    must not treat an uncertain prediction as a confirmed visual lock.
    """

    def __init__(self, measurement_std_mm: float = 2.5, acceleration_std_mm_s2: float = 20.0):
        self.measurement_var = float(measurement_std_mm) ** 2
        self.acceleration_var = float(acceleration_std_mm_s2) ** 2
        self.state: np.ndarray | None = None  # x, y, vx, vy
        self.covariance: np.ndarray | None = None
        self.last_time: float | None = None

    @property
    def uncertainty_mm(self) -> float:
        if self.covariance is None:
            return float("inf")
        pos_cov = self.covariance[:2, :2]
        return float(np.sqrt(max(0.0, np.linalg.eigvalsh(pos_cov).max())))

    def _predict(self, now: float) -> None:
        if self.state is None or self.covariance is None or self.last_time is None:
            return
        dt = float(np.clip(now - self.last_time, 0.0, 0.5))
        if dt > 0.0:
            transition = np.array(
                [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            )
            q = self.acceleration_var
            block = q * np.array(
                [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float
            )
            process = np.zeros((4, 4), dtype=float)
            process[np.ix_([0, 2], [0, 2])] = block
            process[np.ix_([1, 3], [1, 3])] = block
            self.state = transition @ self.state
            self.covariance = transition @ self.covariance @ transition.T + process
        self.last_time = now

    def predict(self, now: float | None = None) -> tuple[np.ndarray, float] | None:
        if self.state is None:
            return None
        self._predict(time.monotonic() if now is None else float(now))
        return self.state[:2].copy(), self.uncertainty_mm

    def update(self, xy_mm: tuple[float, float] | np.ndarray, now: float | None = None) -> tuple[np.ndarray, float]:
        measurement = np.asarray(xy_mm, dtype=float)
        if measurement.shape != (2,) or not np.isfinite(measurement).all():
            raise ValueError("Mesure Kalman (X,Y) invalide.")
        stamp = time.monotonic() if now is None else float(now)
        if self.state is None:
            self.state = np.r_[measurement, 0.0, 0.0]
            self.covariance = np.diag(
                [self.measurement_var, self.measurement_var, 100.0, 100.0]
            )
            self.last_time = stamp
            return self.state[:2].copy(), self.uncertainty_mm

        self._predict(stamp)
        assert self.covariance is not None
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        noise = np.eye(2) * self.measurement_var
        innovation = measurement - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        identity = np.eye(4)
        # Joseph form keeps covariance symmetric and positive semidefinite.
        residual = identity - gain @ observation
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        return self.state[:2].copy(), self.uncertainty_mm


def wrap_angle_90(angle_deg: float) -> float:
    """Normalize an undirected principal-axis angle difference into [-90, +90) deg."""
    return float((float(angle_deg) + 90.0) % 180.0 - 90.0)


def load_custom_detector(path: Path | None = None) -> Callable[[np.ndarray], dict | None] | None:
    """Load optional Data-Science detector plugin `detect(frame_bgr)` if present."""
    target = Path(path) if path is not None else DEFAULT_CUSTOM_DETECTOR_PATH
    if not target.is_file():
        return None
    spec = importlib.util.spec_from_file_location("custom_puff_detector", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le détecteur personnalisé : {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "detect", None)
    if not callable(fn):
        raise RuntimeError(f"{target} doit définir une fonction callable `detect(frame_bgr)`.")
    return fn


def _extract_candidates_from_mask(
    mask: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    allow_border: bool = False,
    min_area: float = 380.0,
    min_width: float = 7.0,
    min_aspect: float | None = None,
    min_length_fraction: float | None = None,
) -> list[dict]:
    h, w = frame_shape
    x0, y0, x1, y1 = roi_bounds
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        (cx, cy), (side_a, side_b), _ = cv2.minAreaRect(contour)
        length_px = float(max(side_a, side_b))
        width_px = float(min(side_a, side_b))
        if width_px < min_width:
            continue
        aspect = length_px / max(width_px, 1e-6)
        aspect_floor = min_aspect if min_aspect is not None else (1.7 if allow_border else 2.1)
        if not (aspect_floor <= aspect <= 11.5):
            continue
        length_fraction = min_length_fraction if min_length_fraction is not None else (0.08 if allow_border else 0.12)
        min_len = length_fraction * w
        if not (min_len <= length_px <= 0.92 * w):
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        touching_border = (bx <= x0 + 2) or (by <= y0 + 2) or (bx + bw >= x1 - 2) or (by + bh >= y1 - 2)
        if touching_border and not allow_border:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if solidity < 0.60:
            continue

        pts = contour.reshape(-1, 2).astype(float)
        if len(pts) < 5:
            continue
        _, vecs = np.linalg.eigh(np.cov(pts.T))
        axis = vecs[:, -1]
        angle_deg = wrap_angle_90(np.degrees(np.arctan2(axis[1], axis[0])))

        aspect_score = np.exp(-abs(aspect - 4.5) / 3.5)
        score = float(area * solidity * (0.5 + 0.5 * aspect_score))
        candidates.append(
            {
                "center": [float(cx), float(cy)],
                "angle_deg": float(angle_deg),
                "length_px": length_px,
                "width_px": width_px,
                "aspect_ratio": float(aspect),
                "solidity": float(solidity),
                "area": area,
                "touching_border": bool(touching_border),
                "score": score,
                "box": cv2.boxPoints(cv2.minAreaRect(contour)).tolist(),
            }
        )
    return candidates


def frame_numeric_summary(frame_bgr: np.ndarray) -> dict:
    """Purely numeric summary of a camera frame for terminal diagnostics (never saves/views images)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    return {
        "shape": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
        "mean_brightness": round(float(gray.mean()), 1),
        "std_brightness": round(float(gray.std()), 1),
        "dark_px_pct": round(100.0 * float((gray < 105).mean()), 1),
        "edge_pct": round(100.0 * float((edges > 0).mean()), 2),
    }


def detect_puff_multicue(
    frame_bgr: np.ndarray,
    expected_center: tuple[float, float] | None = None,
    allow_border: bool = False,
    custom_detector: Callable[[np.ndarray], dict | None] | None = None,
) -> dict | None:
    """Detect a puff (dark, colored, metallic, or light-on-light) in the wrist camera view."""
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None

    if custom_detector is not None:
        custom = custom_detector(frame_bgr)
        if custom is not None and "center" in custom:
            c = custom["center"]
            return {
                "center": [float(c[0]), float(c[1])],
                "angle_deg": wrap_angle_90(custom.get("angle_deg", 0.0)),
                "length_px": float(custom.get("length_px", 150.0)),
                "width_px": float(custom.get("width_px", 30.0)),
                "confidence": float(custom.get("confidence", 1.0)),
                "touching_border": bool(custom.get("touching_border", False)),
                "source": "custom_detector",
            }

    pool: list[dict] = []

    # Cue 0: Exact legacy detector proven in rehearsal 20261114_231754
    legacy = _legacy_detect_vape(frame_bgr)
    if legacy is not None:
        legacy_cand = dict(legacy)
        legacy_cand["aspect_ratio"] = legacy["length_px"] / max(legacy["width_px"], 1.0)
        legacy_cand["solidity"] = 0.90
        legacy_cand["touching_border"] = False
        legacy_cand["score"] = float(legacy["area"] * 1.15)
        legacy_cand["source"] = "legacy_dark"
        pool.append(legacy_cand)

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(0.03 * w), int(0.98 * w)
    y0, y1 = int(0.03 * h), int(0.85 * h)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 255
    roi_bounds = (x0, y0, x1, y1)

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Cue 1: Dark object mask (slightly more permissive luminance up to 115)
    dark_mask = cv2.bitwise_and(cv2.inRange(gray, 0, 115), roi_mask)

    # Estimate the white-paper chroma robustly from the central work area. The
    # median tolerates the puff itself as long as it occupies less than half
    # that patch. A chroma-only mask avoids turning paper shadows into a puff.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx0, cx1 = int(0.20 * w), int(0.80 * w)
    cy0, cy1 = int(0.18 * h), int(0.75 * h)
    bg_lab = np.median(lab[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
    delta_e = np.linalg.norm(lab - bg_lab, axis=2)
    center_de = delta_e[cy0:cy1, cx0:cx1]
    bg_mad = float(np.median(np.abs(center_de - np.median(center_de))))
    thresh_de = max(20.0, 4.2 * bg_mad)
    lab_mask = cv2.bitwise_and((delta_e >= thresh_de).astype(np.uint8) * 255, roi_mask)

    ab_delta = np.linalg.norm(lab[:, :, 1:3] - bg_lab[1:3], axis=2)
    center_ab_delta = ab_delta[cy0:cy1, cx0:cx1]
    chroma_median = float(np.median(center_ab_delta))
    chroma_mad = float(np.median(np.abs(center_ab_delta - chroma_median)))
    # OpenCV's 8-bit Lab a/b channels put neutral white near (128, 128).
    # Estimate noise from the paper instead of assuming a particular puff hue.
    chroma_threshold = max(8.0, chroma_median + 5.0 * 1.4826 * chroma_mad)
    paper_chroma_mask = cv2.bitwise_and(
        (ab_delta >= chroma_threshold).astype(np.uint8) * 255, roi_mask
    )

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    bg_sat = float(np.median(hsv[cy0:cy1, cx0:cx1, 1]))
    sat_mask = cv2.bitwise_and((hsv[:, :, 1] >= max(50.0, bg_sat + 32.0)).astype(np.uint8) * 255, roi_mask)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(lab_mask, sat_mask), paper_chroma_mask)

    # Cue 3: Local adaptive contrast / Canny closed silhouette (light/silver puff on white paper)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.bitwise_and(edges, roi_mask)
    edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    paper_candidates = _extract_candidates_from_mask(
        paper_chroma_mask,
        roi_bounds,
        (h, w),
        allow_border=allow_border,
        # The puff may be seen at an oblique angle or only as a colored logo
        # pattern. Keep a clear elongated-shape test, but allow a shorter and
        # thinner chromatic silhouette than the generic dark-object detector.
        min_area=180.0,
        min_width=5.0,
        min_aspect=1.6,
        min_length_fraction=0.07,
    )
    for cand in paper_candidates:
        cand["source"] = "white_paper_chroma"
        pool.append(cand)

    for source_name, m in (
        ("lab_color", color_mask),
        ("dark_gray", dark_mask),
        ("edge_shape", edge_closed),
    ):
        for cand in _extract_candidates_from_mask(m, roi_bounds, (h, w), allow_border=allow_border):
            cand["source"] = source_name
            pool.append(cand)

    if not pool:
        return None

    # When the central work area is neutral white paper, trust chromatic object
    # silhouettes over brightness edges: paper shadows can otherwise form one
    # large elongated contour and outrank the puff by area.
    neutral_paper = float(np.linalg.norm(bg_lab[1:3] - np.array([128.0, 128.0]))) <= 22.0
    if neutral_paper and paper_candidates:
        pool = paper_candidates

    pool.sort(key=lambda c: c["score"], reverse=True)
    unique: list[dict] = []
    for cand in pool:
        if all(np.hypot(cand["center"][0] - u["center"][0], cand["center"][1] - u["center"][1]) > 22.0 for u in unique):
            unique.append(cand)

    if expected_center is not None:
        ex, ey = float(expected_center[0]), float(expected_center[1])
        unique.sort(
            key=lambda c: c["score"] / (1.0 + 0.008 * np.hypot(c["center"][0] - ex, c["center"][1] - ey)),
            reverse=True,
        )

    best = unique[0]
    best["confidence"] = float(np.clip(best["solidity"], 0.0, 1.0))
    return best


class WristCameraStream:
    """Lightweight UVC wrist camera reader with buffer flushing (no video analysis)."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = int(camera_index)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Caméra poignet (index {self.camera_index}) indisponible. "
                "Vérifier le branchement USB ou tester avec check_wrist_camera.py / --no-camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Warm up auto-exposure on USB UVC sensors (first frames after open are often dark/overexposed!)
        frame = None
        for _ in range(10):
            ok, img = self.cap.read()
            if ok and img is not None:
                frame = img
            time.sleep(0.025)
        if frame is None:
            self.cap.release()
            raise RuntimeError(f"Caméra {self.camera_index} ouverte mais aucune image reçue.")
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])

    def read_fresh_frames(self, count: int = 3, flush: int = 5) -> list[np.ndarray]:
        """Flush buffered UVC frames after arm motion stops, then read `count` crisp frames."""
        for _ in range(flush):
            self.cap.grab()
            time.sleep(0.015)
        frames: list[np.ndarray] = []
        deadline = time.monotonic() + 1.5
        while len(frames) < count and time.monotonic() < deadline:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                frames.append(frame)
                if len(frames) < count:
                    time.sleep(0.035)
            else:
                time.sleep(0.02)
        if len(frames) < count:
            raise RuntimeError("Lecture caméra poignet incomplète ou figée.")
        return frames

    def detect_stable(
        self,
        expected_center: tuple[float, float] | None = None,
        allow_border: bool = False,
        custom_detector: Callable[[np.ndarray], dict | None] | None = None,
        samples: int = 3,
    ) -> dict | None:
        frames = self.read_fresh_frames(count=samples, flush=5)
        detections = [
            detect_puff_multicue(
                f,
                expected_center=expected_center,
                allow_border=allow_border,
                custom_detector=custom_detector,
            )
            for f in frames
        ]
        valid = [d for d in detections if d is not None]
        if len(valid) < max(2, samples - 1):
            return None
        centers = np.array([d["center"] for d in valid], dtype=float)
        spread_px = float(np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
        if spread_px > 16.0:
            return None

        lengths = np.array([float(d.get("length_px", 0.0)) for d in valid])
        widths = np.array([float(d.get("width_px", 0.0)) for d in valid])
        if np.any(lengths <= 0.0) or np.any(widths <= 0.0):
            return None
        # Reject a sequence that jumps between a puff/logo and a much larger
        # shadow or paper edge, even when their image centers happen to agree.
        for dimensions in (lengths, widths):
            median = float(np.median(dimensions))
            if median <= 0.0 or float(np.max(np.abs(dimensions - median)) / median) > 0.35:
                return None
        sources = [d.get("source", "multicue") for d in valid]
        if max(sources.count(source) for source in set(sources)) < len(valid) - 1:
            return None

        rad2 = np.deg2rad([2.0 * d["angle_deg"] for d in valid])
        mean_angle = 0.5 * float(np.rad2deg(np.arctan2(np.mean(np.sin(rad2)), np.mean(np.cos(rad2)))))
        return {
            "center": np.median(centers, axis=0).tolist(),
            "angle_deg": wrap_angle_90(mean_angle),
            "length_px": float(np.median(lengths)),
            "width_px": float(np.median(widths)),
            "confidence": float(np.median([d.get("confidence", 0.0) for d in valid])),
            "spread_px": spread_px,
            "touching_border": any(d.get("touching_border", False) for d in valid),
            "source": valid[0].get("source", "multicue"),
        }

    def last_frame_diagnostics(self) -> dict:
        frames = self.read_fresh_frames(count=1, flush=2)
        return frame_numeric_summary(frames[0])

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


@dataclass
class WristServoCalibration:
    hover_z_mm: float
    u_ref_px: float
    v_ref_px: float
    angle_ref_deg: float
    j_px_per_robot_mm: np.ndarray     # 2x2: d[u, v] / d[X_rob_mm, Y_rob_mm]
    angle_gain_per_yaw_deg: float     # d(angle_img_deg) / d(yaw_rob_deg)
    calibration_sha256: str
    camera_index: int = 0
    pose_mode: str = "observe_pose"
    source: str = "empirical"

    @classmethod
    def theoretical_default(cls, hover_z_mm: float = DEFAULT_HOVER_Z_MM, flip_180: bool = False) -> "WristServoCalibration":
        """Analytical pinhole fallback derived from so101.xml at the validated observation pose."""
        ik = SO101CalibratedIK()
        fy = 240.0 / np.tan(np.deg2rad(30.5))

        def project_origin(x_rob: float, y_rob: float) -> np.ndarray:
            q = ik.solve_observe(x_rob, y_rob).state_deg
            Tc = fk_camera_frame(q)
            p_obj = np.linalg.inv(Tc) @ np.r_[ik.origin_tcp_mm / 1000.0, 1.0]
            u = 320.0 + fy * (p_obj[0] / (-p_obj[2]))
            v = 240.0 - fy * (p_obj[1] / (-p_obj[2]))
            if flip_180:
                u, v = 640.0 - u, 480.0 - v
            return np.array([u, v], dtype=float)

        p0 = project_origin(0.0, 0.0)
        px = project_origin(10.0, 0.0)
        py = project_origin(0.0, 10.0)
        J = np.column_stack([(px - p0) / 10.0, (py - p0) / 10.0])
        return cls(
            hover_z_mm=float(hover_z_mm),
            u_ref_px=float(p0[0]),
            v_ref_px=float(p0[1]),
            angle_ref_deg=0.0,
            j_px_per_robot_mm=J,
            angle_gain_per_yaw_deg=-1.0,
            calibration_sha256=EXPECTED_CALIBRATION_SHA256,
            camera_index=0,
            pose_mode="observe_pose",
            source="theoretical_so101_xml",
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_SERVO_CALIB_PATH) -> "WristServoCalibration":
        raw = json.loads(Path(path).read_text())
        if raw.get("calibration_sha256") != EXPECTED_CALIBRATION_SHA256:
            raise RuntimeError("Calibration caméra-poignet périmée (calibration moteurs changée).")
        J = np.asarray(raw["j_px_per_robot_mm"], dtype=float)
        if J.shape != (2, 2) or not np.isfinite(J).all() or abs(np.linalg.det(J)) < 0.2:
            raise RuntimeError("Matrice Jacobienne caméra invalide ou mal conditionnée.")
        return cls(
            hover_z_mm=float(raw["hover_z_mm"]),
            u_ref_px=float(raw["u_ref_px"]),
            v_ref_px=float(raw["v_ref_px"]),
            angle_ref_deg=float(raw["angle_ref_deg"]),
            j_px_per_robot_mm=J,
            angle_gain_per_yaw_deg=float(raw.get("angle_gain_per_yaw_deg", -1.0)),
            calibration_sha256=str(raw["calibration_sha256"]),
            camera_index=int(raw.get("camera_index", 0)),
            pose_mode=str(raw.get("pose_mode", "observe_pose")),
            source=str(raw.get("source", "empirical")),
        )

    def save(self, path: Path = DEFAULT_SERVO_CALIB_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "wrist_servo_calibration_v2",
            "hover_z_mm": self.hover_z_mm,
            "u_ref_px": self.u_ref_px,
            "v_ref_px": self.v_ref_px,
            "angle_ref_deg": self.angle_ref_deg,
            "j_px_per_robot_mm": self.j_px_per_robot_mm.tolist(),
            "angle_gain_per_yaw_deg": self.angle_gain_per_yaw_deg,
            "det_j": float(np.linalg.det(self.j_px_per_robot_mm)),
            "camera_index": self.camera_index,
            "pose_mode": self.pose_mode,
            "calibration_sha256": self.calibration_sha256,
            "source": self.source,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    def compute_correction_mm(
        self,
        observed_uv: tuple[float, float],
        current_yaw_deg: float = 0.0,
        gain: float = 0.80,
        max_step_mm: float = 28.0,
    ) -> tuple[np.ndarray, float]:
        """Return (delta_xy_robot_mm, estimated_residual_mm) to center puff on (u_ref, v_ref)."""
        du_desired = self.u_ref_px - float(observed_uv[0])
        dv_desired = self.v_ref_px - float(observed_uv[1])
        delta_local = np.linalg.solve(self.j_px_per_robot_mm, np.array([du_desired, dv_desired], dtype=float))

        if abs(current_yaw_deg) > 1e-3:
            rad = np.deg2rad(current_yaw_deg)
            c, s = float(np.cos(rad)), float(np.sin(rad))
            R_yaw = np.array([[c, -s], [s, c]], dtype=float)
            delta_local = R_yaw @ delta_local

        err_mm = float(np.linalg.norm(delta_local))
        step = delta_local * float(gain)
        norm = float(np.linalg.norm(step))
        if norm > max_step_mm:
            step = step * (max_step_mm / norm)
        return step, err_mm

    def compute_yaw_correction_deg(
        self,
        observed_angle_deg: float,
        gain: float = 0.85,
        max_step_deg: float = 35.0,
    ) -> tuple[float, float]:
        """Return (delta_yaw_table_deg, angle_error_deg) via J^{-1} de-foreshortening."""
        a0 = np.deg2rad(self.angle_ref_deg)
        ai = np.deg2rad(float(observed_angle_deg))
        v0 = np.linalg.solve(self.j_px_per_robot_mm, np.array([np.cos(a0), np.sin(a0)], dtype=float))
        vi = np.linalg.solve(self.j_px_per_robot_mm, np.array([np.cos(ai), np.sin(ai)], dtype=float))
        th0 = float(np.degrees(np.arctan2(v0[1], v0[0])))
        thi = float(np.degrees(np.arctan2(vi[1], vi[0])))
        raw_delta = wrap_angle_90(thi - th0)
        return float(np.clip(gain * raw_delta, -max_step_deg, +max_step_deg)), abs(raw_delta)
