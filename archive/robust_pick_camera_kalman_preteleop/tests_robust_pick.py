"""Offline end-to-end verification suite for the Robust Puff Pick system.

Runnable directly with `.venv/bin/python tests_robust_pick.py` (stdlib unittest).
Never opens serial ports, never moves motors, never reads recorded videos.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from homography_table import TableHomography
from pick_puff import default_arguments
from puff_vision import (
    WristServoCalibration,
    WristCameraStream,
    detect_puff_multicue,
    load_custom_detector,
    wrap_angle_90,
)
from robust_pick import (
    RobustPicker,
    check_start_pose_safe,
    execute_robust_pick_cycle,
    run_camera_calibration,
    visual_acquire_and_center,
)
from so101_ik import (
    ORIGIN_GRASP_STATE_5,
    ROOT,
    SAFE_JOINT_HIGH_DEG,
    SAFE_JOINT_LOW_DEG,
    SO101CalibratedIK,
    fk_camera_frame,
    fk_tcp_mm,
    verify_calibration_hash,
)


def _render_synthetic_puff_frame(
    center_uv: tuple[float, float],
    angle_deg: float = 12.0,
    puff_bgr: tuple[int, int, int] = (30, 30, 35),
    bg_bgr: tuple[int, int, int] = (215, 212, 208),
    add_clutter: bool = True,
) -> np.ndarray:
    frame = np.full((480, 640, 3), bg_bgr, dtype=np.uint8)
    if add_clutter:
        cv2.circle(frame, (110, 120), 26, (60, 60, 65), -1)
        cv2.rectangle(frame, (220, 420), (430, 479), (25, 195, 245), -1)
    rect = ((float(center_uv[0]), float(center_uv[1])), (165.0, 32.0), float(angle_deg))
    box = np.int32(cv2.boxPoints(rect))
    cv2.fillConvexPoly(frame, box, puff_bgr)
    return frame


class _SimulatedHardwareWorld:
    """Simulates SO-101 kinematics + pinhole wrist camera + proprioceptive gripper physics."""

    def __init__(
        self,
        ik: SO101CalibratedIK,
        puff_xy_mm: tuple[float, float],
        puff_yaw_deg: float,
        flip_cam_180: bool = False,
        fail_first_grasp: bool = False,
    ):
        self.ik = ik
        self.puff_xy_mm = np.array(puff_xy_mm, dtype=float)
        self.puff_yaw_deg = float(puff_yaw_deg)
        self.flip_cam_180 = flip_cam_180
        self.fail_first_grasp = fail_first_grasp
        self.slips_done = 0
        self.currently_slipped = False
        self.last_cmd_6 = np.array([-8.75, -102.55, 93.27, 77.19, -94.46, 0.55], dtype=float)
        self.fy = 240.0 / np.tan(np.deg2rad(30.5))

    def move_joint_smooth(
        self, target_cmd_5, gripper_cmd, speed_deg_per_s=42.0, min_seconds=0.1, label="", check_stall=True
    ):
        if gripper_cmd > 10.0:
            self.currently_slipped = False
        self.last_cmd_6 = np.r_[np.asarray(target_cmd_5[:5], dtype=float), float(gripper_cmd)]
        return self.read_measured()

    def move_cartesian_vertical(
        self, ik, x_mm, y_mm, z_start_mm, z_end_mm, yaw_delta_deg, lock_roll_deg, gripper_cmd, duration=0.1, label=""
    ):
        sol = ik.solve(x_mm, y_mm, z_end_mm, yaw_delta_deg=yaw_delta_deg, lock_wrist_roll_deg=lock_roll_deg)
        self.last_cmd_6 = np.r_[sol.command_deg, float(gripper_cmd)]
        return self.read_measured()

    def hold_seconds(self, seconds, gripper_cmd=None):
        if gripper_cmd is not None:
            self.last_cmd_6[5] = float(gripper_cmd)
        return self.read_measured()

    def read_measured(self) -> np.ndarray:
        meas = self.last_cmd_6.copy()
        tcp_mm, _ = fk_tcp_mm(meas[:5])
        rob_xyz = self.ik.base_to_table_mm(tcp_mm)
        dist_xy = float(np.linalg.norm(rob_xyz[:2] - self.puff_xy_mm))
        if meas[5] < 4.5:
            if dist_xy <= 14.0:
                if rob_xyz[2] > 20.0 and self.fail_first_grasp and self.slips_done == 0:
                    self.slips_done += 1
                    self.currently_slipped = True
                meas[5] = 3.63 if self.currently_slipped else 6.84
            else:
                meas[5] = 3.63      # Closed on empty air
        return meas

    def detect_stable(self, expected_center=None, allow_border=False, custom_detector=None, samples=3):
        Tc = fk_camera_frame(self.last_cmd_6[:5])
        T_inv = np.linalg.inv(Tc)
        ang_rad = np.deg2rad(self.puff_yaw_deg)
        dir_xy = np.array([np.cos(ang_rad), np.sin(ang_rad)], dtype=float)
        pts_uv = []
        for s in (-42.0, 0.0, +42.0):
            pt_tb = np.r_[self.puff_xy_mm + s * dir_xy, 0.0]
            p_base = self.ik.table_to_base_mm(pt_tb[0], pt_tb[1], 0.0) / 1000.0
            p_cam = T_inv @ np.r_[p_base, 1.0]
            if -p_cam[2] <= 0.02:
                return None
            u = 320.0 + self.fy * (p_cam[0] / (-p_cam[2]))
            v = 240.0 - self.fy * (p_cam[1] / (-p_cam[2]))
            if self.flip_cam_180:
                u, v = 640.0 - u, 480.0 - v
            pts_uv.append((u, v))
        cu, cv_px = pts_uv[1]
        if not (25.0 <= cu <= 615.0 and 20.0 <= cv_px <= 390.0):
            return None
        img_ang = wrap_angle_90(np.degrees(np.arctan2(pts_uv[2][1] - pts_uv[0][1], pts_uv[2][0] - pts_uv[0][0])))
        frame = _render_synthetic_puff_frame((cu, cv_px), angle_deg=img_ang, puff_bgr=(40, 90, 230))
        return detect_puff_multicue(frame, expected_center=expected_center, allow_border=allow_border)


class RobustPickTests(unittest.TestCase):
    def test_pick_puff_entrypoint_uses_calibrated_single_attempt_defaults(self):
        self.assertEqual(
            default_arguments(["--hardware"]),
            ["--x-mm", "0", "--y-mm", "0", "--max-retries", "0", "--hardware"],
        )
        self.assertEqual(
            default_arguments(["--x-mm", "5", "--hardware"]),
            ["--y-mm", "0", "--max-retries", "0", "--x-mm", "5", "--hardware"],
        )

    def test_calibration_hash_matches_real_follower(self):
        self.assertEqual(len(verify_calibration_hash()), 64)

    def test_ik_exact_identity_at_validated_origin(self):
        ik = SO101CalibratedIK()
        res = ik.solve(0.0, 0.0, 0.0, yaw_delta_deg=0.0)
        self.assertTrue(np.allclose(res.state_deg, ORIGIN_GRASP_STATE_5, atol=1e-2))
        self.assertLess(res.pos_error_mm, 0.01)
        self.assertFalse(res.yaw_clamped)

    def test_fk_matches_all_11_real_table_captures(self):
        table_data = json.loads((ROOT / "real_transfer/table_20261115_010604.json").read_text())
        ik = SO101CalibratedIK()
        names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        ]
        fk_points, targets = [], []
        for pt in table_data["points"]:
            q = np.array([pt["positions"][n] for n in names], dtype=float)
            tcp_mm, _ = fk_tcp_mm(q)
            fk_points.append(ik.base_to_table_mm(tcp_mm)[:2])
            targets.append(pt["xy_mm"])
        fk_arr = np.asarray(fk_points)
        tgt_arr = np.asarray(targets, dtype=float)
        centered_fk = fk_arr - fk_arr.mean(axis=0) + tgt_arr.mean(axis=0)
        errors_mm = np.linalg.norm(centered_fk - tgt_arr, axis=1)
        self.assertLess(float(np.mean(errors_mm)), 3.5)
        self.assertLess(float(np.max(errors_mm)), 6.0)

    def test_ik_workspace_grid_precision_and_joint_limits(self):
        ik = SO101CalibratedIK()
        for x in (-85.0, -40.0, 0.0, 5.0, 40.0, 85.0):
            for y in (-45.0, -15.0, 0.0, 25.0, 75.0):
                for z in (0.0, 80.0):
                    sol = ik.solve(x, y, z)
                    self.assertLess(sol.pos_error_mm, 0.15)
                    self.assertTrue(np.all(sol.state_deg >= SAFE_JOINT_LOW_DEG - 1e-6))
                    self.assertTrue(np.all(sol.state_deg <= SAFE_JOINT_HIGH_DEG + 1e-6))

    def test_ik_rejects_out_of_bounds_and_nan(self):
        ik = SO101CalibratedIK()
        with self.assertRaises(ValueError):
            ik.solve(180.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            ik.solve(0.0, -120.0, 0.0)
        with self.assertRaises(ValueError):
            ik.solve(float("nan"), 0.0, 0.0)

    def test_multicue_detector_handles_dark_and_colored_puffs_with_clutter(self):
        for puff_bgr, true_angle in [((25, 25, 28), 0.0), ((165, 55, 235), -28.0), ((230, 175, 40), +34.0)]:
            frame = _render_synthetic_puff_frame((345.0, 195.0), angle_deg=true_angle, puff_bgr=puff_bgr)
            det = detect_puff_multicue(frame, expected_center=(360.0, 180.0))
            self.assertIsNotNone(det)
            assert det is not None
            self.assertLess(np.hypot(det["center"][0] - 345.0, det["center"][1] - 195.0), 3.0)
            self.assertLess(abs(wrap_angle_90(det["angle_deg"] - true_angle)), 3.0)

    def test_white_paper_chroma_detector_handles_color_cast_shadow_and_clutter(self):
        h, w = 480, 640
        y = np.linspace(0.92, 1.0, h, dtype=np.float32)[:, None, None]
        paper = np.clip(np.full((h, w, 3), 255, dtype=np.float32) * y, 0, 255).astype(np.uint8)
        # Mild warm color cast and a broad soft shadow on the white paper.
        paper[:, :, 2] = np.clip(paper[:, :, 2].astype(np.int16) - 5, 0, 255).astype(np.uint8)
        shadow = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(shadow, (170, 190), (105, 60), -15, 0, 360, 45, -1)
        paper = cv2.subtract(paper, cv2.merge([shadow, shadow, shadow]))
        cv2.circle(paper, (105, 105), 24, (20, 100, 220), -1)  # colored distractor
        cv2.rectangle(paper, (470, 355), (600, 390), (40, 190, 210), -1)
        target_center = (347.0, 236.0)
        rect = (target_center, (172.0, 31.0), 21.0)
        cv2.fillConvexPoly(paper, np.int32(cv2.boxPoints(rect)), (70, 205, 100))

        det = detect_puff_multicue(paper, expected_center=(322.0, 265.0))
        self.assertIsNotNone(det)
        assert det is not None
        self.assertLess(np.hypot(det["center"][0] - target_center[0], det["center"][1] - target_center[1]), 4.0)
        self.assertLess(abs(wrap_angle_90(det["angle_deg"] - 21.0)), 4.0)

    def test_white_paper_chroma_accepts_shorter_elongated_puff_pattern(self):
        frame = np.full((480, 640, 3), (242, 242, 242), dtype=np.uint8)
        center = (318.0, 238.0)
        angle = 27.0
        # A compact visible pattern, shorter than the old generic 12%-of-frame
        # threshold, while still distinctly elongated and chromatic.
        box = np.int32(cv2.boxPoints((center, (62.0, 15.0), angle)))
        cv2.fillConvexPoly(frame, box, (70, 205, 100))
        det = detect_puff_multicue(frame)
        self.assertIsNotNone(det)
        assert det is not None
        self.assertEqual(det["source"], "white_paper_chroma")
        self.assertLess(np.hypot(det["center"][0] - center[0], det["center"][1] - center[1]), 4.0)
        self.assertLess(abs(wrap_angle_90(det["angle_deg"] - angle)), 4.0)

    def test_stable_camera_rejects_candidate_that_changes_size(self):
        camera = WristCameraStream.__new__(WristCameraStream)
        camera.read_fresh_frames = lambda count=3, flush=5: [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(count)]
        detections = iter([
            {"center": [320.0, 240.0], "angle_deg": 10.0, "length_px": 160.0, "width_px": 30.0,
             "confidence": 0.9, "source": "white_paper_chroma"},
            {"center": [321.0, 240.0], "angle_deg": 10.0, "length_px": 310.0, "width_px": 58.0,
             "confidence": 0.9, "source": "legacy_dark"},
            {"center": [320.0, 241.0], "angle_deg": 11.0, "length_px": 162.0, "width_px": 31.0,
             "confidence": 0.9, "source": "white_paper_chroma"},
        ])
        from unittest.mock import patch
        with patch("puff_vision.detect_puff_multicue", side_effect=lambda *args, **kwargs: next(detections)):
            self.assertIsNone(camera.detect_stable(samples=3))

    def test_uncertain_camera_view_scans_small_table_y_offset_at_hover(self):
        ik = SO101CalibratedIK()
        picker = RobustPicker(ik=ik)

        class Runner:
            def __init__(self):
                self.labels = []

            def move_joint_smooth(self, target_cmd_5, gripper_cmd, **kwargs):
                self.labels.append(kwargs.get("label", ""))

            def hold_seconds(self, seconds):
                return None

        class Camera:
            def __init__(self):
                self.views = [
                    {"center": [473.0, 377.0], "angle_deg": 0.0, "length_px": 90.0,
                     "width_px": 45.0, "confidence": 0.45, "spread_px": 3.0,
                     "touching_border": True, "source": "white_paper_chroma"},
                    {"center": [320.0, 240.0], "angle_deg": 0.0, "length_px": 170.0,
                     "width_px": 32.0, "confidence": 0.95, "spread_px": 1.0,
                     "touching_border": False, "source": "white_paper_chroma"},
                    {"center": [320.0, 240.0], "angle_deg": 0.0, "length_px": 170.0,
                     "width_px": 32.0, "confidence": 0.95, "spread_px": 1.0,
                     "touching_border": False, "source": "white_paper_chroma"},
                ]

            def detect_stable(self, **kwargs):
                return self.views.pop(0)

        calibration = WristServoCalibration(
            hover_z_mm=80.0, u_ref_px=320.0, v_ref_px=240.0, angle_ref_deg=0.0,
            j_px_per_robot_mm=np.eye(2), angle_gain_per_yaw_deg=-1.0,
            calibration_sha256="synthetic",
        )
        runner, camera, log = Runner(), Camera(), []
        x, y, _, locked = visual_acquire_and_center(
            runner, picker, camera, calibration, 0.0, 0.0, None, log
        )
        self.assertTrue(locked)
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 8.0)
        self.assertTrue(any("Balayage caméra Y" in label for label in runner.labels))
        self.assertEqual(log[0]["stage"], "camera_y_scan")

    def test_custom_detector_plugin_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = Path(tmp) / "my_ds_detector.py"
            plugin.write_text(
                "def detect(frame):\n"
                "    return {'center': [312.5, 204.0], 'angle_deg': -17.5, 'confidence': 0.97}\n"
            )
            fn = load_custom_detector(plugin)
            self.assertIsNotNone(fn)
            det = detect_puff_multicue(np.zeros((480, 640, 3), dtype=np.uint8), custom_detector=fn)
            assert det is not None
            self.assertEqual(det["source"], "custom_detector")
            self.assertEqual(det["center"], [312.5, 204.0])

    def test_end_to_end_servo_0deg_and_180deg_with_slip_retry_and_out_of_fov_search(self):
        ik = SO101CalibratedIK()
        picker = RobustPicker(ik=ik)
        for flip_180 in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                calib_file = Path(tmp) / "servo_cal.json"
                world_cal = _SimulatedHardwareWorld(ik, puff_xy_mm=(0.0, 0.0), puff_yaw_deg=0.0, flip_cam_180=flip_180)
                run_camera_calibration(world_cal, picker, world_cal, None, calib_file)
                loaded = WristServoCalibration.load(calib_file)

                # Case A: 52 mm input error + 22 deg rotated puff + slip on attempt 1!
                world_pick = _SimulatedHardwareWorld(
                    ik, puff_xy_mm=(28.0, -18.0), puff_yaw_deg=22.0, flip_cam_180=flip_180, fail_first_grasp=True
                )
                cycle_log = {"requested_xy_mm": [75.0, 5.0], "attempts": []}
                ok = execute_robust_pick_cycle(
                    runner=world_pick,
                    picker=picker,
                    x_req_mm=75.0,
                    y_req_mm=5.0,
                    cam=world_pick,
                    calib=loaded,
                    custom_detector=None,
                    drop_xy_mm=(-50.0, 10.0),
                    max_retries=2,
                    cycle_log=cycle_log,
                )
                self.assertTrue(ok)
                self.assertEqual(len(cycle_log["attempts"]), 2)
                final_xy = cycle_log["attempts"][-1]["grasp_xy_mm"]
                final_yaw = cycle_log["attempts"][-1]["grasp_yaw_deg"]
                self.assertLess(np.hypot(final_xy[0] - 28.0, final_xy[1] - (-18.0)), 3.5)
                self.assertLess(abs(final_yaw - 22.0), 6.0)

                # Case B: Blind mode (--no-camera) with 11 mm error -> caught on try 1 via wide funnel,
                #         and with 19 mm error -> missed try 1 (3.63 pts) -> caught try 2 via +8mm tactile step!
                world_blind = _SimulatedHardwareWorld(ik, puff_xy_mm=(19.0, 0.0), puff_yaw_deg=0.0)
                blind_log = {"requested_xy_mm": [0.0, 0.0], "attempts": []}
                ok_blind = execute_robust_pick_cycle(
                    runner=world_blind,
                    picker=picker,
                    x_req_mm=0.0,
                    y_req_mm=0.0,
                    cam=None,
                    calib=None,
                    custom_detector=None,
                    drop_xy_mm=None,
                    max_retries=2,
                    cycle_log=blind_log,
                )
                self.assertTrue(ok_blind)
                self.assertEqual(len(blind_log["attempts"]), 2)

    def test_planar_homography_accuracy_under_perspective(self):
        H_true = np.array([[0.35, -0.04, -95.0], [0.02, -0.32, 110.0], [0.0001, 0.0002, 1.0]])
        uvs = [[200, 150], [450, 155], [460, 360], [190, 350], [325, 255], [280, 210]]
        xys = []
        for u, v in uvs:
            p = H_true @ [u, v, 1.0]
            xys.append([p[0] / p[2], p[1] / p[2]])
        mapper = TableHomography.fit(uvs, xys)
        self.assertIsNotNone(mapper.loo_max_error_mm)
        assert mapper.loo_max_error_mm is not None
        self.assertLess(mapper.loo_max_error_mm, 0.01)
        px, py = mapper.pixel_to_table_mm(310.0, 240.0)
        gt = H_true @ [310.0, 240.0, 1.0]
        self.assertLess(np.hypot(px - gt[0] / gt[2], py - gt[1] / gt[2]), 0.01)

    def test_start_pose_accepts_current_midair_held_pose(self):
        ik = SO101CalibratedIK()
        held_pose_030321 = np.array([-12.352, -8.132, 13.275, 83.341, -165.055, 9.644], dtype=float)
        self.assertTrue(check_start_pose_safe(held_pose_030321, ik, hover_z_mm=80.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
