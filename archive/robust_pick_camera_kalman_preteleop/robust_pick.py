"""Robust (X, Y) Puff Pick & Sort Controller for SO-101 ('mon_bras').

Combines calibrated 3D SO-101 inverse kinematics, wrist-camera centering with a
table-frame Kalman estimate, a brief bounded occlusion bridge, and gripper-position
feedback. A pick is allowed only after a fresh visual centering measurement. The
software does not yet certify physical grasp robustness or compensate for jaw geometry.

Safety: never moves without operator 'ATTRAPER'/'CALIBRER' confirmation in Terminal;
Ctrl+C holds current arm pose and preserves gripper squeeze torque.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from hardware_guard import add_hardware_flag, require_hardware

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from puff_vision import (
    DEFAULT_HOVER_Z_MM,
    DEFAULT_SERVO_CALIB_PATH,
    TableTargetKalman,
    WristCameraStream,
    WristServoCalibration,
    load_custom_detector,
)
from so101_ik import (
    CALIBRATION_PATH,
    JOINT_NAMES,
    ORIGIN_REST_ACTION,
    ORIGIN_REST_STATE,
    ROOT,
    SO101CalibratedIK,
    WORKSPACE_X_MM,
    WORKSPACE_Y_MM,
    verify_calibration_hash,
)

FOLLOWER_PORT = "/dev/cu.usbmodem5B610354211"
FOLLOWER_ID = "mon_bras"


def smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def hold_and_disconnect(bus) -> None:
    if not bus.is_connected:
        return
    try:
        actual = bus.sync_read("Present_Position", num_retry=2)
        actual["gripper"] = bus.read("Goal_Position", "gripper", num_retry=2)
        bus.sync_write("Goal_Position", actual, num_retry=2)
        print("ARRÊT SÉCURISÉ : bras maintenu sous couple, consigne de pince conservée.", flush=True)
    except Exception as error:
        print(f"Maintien non confirmé ({error}). Soutenir le bras avant toute coupure !", flush=True)
    finally:
        bus.disconnect(disable_torque=False)


class RobustPicker:
    """High-level safe motion & closed-loop grasp executor for SO-101."""

    def __init__(
        self,
        ik: SO101CalibratedIK | None = None,
        hover_z_mm: float = DEFAULT_HOVER_Z_MM,
        z_offset_mm: float = 0.0,
        gripper_open: float = 36.0,
        gripper_close: float = 3.2,
        min_grasp_pos: float = 4.6,
        speed_deg_per_s: float = 42.0,
        align_angle: bool = True,
    ):
        self.ik = ik or SO101CalibratedIK()
        self.hover_z_mm = float(hover_z_mm)
        self.z_offset_mm = float(z_offset_mm)
        self.gripper_open = float(np.clip(gripper_open, 5.0, 65.0))
        self.gripper_close = float(np.clip(gripper_close, 1.5, 8.0))
        self.min_grasp_pos = float(min_grasp_pos)
        self.speed_deg_per_s = float(np.clip(speed_deg_per_s, 15.0, 75.0))
        self.align_angle = bool(align_angle)

    def plan_blind_sequence(
        self,
        x_mm: float,
        y_mm: float,
        yaw_delta_deg: float = 0.0,
        drop_xy_mm: tuple[float, float] | None = None,
    ) -> dict:
        """Validate all IK waypoints offline before touching any motor."""
        hover = self.ik.solve(x_mm, y_mm, self.hover_z_mm, yaw_delta_deg=0.0)
        pre_aligned = self.ik.solve(x_mm, y_mm, self.hover_z_mm, yaw_delta_deg=yaw_delta_deg)
        grasp = self.ik.solve(
            x_mm,
            y_mm,
            self.z_offset_mm,
            yaw_delta_deg=yaw_delta_deg,
            initial_deg=pre_aligned.state_deg,
            lock_wrist_roll_deg=pre_aligned.state_deg[4],
        )
        lift = self.ik.solve(
            x_mm,
            y_mm,
            self.hover_z_mm + 5.0,
            yaw_delta_deg=yaw_delta_deg,
            initial_deg=grasp.state_deg,
            lock_wrist_roll_deg=grasp.state_deg[4],
        )
        plan = {
            "hover": hover,
            "pre_aligned": pre_aligned,
            "grasp": grasp,
            "lift": lift,
        }
        if drop_xy_mm is not None:
            dx, dy = drop_xy_mm
            drop_hover = self.ik.solve(dx, dy, self.hover_z_mm + 5.0, yaw_delta_deg=0.0)
            drop_low = self.ik.solve(dx, dy, max(15.0, self.z_offset_mm + 20.0), yaw_delta_deg=0.0)
            plan["drop_hover"] = drop_hover
            plan["drop_low"] = drop_low
        return plan


class HardwareRunner:
    """Manages 120 Hz smooth interpolation, tracking supervision, and telemetry logging."""

    def __init__(self, robot: SO101Follower, run_log: dict):
        self.robot = robot
        self.run_log = run_log
        self.last_cmd_6: np.ndarray | None = None

    def read_measured(self) -> np.ndarray:
        vals = self.robot.bus.sync_read("Present_Position", num_retry=2)
        arr = np.array([vals[n] for n in JOINT_NAMES], dtype=float)
        if not np.isfinite(arr).all():
            raise RuntimeError("Mesure moteur non finie.")
        return arr

    def seed_from_present(self) -> np.ndarray:
        actual = self.read_measured()
        self.robot.bus.sync_write(
            "Goal_Position",
            {n: float(v) for n, v in zip(JOINT_NAMES, actual)},
            num_retry=2,
        )
        self.last_cmd_6 = actual.copy()
        return actual

    def move_joint_smooth(
        self,
        target_cmd_5: np.ndarray,
        gripper_cmd: float,
        speed_deg_per_s: float = 42.0,
        min_seconds: float = 0.45,
        label: str = "",
        check_stall: bool = True,
    ) -> np.ndarray:
        if self.last_cmd_6 is None:
            self.seed_from_present()
        assert self.last_cmd_6 is not None
        start_6 = self.last_cmd_6.copy()
        goal_6 = np.r_[np.asarray(target_cmd_5[:5], dtype=float), float(gripper_cmd)]

        max_arm_delta = float(np.max(np.abs(goal_6[:5] - start_6[:5])))
        max_grip_delta = abs(float(goal_6[5] - start_6[5]))
        duration = max(min_seconds, max_arm_delta / max(speed_deg_per_s, 10.0), max_grip_delta / 70.0)

        if label:
            print(f"  -> {label} ({duration:.2f}s)", flush=True)

        t0 = time.monotonic()
        last_sample = -1.0
        while True:
            tick = time.monotonic()
            elapsed = tick - t0
            alpha = min(1.0, elapsed / duration)
            w = smoothstep(alpha)
            cmd_6 = start_6 + w * (goal_6 - start_6)

            actual = self.read_measured()
            if check_stall:
                arm_err = np.abs(actual[:5] - cmd_6[:5])
                if float(np.max(arm_err)) > 18.0:
                    worst = JOINT_NAMES[int(np.argmax(arm_err))]
                    raise RuntimeError(
                        f"Écart de suivi excessif sur {worst} ({np.max(arm_err):.1f}°) pendant '{label}'."
                    )

            sent = self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, cmd_6)})
            self.last_cmd_6 = np.array([sent[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)

            if elapsed - last_sample >= 0.10 or alpha >= 1.0:
                last_sample = elapsed
                self.run_log["samples"].append(
                    {
                        "time_s": round(time.monotonic() - self.run_log["t_start_mono"], 3),
                        "phase": label,
                        "positions": np.round(actual, 3).tolist(),
                        "command": np.round(cmd_6, 3).tolist(),
                    }
                )
            if alpha >= 1.0:
                self.last_cmd_6 = goal_6.copy()
                break
            time.sleep(max(0.0, 1.0 / 120.0 - (time.monotonic() - tick)))

        return self.read_measured()

    def move_cartesian_vertical(
        self,
        ik: SO101CalibratedIK,
        x_mm: float,
        y_mm: float,
        z_start_mm: float,
        z_end_mm: float,
        yaw_delta_deg: float,
        lock_roll_deg: float,
        gripper_cmd: float,
        duration: float = 1.15,
        label: str = "",
    ) -> np.ndarray:
        """Pure vertical straight-line motion in Z with wrist_roll locked (zero lever twist!)."""
        if label:
            print(f"  -> {label} (Z: {z_start_mm:+.0f} -> {z_end_mm:+.0f} mm, {duration:.2f}s)", flush=True)
        # Pre-solve 16 smooth vertical waypoints BEFORE starting motion so the 120 Hz loop stays fast
        n_wp = 16
        zs = np.linspace(z_start_mm, z_end_mm, n_wp)
        cmds_5 = []
        seed = self.last_cmd_6[:5].copy() if self.last_cmd_6 is not None else None
        for z_val in zs:
            sol = ik.solve(
                x_mm,
                y_mm,
                float(z_val),
                yaw_delta_deg=yaw_delta_deg,
                initial_deg=seed,
                lock_wrist_roll_deg=lock_roll_deg,
            )
            seed = sol.state_deg
            cmds_5.append(sol.command_deg)
        cmds_5_arr = np.stack(cmds_5, axis=0)
        grid = np.linspace(0.0, 1.0, n_wp)

        t0 = time.monotonic()
        last_sample = -1.0
        while True:
            tick = time.monotonic()
            elapsed = tick - t0
            alpha = min(1.0, elapsed / max(duration, 0.4))
            w = smoothstep(alpha)
            cmd_5 = np.array([np.interp(w, grid, cmds_5_arr[:, j]) for j in range(5)], dtype=float)
            cmd_6 = np.r_[cmd_5, float(gripper_cmd)]

            actual = self.read_measured()
            arm_err = np.abs(actual[:5] - cmd_5)
            if float(np.max(arm_err)) > 18.0:
                worst = JOINT_NAMES[int(np.argmax(arm_err))]
                raise RuntimeError(
                    f"Écart de suivi vertical sur {worst} ({np.max(arm_err):.1f}°) pendant '{label}'."
                )

            sent = self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, cmd_6)})
            self.last_cmd_6 = np.array([sent[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)

            if elapsed - last_sample >= 0.10 or alpha >= 1.0:
                last_sample = elapsed
                self.run_log["samples"].append(
                    {
                        "time_s": round(time.monotonic() - self.run_log["t_start_mono"], 3),
                        "phase": label,
                        "positions": np.round(actual, 3).tolist(),
                        "command": np.round(cmd_6, 3).tolist(),
                    }
                )
            if alpha >= 1.0:
                self.last_cmd_6 = cmd_6.copy()
                break
            time.sleep(max(0.0, 1.0 / 120.0 - (time.monotonic() - tick)))

        return self.read_measured()

    def hold_seconds(self, seconds: float, gripper_cmd: float | None = None) -> np.ndarray:
        assert self.last_cmd_6 is not None
        cmd_6 = self.last_cmd_6.copy()
        if gripper_cmd is not None:
            cmd_6[5] = float(gripper_cmd)
            self.last_cmd_6 = cmd_6.copy()
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, cmd_6)})
            time.sleep(1.0 / 60.0)
        return self.read_measured()


def check_start_pose_safe(actual_6: np.ndarray, ik: SO101CalibratedIK, hover_z_mm: float) -> bool:
    """Allow starting from known rest pose (tolerating locked wrist_roll) OR a safe high hover pose."""
    rest_limits = np.array([16.0, 12.0, 14.0, 14.0, 90.0, 45.0], dtype=float)
    if np.all(np.abs(actual_6 - ORIGIN_REST_STATE) <= rest_limits):
        return True
    # Also allow starting if TCP is already safely above table (Z >= +45 mm)
    from so101_ik import fk_tcp_mm

    tcp_base_mm, _ = fk_tcp_mm(actual_6[:5])
    table_xyz = ik.base_to_table_mm(tcp_base_mm)
    return bool(table_xyz[2] >= 45.0)


def run_camera_calibration(
    runner: HardwareRunner,
    picker: RobustPicker,
    cam: WristCameraStream,
    custom_detector,
    calib_out_path: Path,
    auto_scan_cameras: bool = True,
) -> WristServoCalibration:
    """12-second mid-air self-calibration at the validated observation pose above (0, 0)."""
    z_h = picker.hover_z_mm
    print("\n[CALIBRATION CAMÉRA-POIGNET EN POSE D'OBSERVATION AU-DESSUS DE (0, 0)]", flush=True)

    # 0. Move to validated observation pose at (0, 0) (wrist_roll ~ -85 deg, camera facing table)
    sol_0 = picker.ik.solve_observe(0.0, 0.0)
    runner.move_joint_smooth(sol_0.command_deg, gripper_cmd=10.0, speed_deg_per_s=38.0, label="Pose d'observation (0,0)")
    runner.hold_seconds(0.50)

    active_cam = cam
    extra_cams: list[WristCameraStream] = []
    tested_indices: set[int] = set()
    if hasattr(cam, "camera_index"):
        tested_indices.add(int(cam.camera_index))

    def _open_next_detecting_cam(diag_list: list[str]) -> tuple[WristCameraStream | None, dict | None]:
        for candidate_idx in (0, 1, 2):
            if candidate_idx in tested_indices:
                continue
            tested_indices.add(candidate_idx)
            try:
                test_cam = WristCameraStream(camera_index=candidate_idx)
                extra_cams.append(test_cam)
            except Exception as exc:
                diag_list.append(f"cam {candidate_idx}: indisponible ({exc})")
                continue
            cand_det = test_cam.detect_stable(allow_border=True, custom_detector=custom_detector, samples=4)
            if cand_det is not None:
                print(f"  -> Puff détectée sur la caméra index {candidate_idx} !", flush=True)
                return test_cam, cand_det
            diag_list.append(f"cam {candidate_idx}: {test_cam.last_frame_diagnostics()}")
        return None, None

    try:
        diag_list: list[str] = []
        det_0 = active_cam.detect_stable(allow_border=True, custom_detector=custom_detector, samples=4)
        if det_0 is None and auto_scan_cameras and hasattr(active_cam, "camera_index"):
            diag_list.append(f"cam {active_cam.camera_index}: {active_cam.last_frame_diagnostics()}")
            print("  Puff non vue sur cette caméra -> test automatique des autres caméras USB (0, 1, 2)...", flush=True)
            found_cam, found_det = _open_next_detecting_cam(diag_list)
            if found_cam is not None and found_det is not None:
                active_cam, det_0 = found_cam, found_det
            else:
                print("  Diagnostics numériques caméras : " + " | ".join(diag_list), flush=True)

        if det_0 is None:
            print("  Retour sécurisé au repos avant arrêt...", flush=True)
            runner.move_joint_smooth(
                ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=36.0, label="Retour au repos"
            )
            raise RuntimeError(
                "Puff non détectée au-dessus de (0, 0) sur aucune caméra. "
                "Vérifier que la puff est bien posée sur (0, 0) et que la caméra poignet est branchée."
            )

        step_mm = 15.0
        while True:
            cam_idx = int(getattr(active_cam, "camera_index", 0))
            p0 = np.array(det_0["center"], dtype=float)
            angle_0 = float(det_0["angle_deg"])
            print(
                f"  Origine (0,0) vue sur caméra {cam_idx} à pixel=({p0[0]:.1f}, {p0[1]:.1f}), "
                f"angle={angle_0:+.1f}° [{det_0['source']}]",
                flush=True,
            )

            # 1. Step +15 mm in X (in observation pose) — also proves camera moves WITH the wrist!
            sol_x = picker.ik.solve_observe(step_mm, 0.0, initial_deg=sol_0.state_deg)
            runner.move_joint_smooth(sol_x.command_deg, gripper_cmd=10.0, speed_deg_per_s=35.0, label=f"Pas +X ({step_mm:.0f}mm)")
            runner.hold_seconds(0.42)
            det_x = active_cam.detect_stable(expected_center=tuple(p0), allow_border=True, custom_detector=custom_detector, samples=4)
            shift_x_px = float(np.linalg.norm(np.array(det_x["center"]) - p0)) if det_x is not None else 0.0

            if det_x is not None and shift_x_px >= 6.0:
                px = np.array(det_x["center"], dtype=float)
                break

            if auto_scan_cameras and hasattr(active_cam, "camera_index") and tested_indices != {0, 1, 2}:
                reason = "perdue au pas +X" if det_x is None else f"caméra fixe détectée (déplacement={shift_x_px:.1f} px)"
                print(f"  Caméra {cam_idx} écartée ({reason}) -> test de la caméra USB suivante...", flush=True)
                runner.move_joint_smooth(sol_0.command_deg, gripper_cmd=10.0, speed_deg_per_s=35.0, label="Retour (0,0)")
                runner.hold_seconds(0.42)
                found_cam, found_det = _open_next_detecting_cam(diag_list)
                if found_cam is not None and found_det is not None:
                    active_cam, det_0 = found_cam, found_det
                    continue

            runner.move_joint_smooth(ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=36.0, label="Retour au repos")
            if det_x is None:
                raise RuntimeError("Puff perdue lors du pas +X de calibration.")
            raise RuntimeError(
                f"La caméra {cam_idx} semble fixe (déplacement image={shift_x_px:.1f} px pour 15 mm). "
                "Vérifier que la caméra poignet HBV HD CAMERA est bien branchée."
            )

        # 2. Step +15 mm in Y (in observation pose)
        sol_y = picker.ik.solve_observe(0.0, step_mm, initial_deg=sol_0.state_deg)
        runner.move_joint_smooth(sol_y.command_deg, gripper_cmd=10.0, speed_deg_per_s=35.0, label=f"Pas +Y ({step_mm:.0f}mm)")
        runner.hold_seconds(0.42)
        det_y = active_cam.detect_stable(expected_center=tuple(p0), allow_border=True, custom_detector=custom_detector, samples=4)
        if det_y is None:
            runner.move_joint_smooth(ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=36.0, label="Retour au repos")
            raise RuntimeError("Puff perdue lors du pas +Y de calibration.")
        py = np.array(det_y["center"], dtype=float)

        J = np.column_stack([(px - p0) / step_mm, (py - p0) / step_mm])
        det_j = float(np.linalg.det(J))
        print(
            f"  Jacobien mesuré (px/mm) : col_X={J[:,0].round(2).tolist()}, col_Y={J[:,1].round(2).tolist()} (det={det_j:.2f})",
            flush=True,
        )
        if abs(det_j) < 0.35:
            runner.move_joint_smooth(ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=36.0, label="Retour au repos")
            raise RuntimeError(f"Jacobien caméra mal conditionné (det={det_j:.3f}). Vérifier que la caméra choisie est bien sur le poignet !")

        calib = WristServoCalibration(
            hover_z_mm=z_h,
            u_ref_px=float(p0[0]),
            v_ref_px=float(p0[1]),
            angle_ref_deg=angle_0,
            j_px_per_robot_mm=J,
            angle_gain_per_yaw_deg=-1.0,
            calibration_sha256=verify_calibration_hash(),
            camera_index=cam_idx,
            pose_mode="observe_pose",
            source="empirical_midair_3pt",
        )
        calib.save(calib_out_path)
        print(f"  Calibration sauvegardée dans : {calib_out_path} (caméra index={cam_idx})", flush=True)

        runner.move_joint_smooth(ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=38.0, label="Retour au repos")
        return calib
    finally:
        for extra in extra_cams:
            extra.close()


def visual_acquire_and_center(
    runner: HardwareRunner,
    picker: RobustPicker,
    cam: WristCameraStream,
    calib: WristServoCalibration,
    x_init_mm: float,
    y_init_mm: float,
    custom_detector,
    servo_log: list[dict],
) -> tuple[float, float, float, bool]:
    """Center the puff at hover; only return locked after a fresh converged view.

    The table-frame Kalman estimate bridges one short dropout. It never counts a
    prediction as a visual lock: the final centering decision requires a fresh,
    stable camera detection within the capture tolerance.
    """
    curr_x, curr_y = float(x_init_mm), float(y_init_mm)
    ref_uv = (calib.u_ref_px, calib.v_ref_px)

    # Move to validated observation pose above requested (X, Y)
    sol = picker.ik.solve_observe(curr_x, curr_y)
    runner.move_joint_smooth(
        sol.command_deg,
        gripper_cmd=12.0,
        speed_deg_per_s=picker.speed_deg_per_s,
        label=f"Observation ({curr_x:+.0f}, {curr_y:+.0f}) mm",
    )
    runner.hold_seconds(0.32)

    det = cam.detect_stable(expected_center=ref_uv, allow_border=True, custom_detector=custom_detector, samples=3)

    def detection_quality(candidate: dict | None) -> float:
        if candidate is None:
            return float("-inf")
        length = float(candidate.get("length_px", 0.0))
        width = max(float(candidate.get("width_px", 0.0)), 1.0)
        aspect = length / width
        edge_penalty = 10.0 if candidate.get("touching_border", False) else 0.0
        shape_penalty = min(abs(aspect - 4.5), 4.5)
        spread = float(candidate.get("spread_px", 16.0))
        return (
            2.0 * float(candidate.get("confidence", 0.0))
            + min(aspect, 8.0) * 0.25
            - edge_penalty
            - shape_penalty
            - spread * 0.05
        )

    # A border-touching/poor-shape detection can be the sheet edge or its
    # shadow. Try a small, bounded lateral sweep along table Y at the same
    # observation height, then keep the clearest fresh view. No descent occurs
    # during this search.
    det_aspect = 0.0 if det is None else float(det.get("length_px", 0.0)) / max(float(det.get("width_px", 0.0)), 1.0)
    if det is not None and (
        det.get("touching_border", False)
        or float(det.get("confidence", 0.0)) < 0.65
        or det_aspect < 2.3
    ):
        base_x, base_y = curr_x, curr_y
        best_det, best_x, best_y = det, curr_x, curr_y
        best_quality = detection_quality(det)
        print("  [VISION] Vue incertaine : petit balayage de caméra de ±8 mm sur Y, en hauteur.", flush=True)
        for y_offset_mm in (8.0, -8.0):
            scan_y = float(np.clip(base_y + y_offset_mm, WORKSPACE_Y_MM[0] + 5.0, WORKSPACE_Y_MM[1] - 5.0))
            if abs(scan_y - curr_y) < 0.1:
                continue
            try:
                scan_solution = picker.ik.solve_observe(base_x, scan_y)
            except ValueError:
                continue
            runner.move_joint_smooth(
                scan_solution.command_deg,
                gripper_cmd=12.0,
                speed_deg_per_s=28.0,
                min_seconds=0.35,
                label=f"Balayage caméra Y ({scan_y:+.1f} mm)",
            )
            runner.hold_seconds(0.20)
            candidate = cam.detect_stable(
                expected_center=ref_uv, allow_border=True,
                custom_detector=custom_detector, samples=3,
            )
            quality = detection_quality(candidate)
            servo_log.append({
                "stage": "camera_y_scan",
                "xy_mm": [round(base_x, 2), round(scan_y, 2)],
                "detected": candidate is not None,
                "detection_source": None if candidate is None else candidate.get("source", "unknown"),
                "detection_confidence": None if candidate is None else round(float(candidate.get("confidence", 0.0)), 3),
                "obs_px": None if candidate is None else np.round(candidate["center"], 1).tolist(),
                "touching_border": None if candidate is None else bool(candidate.get("touching_border", False)),
                "quality": round(quality, 3) if np.isfinite(quality) else None,
            })
            if quality > best_quality:
                best_det, best_x, best_y, best_quality = candidate, base_x, scan_y, quality
            if candidate is not None and not candidate.get("touching_border", False) and quality >= 1.0:
                best_det, best_x, best_y, best_quality = candidate, base_x, scan_y, quality
                break

        if (best_x, best_y) != (curr_x, curr_y):
            restore = picker.ik.solve_observe(best_x, best_y)
            runner.move_joint_smooth(
                restore.command_deg,
                gripper_cmd=12.0,
                speed_deg_per_s=28.0,
                min_seconds=0.35,
                label=f"Position caméra retenue Y={best_y:+.1f} mm",
            )
            runner.hold_seconds(0.20)
            best_det = cam.detect_stable(
                expected_center=ref_uv, allow_border=True,
                custom_detector=custom_detector, samples=3,
            )
        curr_x, curr_y, det = best_x, best_y, best_det

    # If puff is outside initial camera view, search neighborhood at z_extra=+20mm
    if det is None:
        print("  [VISION] Puff hors champ initial — scan visuel automatique autour de la cible...", flush=True)
        search_offsets = [(32.0, 0.0), (-32.0, 0.0), (0.0, 26.0), (0.0, -26.0)]
        for dx, dy in search_offsets:
            sx = float(np.clip(curr_x + dx, WORKSPACE_X_MM[0] + 5.0, WORKSPACE_X_MM[1] - 5.0))
            sy = float(np.clip(curr_y + dy, WORKSPACE_Y_MM[0] + 5.0, WORKSPACE_Y_MM[1] - 5.0))
            try:
                s_sol = picker.ik.solve_observe(sx, sy, z_extra_mm=20.0)
            except ValueError:
                continue
            runner.move_joint_smooth(s_sol.command_deg, gripper_cmd=12.0, speed_deg_per_s=40.0, label=f"Scan ({sx:+.0f},{sy:+.0f})")
            runner.hold_seconds(0.28)
            det = cam.detect_stable(expected_center=ref_uv, allow_border=True, custom_detector=custom_detector, samples=3)
            if det is not None:
                curr_x, curr_y = sx, sy
                print(f"  [VISION] Puff repérée pendant le scan vers ({curr_x:+.0f}, {curr_y:+.0f}) mm !", flush=True)
                h_sol = picker.ik.solve_observe(curr_x, curr_y)
                runner.move_joint_smooth(h_sol.command_deg, gripper_cmd=12.0, speed_deg_per_s=40.0)
                runner.hold_seconds(0.28)
                det = cam.detect_stable(expected_center=ref_uv, allow_border=True, custom_detector=custom_detector, samples=3)
                break

    if det is None:
        print("  [VISION] Puff toujours invisible : aucune descente ni fermeture autorisée.", flush=True)
        return curr_x, curr_y, 0.0, False

    yaw_target_deg = 0.0
    tracker = TableTargetKalman()
    missing_views = 0
    # Keep the conservative camera lock from the pre-teleop controller: require
    # both the fresh measurement and filtered estimate to be tightly centered.
    capture_tolerance_mm = 2.4
    raw_capture_tolerance_mm = 3.5
    max_prediction_uncertainty_mm = 5.0
    max_servo_steps = 5

    for step_idx in range(max_servo_steps):
        if step_idx > 0:
            det = cam.detect_stable(expected_center=ref_uv, allow_border=False, custom_detector=custom_detector, samples=3)
            if det is None:
                # A second short window tolerates a transient bad frame. The
                # filter may bridge one missed view while the arm remains high.
                det = cam.detect_stable(expected_center=ref_uv, allow_border=True, custom_detector=custom_detector, samples=3)

        if det is None:
            missing_views += 1
            prediction = tracker.predict()
            if prediction is None:
                print("  [VISION] Pas de position mémorisée ; maintien en hauteur.", flush=True)
                return curr_x, curr_y, yaw_target_deg, False
            object_xy_mm, uncertainty_mm = prediction
            servo_log.append(
                {
                    "iter": step_idx,
                    "xy_mm": [round(curr_x, 2), round(curr_y, 2)],
                    "predicted_object_xy_mm": np.round(object_xy_mm, 2).tolist(),
                    "uncertainty_mm": round(uncertainty_mm, 2),
                    "camera_missing": True,
                }
            )
            if missing_views > 1 or uncertainty_mm > max_prediction_uncertainty_mm:
                print(
                    f"  [VISION] Occultation trop longue/incertaine ({uncertainty_mm:.1f} mm) : "
                    "maintien en hauteur, aucune prise à l'aveugle.",
                    flush=True,
                )
                return curr_x, curr_y, yaw_target_deg, False

            # One bounded prediction-only step can bridge an occlusion. We then
            # require a new camera detection before allowing descent.
            predicted_delta = object_xy_mm - np.array([curr_x, curr_y], dtype=float)
            norm = float(np.linalg.norm(predicted_delta))
            step = predicted_delta * 0.5
            step_norm = float(np.linalg.norm(step))
            if step_norm > 8.0:
                step *= 8.0 / step_norm
            next_x = float(np.clip(curr_x + step[0], WORKSPACE_X_MM[0] + 2.0, WORKSPACE_X_MM[1] - 2.0))
            next_y = float(np.clip(curr_y + step[1], WORKSPACE_Y_MM[0] + 2.0, WORKSPACE_Y_MM[1] - 2.0))
            if norm < capture_tolerance_mm:
                print("  [VISION] Puff prédite près du centre, mais une détection fraîche reste requise.", flush=True)
                return curr_x, curr_y, yaw_target_deg, False
            step_sol = picker.ik.solve_observe(next_x, next_y)
            curr_x, curr_y = next_x, next_y
            runner.move_joint_smooth(
                step_sol.command_deg, gripper_cmd=12.0, speed_deg_per_s=30.0,
                min_seconds=0.35, label=f"Recentrage prédit borné ({curr_x:+.1f}, {curr_y:+.1f}) mm",
            )
            runner.hold_seconds(0.25)
            continue

        missing_views = 0
        obs_uv = (det["center"][0], det["center"][1])
        # Recover the full camera-to-table correction (gain 1) to express the
        # object's estimated position in the fixed paper frame.
        raw_delta_mm, raw_err_mm = calib.compute_correction_mm(
            obs_uv, current_yaw_deg=0.0, gain=1.0, max_step_mm=200.0
        )
        measured_object_xy = np.array([curr_x, curr_y], dtype=float) + raw_delta_mm
        filtered_object_xy, uncertainty_mm = tracker.update(measured_object_xy)
        dxy_mm = (filtered_object_xy - np.array([curr_x, curr_y], dtype=float)) * 0.82
        step_norm = float(np.linalg.norm(dxy_mm))
        if step_norm > 26.0:
            dxy_mm *= 26.0 / step_norm
        est_err_mm = float(np.linalg.norm(filtered_object_xy - np.array([curr_x, curr_y], dtype=float)))
        dyaw_deg, _ = calib.compute_yaw_correction_deg(det["angle_deg"], gain=0.85, max_step_deg=35.0)
        if picker.align_angle and not det.get("touching_border", False):
            yaw_target_deg = float(np.clip(dyaw_deg, -35.0, +35.0))

        servo_log.append(
            {
                "iter": step_idx,
                "xy_mm": [round(curr_x, 2), round(curr_y, 2)],
                "obs_px": [round(obs_uv[0], 1), round(obs_uv[1], 1)],
                "raw_est_err_mm": round(raw_err_mm, 2),
                "est_err_mm": round(est_err_mm, 2),
                "filtered_object_xy_mm": np.round(filtered_object_xy, 2).tolist(),
                "uncertainty_mm": round(uncertainty_mm, 2),
                "angle_obs_deg": round(det["angle_deg"], 1),
                "detection_source": det.get("source", "unknown"),
                "detection_confidence": round(float(det.get("confidence", 0.0)), 3),
                "detection_size_px": [round(float(det.get("length_px", 0.0)), 1),
                                      round(float(det.get("width_px", 0.0)), 1)],
                "detection_spread_px": round(float(det.get("spread_px", 0.0)), 2),
                "yaw_target_deg": round(yaw_target_deg, 1),
                "touching_border": bool(det.get("touching_border", False)),
            }
        )
        print(
            f"  [SERVO {step_idx+1}] pixel=({obs_uv[0]:.0f},{obs_uv[1]:.0f}) -> "
            f"écart brut {raw_err_mm:.1f} mm, filtré {est_err_mm:.1f} ± {uncertainty_mm:.1f} mm "
            f"(correction dX={dxy_mm[0]:+.1f}, dY={dxy_mm[1]:+.1f} mm, yaw={yaw_target_deg:+.1f}°)",
            flush=True,
        )

        # A filtered estimate alone can lag a real puff displacement. Require
        # the current fresh measurement to agree as well as the Kalman state.
        if (raw_err_mm <= raw_capture_tolerance_mm
                and est_err_mm <= capture_tolerance_mm
                and uncertainty_mm <= max_prediction_uncertainty_mm
                and not det.get("touching_border", False)):
            print("  [VISION] Centrage confirmé par une vue fraîche.", flush=True)
            return curr_x, curr_y, yaw_target_deg, True

        if step_idx == max_servo_steps - 1:
            break

        next_x = float(np.clip(curr_x + dxy_mm[0], WORKSPACE_X_MM[0] + 2.0, WORKSPACE_X_MM[1] - 2.0))
        next_y = float(np.clip(curr_y + dxy_mm[1], WORKSPACE_Y_MM[0] + 2.0, WORKSPACE_Y_MM[1] - 2.0))
        try:
            step_sol = picker.ik.solve_observe(next_x, next_y)
        except ValueError:
            print("  [VISION] Recentrage hors enveloppe : maintien en hauteur.", flush=True)
            return curr_x, curr_y, yaw_target_deg, False
        curr_x, curr_y = next_x, next_y
        runner.move_joint_smooth(
            step_sol.command_deg,
            gripper_cmd=12.0,
            speed_deg_per_s=36.0,
            min_seconds=0.30,
            label=f"Recentrage ({curr_x:+.1f}, {curr_y:+.1f}) mm",
        )
        runner.hold_seconds(0.25)

    print(
        f"  [VISION] Centrage non confirmé après {max_servo_steps} étapes "
        f"(dernière erreur estimée {est_err_mm:.1f} mm) : aucune descente.",
        flush=True,
    )
    return curr_x, curr_y, yaw_target_deg, False


def execute_robust_pick_cycle(
    runner: HardwareRunner,
    picker: RobustPicker,
    x_req_mm: float,
    y_req_mm: float,
    cam: WristCameraStream | None,
    calib: WristServoCalibration | None,
    custom_detector,
    drop_xy_mm: tuple[float, float] | None,
    max_retries: int,
    cycle_log: dict,
) -> bool:
    """Execute one complete pick-verify-lift-place cycle with automatic retry on slip/miss."""
    blind_retry_offsets = [(0.0, 0.0), (8.0, 0.0), (-8.0, 0.0), (0.0, 8.0)]
    caught = False
    target_x, target_y, target_yaw = float(x_req_mm), float(y_req_mm), 0.0

    for attempt in range(max_retries + 1):
        attempt_info: dict = {"attempt": attempt + 1, "servo_steps": []}
        cycle_log["attempts"].append(attempt_info)
        print(f"\n=== TENTATIVE {attempt + 1}/{max_retries + 1} vers ({target_x:+.1f}, {target_y:+.1f}) mm ===", flush=True)

        if cam is not None and calib is not None:
            cx, cy, cyaw, locked = visual_acquire_and_center(
                runner=runner,
                picker=picker,
                cam=cam,
                calib=calib,
                x_init_mm=target_x,
                y_init_mm=target_y,
                custom_detector=custom_detector,
                servo_log=attempt_info["servo_steps"],
            )
            attempt_info["visually_locked"] = locked
            if not locked:
                attempt_info["stopped_before_descent"] = "camera_lock_not_confirmed"
                cycle_log["grasp_confirmed_by_gripper"] = False
                print(
                    "  [SÉCURITÉ] Le centrage caméra n'est pas confirmé : "
                    "pas de descente, pas de fermeture. Retour au repos.",
                    flush=True,
                )
                break
        else:
            ox, oy = blind_retry_offsets[min(attempt, len(blind_retry_offsets) - 1)]
            cx = float(np.clip(x_req_mm + ox, WORKSPACE_X_MM[0] + 2.0, WORKSPACE_X_MM[1] - 2.0))
            cy = float(np.clip(y_req_mm + oy, WORKSPACE_Y_MM[0] + 2.0, WORKSPACE_Y_MM[1] - 2.0))
            cyaw = 0.0
            attempt_info["visually_locked"] = False
            hover_sol = picker.ik.solve(cx, cy, picker.hover_z_mm, yaw_delta_deg=0.0)
            runner.move_joint_smooth(
                hover_sol.command_deg,
                gripper_cmd=picker.gripper_open,
                speed_deg_per_s=picker.speed_deg_per_s,
                label=f"Survol IK ({cx:+.1f}, {cy:+.1f}) mm",
            )

        attempt_info["grasp_xy_mm"] = [round(cx, 2), round(cy, 2)]
        attempt_info["grasp_yaw_deg"] = round(cyaw, 2)

        # Orient wrist in mid-air + open funnel jaws wide BEFORE descending vertically
        pre_sol = picker.ik.solve(cx, cy, picker.hover_z_mm, yaw_delta_deg=cyaw)
        locked_roll = float(pre_sol.state_deg[4])
        runner.move_joint_smooth(
            pre_sol.command_deg,
            gripper_cmd=picker.gripper_open,
            speed_deg_per_s=picker.speed_deg_per_s,
            min_seconds=0.35,
            label=f"Ouverture entonnoir ({picker.gripper_open:.0f} pts)",
        )

        # Pure vertical straight-line descent to table Z
        runner.move_cartesian_vertical(
            ik=picker.ik,
            x_mm=cx,
            y_mm=cy,
            z_start_mm=picker.hover_z_mm,
            z_end_mm=picker.z_offset_mm,
            yaw_delta_deg=cyaw,
            lock_roll_deg=locked_roll,
            gripper_cmd=picker.gripper_open,
            duration=1.25,
            label="Descente verticale",
        )

        # Smooth firm squeeze + pad compression dwell
        grasp_sol = picker.ik.solve(
            cx,
            cy,
            picker.z_offset_mm,
            yaw_delta_deg=cyaw,
            initial_deg=pre_sol.state_deg,
            lock_wrist_roll_deg=locked_roll,
        )
        runner.move_joint_smooth(
            grasp_sol.command_deg,
            gripper_cmd=picker.gripper_close,
            speed_deg_per_s=30.0,
            min_seconds=0.65,
            label="Fermeture franche des mors",
        )
        bottom_meas = runner.hold_seconds(0.45, gripper_cmd=picker.gripper_close)
        grip_bottom = float(bottom_meas[5])
        attempt_info["gripper_pos_bottom"] = round(grip_bottom, 2)

        # Pure vertical straight-line liftoff WITHOUT wrist_roll twist!
        lift_z = picker.hover_z_mm + 8.0
        runner.move_cartesian_vertical(
            ik=picker.ik,
            x_mm=cx,
            y_mm=cy,
            z_start_mm=picker.z_offset_mm,
            z_end_mm=lift_z,
            yaw_delta_deg=cyaw,
            lock_roll_deg=locked_roll,
            gripper_cmd=picker.gripper_close,
            duration=1.35,
            label="Levage vertical sans torsion",
        )
        top_meas = runner.hold_seconds(0.40, gripper_cmd=picker.gripper_close)
        grip_top = float(top_meas[5])
        attempt_info["gripper_pos_top"] = round(grip_top, 2)

        if grip_top >= picker.min_grasp_pos:
            print(
                f"  [SUCCÈS DÉTECTÉ PAR PROPRIOCEPTION] Puff maintenue en haut "
                f"(pince={grip_top:.2f} pts >= seuil {picker.min_grasp_pos:.1f} pts) !",
                flush=True,
            )
            caught = True
            target_x, target_y, target_yaw = cx, cy, cyaw
            break

        print(
            f"  [ALERTE GLISSEMENT/VIDE] Pince mesurée={grip_top:.2f} pts (< {picker.min_grasp_pos:.1f} pts) : "
            "la puff n'est pas dans les mors.",
            flush=True,
        )
        if attempt < max_retries:
            print("  -> Réouverture et reprise automatique...", flush=True)
            lift_sol = picker.ik.solve(cx, cy, picker.hover_z_mm, yaw_delta_deg=0.0)
            runner.move_joint_smooth(
                lift_sol.command_deg,
                gripper_cmd=picker.gripper_open * 0.7,
                speed_deg_per_s=picker.speed_deg_per_s,
                label="Remontée de reprise",
            )
            target_x, target_y = cx, cy

    cycle_log["grasp_confirmed_by_gripper"] = caught

    # Place / Drop phase (only if caught, otherwise return cleanly to rest)
    if caught:
        if drop_xy_mm is not None:
            dx, dy = drop_xy_mm
            drop_h = picker.ik.solve(dx, dy, picker.hover_z_mm + 10.0, yaw_delta_deg=0.0)
            drop_low = picker.ik.solve(dx, dy, max(12.0, picker.z_offset_mm + 18.0), yaw_delta_deg=0.0)
            runner.move_joint_smooth(
                drop_h.command_deg,
                gripper_cmd=picker.gripper_close,
                speed_deg_per_s=picker.speed_deg_per_s,
                label=f"Transport vers zone de tri ({dx:+.0f}, {dy:+.0f}) mm",
            )
            runner.move_joint_smooth(
                drop_low.command_deg,
                gripper_cmd=picker.gripper_close,
                speed_deg_per_s=35.0,
                label="Approche dépose tri",
            )
            runner.move_joint_smooth(
                drop_low.command_deg,
                gripper_cmd=picker.gripper_open,
                speed_deg_per_s=35.0,
                min_seconds=0.45,
                label="Ouverture et lâcher",
            )
            runner.hold_seconds(0.30)
            runner.move_joint_smooth(
                drop_h.command_deg,
                gripper_cmd=12.0,
                speed_deg_per_s=picker.speed_deg_per_s,
                label="Dégagement vertical",
            )
        else:
            # Gentle put-down back on table (same behavior as fixed_pick.py when no bin is specified)
            locked_roll = float(
                picker.ik.solve(target_x, target_y, picker.hover_z_mm, yaw_delta_deg=target_yaw).state_deg[4]
            )
            runner.move_cartesian_vertical(
                ik=picker.ik,
                x_mm=target_x,
                y_mm=target_y,
                z_start_mm=picker.hover_z_mm + 8.0,
                z_end_mm=picker.z_offset_mm + 3.0,
                yaw_delta_deg=target_yaw,
                lock_roll_deg=locked_roll,
                gripper_cmd=picker.gripper_close,
                duration=1.15,
                label="Repose douce sur table",
            )
            grasp_sol = picker.ik.solve(
                target_x,
                target_y,
                picker.z_offset_mm + 3.0,
                yaw_delta_deg=target_yaw,
                lock_wrist_roll_deg=locked_roll,
            )
            runner.move_joint_smooth(
                grasp_sol.command_deg,
                gripper_cmd=picker.gripper_open,
                speed_deg_per_s=35.0,
                min_seconds=0.45,
                label="Ouverture franche des mors",
            )
            runner.hold_seconds(0.25)
            runner.move_cartesian_vertical(
                ik=picker.ik,
                x_mm=target_x,
                y_mm=target_y,
                z_start_mm=picker.z_offset_mm + 3.0,
                z_end_mm=picker.hover_z_mm,
                yaw_delta_deg=target_yaw,
                lock_roll_deg=locked_roll,
                gripper_cmd=12.0,
                duration=1.0,
                label="Dégagement vertical",
            )

    # Return cleanly via high intermediate pose to validated rest pose
    safe_mid = np.array([ORIGIN_REST_ACTION[0], -55.0, 55.0, 65.0, ORIGIN_REST_ACTION[4]], dtype=float)
    runner.move_joint_smooth(safe_mid, gripper_cmd=4.0, speed_deg_per_s=picker.speed_deg_per_s, label="Repli sécurisé")
    runner.move_joint_smooth(
        ORIGIN_REST_ACTION[:5],
        gripper_cmd=ORIGIN_REST_ACTION[5],
        speed_deg_per_s=38.0,
        label="Retour au repos",
    )
    return caught


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-mm", type=float, default=0.0, help="Position cible X sur table en mm (+X = droite du robot).")
    parser.add_argument("--y-mm", type=float, default=0.0, help="Position cible Y sur table en mm (+Y = loin du socle).")
    parser.add_argument("--drop-xy", type=float, nargs=2, metavar=("DROP_X", "DROP_Y"), help="Coordonnées (X, Y) mm de dépose/tri optionnelles.")
    parser.add_argument("--interactive", action="store_true", help="Mode continu : saisir des positions 'X Y' successives.")
    parser.add_argument("--calibrate-camera", action="store_true", help="Auto-calibrer la caméra poignet en 10s en pose d'observation au-dessus de (0,0).")
    parser.add_argument("--return-rest", action="store_true", help="Ramener doucement le bras à la pose de repos sous couple.")
    parser.add_argument("--maintenance-pose", action="store_true", help="Ouvrir la pince et la placer à la pose verticale connue, 80 mm au-dessus de (0,0), puis maintenir le couple pour intervenir sur les mors.")
    parser.add_argument("--no-camera", action="store_true", help="Désactiver la caméra poignet (IK 3D + prise entonnoir + reprise proprioceptive).")
    parser.add_argument("--use-theoretical-camera", action="store_true", help="Utiliser le Jacobien analytique so101.xml si pas encore calibré.")
    parser.add_argument("--camera", type=int, default=None, help="Index OpenCV de la caméra poignet (auto-détecté / lu depuis la calibration si omis).")
    parser.add_argument("--servo-calib", type=Path, default=DEFAULT_SERVO_CALIB_PATH, help="Fichier de calibration caméra-poignet.")
    parser.add_argument("--calibration-output", type=Path, help="Destination nouvelle de calibration; refuse tout fichier existant.")
    parser.add_argument("--detector", type=Path, help="Module Python optionnel des Data Scientists définissant detect(frame_bgr).")
    parser.add_argument("--no-angle-align", action="store_true", help="Garder l'orientation nominale sans tourner le poignet selon l'angle vu.")
    parser.add_argument("--z-offset-mm", type=float, default=0.0, help="Ajustement fin de hauteur de prise Z en mm (ex. +2 plus haut, -1 plus bas).")
    parser.add_argument("--hover-z-mm", type=float, default=DEFAULT_HOVER_Z_MM, help="Hauteur de survol/observation en mm (défaut : 80).")
    parser.add_argument("--gripper-open", type=float, default=36.0, help="Ouverture de pince en points [5..65] (défaut : 36).")
    parser.add_argument("--gripper-close", type=float, default=3.2, help="Consigne de serrage de la pince (défaut : 3.2).")
    parser.add_argument("--min-grasp-pos", type=float, default=4.6, help="Seuil Present_Position pince confirmant la présence de la puff (défaut : 4.6).")
    parser.add_argument("--max-retries", type=int, default=2, help="Nombre maximal de reprises automatiques si glissement/vide (défaut : 2).")
    parser.add_argument("--speed", type=float, default=42.0, help="Vitesse articulaire max en deg/s [15..75] (défaut : 42).")
    parser.add_argument("--dry-run", action="store_true", help="Vérifier toute la cinématique et la configuration hors matériel.")
    add_hardware_flag(parser)
    args = parser.parse_args()
    if not args.dry_run:
        require_hardware(parser, args.hardware)

    verify_calibration_hash(CALIBRATION_PATH)
    ik = SO101CalibratedIK()
    picker = RobustPicker(
        ik=ik,
        hover_z_mm=args.hover_z_mm,
        z_offset_mm=args.z_offset_mm,
        gripper_open=args.gripper_open,
        gripper_close=args.gripper_close,
        min_grasp_pos=args.min_grasp_pos,
        speed_deg_per_s=args.speed,
        align_angle=not args.no_angle_align,
    )
    drop_xy = (float(args.drop_xy[0]), float(args.drop_xy[1])) if args.drop_xy else None
    custom_detector = load_custom_detector(args.detector)

    # Validate kinematics offline BEFORE connecting to hardware
    maintenance_pose = None
    if args.maintenance_pose:
        maintenance_pose = ik.solve(0.0, 0.0, args.hover_z_mm, yaw_delta_deg=0.0)
        plan = None
        print(
            f"Pose de maintenance validée : pince verticale ouverte à Z=+{args.hover_z_mm:.0f} mm au-dessus de (0,0).",
            flush=True,
        )
        print("  Angles (deg) :", dict(zip(JOINT_NAMES[:5], np.round(maintenance_pose.state_deg, 2).tolist())))
    else:
        plan = picker.plan_blind_sequence(args.x_mm, args.y_mm, yaw_delta_deg=0.0, drop_xy_mm=drop_xy)
    if plan is not None:
        print(
            f"Plan IK validé pour X={args.x_mm:+.1f} mm, Y={args.y_mm:+.1f} mm "
            f"(erreur résiduelle {plan['grasp'].pos_error_mm:.3f} mm).",
            flush=True,
        )
        print("  Angles prise (deg) :", dict(zip(JOINT_NAMES[:5], np.round(plan["grasp"].state_deg, 2).tolist())))

    if args.dry_run:
        print("Mode --dry-run terminé avec succès : aucune caméra ni moteur ouvert.")
        return

    # Load or prepare camera mode
    use_cam = False if (args.return_rest or args.maintenance_pose) else (args.calibrate_camera or (not args.no_camera))
    servo_calib: WristServoCalibration | None = None
    if not args.calibrate_camera and use_cam:
        if Path(args.servo_calib).is_file():
            servo_calib = WristServoCalibration.load(args.servo_calib)
            print(
                f"Guidage visuel poignet ACTIF (calibration : {args.servo_calib.name}, caméra index={servo_calib.camera_index}).",
                flush=True,
            )
        elif args.use_theoretical_camera:
            servo_calib = WristServoCalibration.theoretical_default(args.hover_z_mm)
            print("Guidage visuel poignet ACTIF (modèle théorique so101.xml).", flush=True)
        else:
            raise RuntimeError(
                "Calibration caméra-poignet absente : aucune prise robuste ne sera lancée à l'aveugle. "
                "Lancez `python robust_pick.py --hardware --calibrate-camera`, ou demandez explicitement `--no-camera` "
                "pour un essai expérimental sans vision."
            )

    log_dir = ROOT / "outputs/robust_pick_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")

    mode_name = (
        "maintenance_pose" if args.maintenance_pose else (
            "return_rest" if args.return_rest else (
                "calibrate_camera" if args.calibrate_camera else ("interactive" if args.interactive else "single_pick")
            )
        )
    )
    run_log: dict = {
        "controller": "robust_pick_v3",
        "mode": mode_name,
        "camera_used": bool(use_cam),
        "custom_detector_used": custom_detector is not None,
        "t_start_mono": time.monotonic(),
        "completed": False,
        "cycles": [],
        "samples": [],
        "controller_snapshot": "reconstructed_pre_teleop_camera_kalman",
    }

    cam_stream: WristCameraStream | None = None
    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            calibration_dir=ROOT / "calibrations/follower",
            max_relative_target=6.0,
        )
    )
    robot.bus.default_timeout = 150
    active = False

    try:
        if use_cam:
            if args.camera is not None:
                cam_stream = WristCameraStream(camera_index=args.camera)
            else:
                preferred_idx = servo_calib.camera_index if servo_calib is not None else 0
                candidates = [preferred_idx] + [i for i in (0, 1, 2) if i != preferred_idx]
                last_cam_err = None
                for c_idx in candidates:
                    try:
                        cam_stream = WristCameraStream(camera_index=c_idx)
                        break
                    except Exception as exc:
                        last_cam_err = exc
                if cam_stream is None:
                    raise RuntimeError(f"Aucune caméra USB accessible sur les index (0, 1, 2) : {last_cam_err}")
            print(
                f"Caméra ouverte (index {cam_stream.camera_index}, {cam_stream.width}x{cam_stream.height}).",
                flush=True,
            )

        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration moteurs non conforme au fichier mon_bras.json.")
        for name in JOINT_NAMES:
            if robot.bus.read("Operating_Mode", name, normalize=False, num_retry=2) != 0:
                raise RuntimeError(f"Mode position (0) requis sur {name}.")

        runner = HardwareRunner(robot, run_log)
        initial = runner.read_measured()
        if not check_start_pose_safe(initial, ik, picker.hover_z_mm):
            print("Positions actuelles :", dict(zip(JOINT_NAMES, np.round(initial, 2).tolist())))
            print("Repos attendu       :", dict(zip(JOINT_NAMES, np.round(ORIGIN_REST_STATE, 2).tolist())))
            raise RuntimeError(
                "Pose de départ hors zone sûre. Ramener le bras au repos connu (ou utiliser release_follower.py en le soutenant)."
            )

        word = "MAINTENANCE" if args.maintenance_pose else ("REPOS" if args.return_rest else ("CALIBRER" if args.calibrate_camera else "ATTRAPER"))
        if args.calibrate_camera:
            print("\nPlacez la puff exactement sur le repère (0, 0), même orientation que fixed_pick.py, zone dégagée.")
        elif args.maintenance_pose:
            print("\nZone dégagée ; le bras ira à la pose de maintenance avec la pince ouverte et restera sous couple.")
        else:
            print("\nZone dégagée ; Ctrl+C arrête immédiatement avec maintien sous couple.")
        if input(f"Tape {word} pour activer le bras : ").strip() != word:
            print("Annulé par l'opérateur.")
            return

        recheck = runner.read_measured()
        if np.max(np.abs(recheck[:5] - initial[:5])) > 6.0:
            raise RuntimeError("Le bras a bougé pendant la confirmation : relancer.")

        runner.seed_from_present()
        active = True
        robot.bus.enable_torque(num_retry=3)

        if args.maintenance_pose:
            assert maintenance_pose is not None
            safe_mid = np.array([ORIGIN_REST_ACTION[0], -55.0, 55.0, 65.0, ORIGIN_REST_ACTION[4]], dtype=float)
            runner.move_joint_smooth(
                safe_mid, gripper_cmd=float(initial[5]), speed_deg_per_s=30.0, label="Dégagement pour maintenance"
            )
            runner.move_joint_smooth(
                maintenance_pose.command_deg,
                gripper_cmd=picker.gripper_open,
                speed_deg_per_s=30.0,
                min_seconds=0.8,
                label="Pince ouverte, verticale pour pose du matériau",
            )
            run_log["completed"] = True
            run_log["maintenance_pose_deg"] = np.round(maintenance_pose.state_deg, 2).tolist()
            print("\nPince ouverte en pose de maintenance. Couple maintenu : pose le matériau sur les deux mors, puis arrête ce programme avec Ctrl+C.", flush=True)
            return

        if args.return_rest:
            safe_mid = np.array([ORIGIN_REST_ACTION[0], -55.0, 55.0, 65.0, ORIGIN_REST_ACTION[4]], dtype=float)
            runner.move_joint_smooth(safe_mid, gripper_cmd=float(initial[5]), speed_deg_per_s=36.0, label="Repli intermédiaire")
            runner.move_joint_smooth(
                ORIGIN_REST_ACTION[:5], gripper_cmd=ORIGIN_REST_ACTION[5], speed_deg_per_s=35.0, label="Retour au repos"
            )
            run_log["completed"] = True
            print("\nBras ramené au repos sous couple.", flush=True)
            return

        if args.calibrate_camera:
            assert cam_stream is not None
            calibration_output = args.calibration_output or (
                ROOT / "real_transfer" /
                ("wrist_servo_calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
            )
            if calibration_output.exists():
                raise FileExistsError(f"Le fichier existant sera préservé : {calibration_output}")
            run_camera_calibration(
                runner,
                picker,
                cam_stream,
                custom_detector,
                calibration_output,
                auto_scan_cameras=(args.camera is None),
            )
            run_log["completed"] = True
            print("\nCalibration caméra-poignet terminée ! Vous pouvez maintenant lancer des prises robustes.", flush=True)
            return

        if args.interactive:
            print("\nMODE INTERACTIF : entrez 'X Y' en millimètres (ex: '25 -15'), ou 'q' pour quitter.", flush=True)
            while True:
                raw = input("\nCible 'X Y' (mm) [ou q] > ").strip()
                if raw.lower() in ("q", "quit", "exit", ""):
                    break
                parts = raw.replace(",", " ").split()
                if len(parts) != 2:
                    print("Format attendu : deux nombres 'X Y' en mm (ex: 20 -10).")
                    continue
                try:
                    tx, ty = float(parts[0]), float(parts[1])
                    picker.plan_blind_sequence(tx, ty, drop_xy_mm=drop_xy)
                except ValueError as exc:
                    print(f"Cible refusée : {exc}")
                    continue
                cyc = {"requested_xy_mm": [tx, ty], "attempts": []}
                run_log["cycles"].append(cyc)
                ok = execute_robust_pick_cycle(
                    runner, picker, tx, ty, cam_stream, servo_calib, custom_detector, drop_xy, args.max_retries, cyc
                )
                print(f"Résultat cycle ({tx:+.0f}, {ty:+.0f}) mm : {'PRISE CONFIRMÉE' if ok else 'ÉCHEC APRÈS REPRISES'}.", flush=True)
            run_log["completed"] = True
        else:
            cyc = {"requested_xy_mm": [args.x_mm, args.y_mm], "attempts": []}
            run_log["cycles"].append(cyc)
            ok = execute_robust_pick_cycle(
                runner,
                picker,
                args.x_mm,
                args.y_mm,
                cam_stream,
                servo_calib,
                custom_detector,
                drop_xy,
                args.max_retries,
                cyc,
            )
            run_log["completed"] = True
            print(
                f"\nCYCLE TERMINÉ : {'PRISE ET MAINTIEN CONFIRMÉS' if ok else 'OBJET NON MAINTENU'}. "
                "Confirmez visuellement le résultat physique.",
                flush=True,
            )

    except KeyboardInterrupt:
        run_log["stop"] = "Ctrl+C"
        print("\nInterruption manuelle (Ctrl+C).", flush=True)
    except Exception as error:
        run_log["error"] = str(error)
        print(f"\nARRÊT SÉCURITÉ : {error}", flush=True)
    finally:
        if cam_stream is not None:
            cam_stream.close()
        if robot.bus.is_connected:
            if active:
                hold_and_disconnect(robot.bus)
            else:
                robot.bus.disconnect(disable_torque=False)
        run_log.pop("t_start_mono", None)
        log_path.write_text(json.dumps(run_log, indent=2) + "\n")
        print(f"Journal articulaire sauvegardé : {log_path}", flush=True)


if __name__ == "__main__":
    main()
