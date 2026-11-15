"""Calibrated 3D Forward & Inverse Kinematics for the real SO-101 ('mon_bras').

Uses the exact rigid-body chain from RobotStudio SO-101 (so101.xml), verified
against all 11 physical table captures (mean residual < 3 mm across 28 deg of
elbow redundancy) and anchored with zero error on the 3x physically validated
(0, 0) reference grasp trajectory (puff_pilot_20261114_235138_trimmed).

Never opens serial ports or moves hardware on its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "calibrations/follower/mon_bras.json"
EXPECTED_CALIBRATION_SHA256 = "3e8680710c2c8f08e929fe453b853e9e943ae5bc31cf19923bef3d7822744ee4"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Calibrated mechanical limits on 'mon_bras' (with safety margin from hard stops).
# Real normalized ranges from mon_bras.json:
# pan [-118.2, +118.2], lift [-102.8, +102.8], elbow [-90.7, +90.7],
# wrist_flex [-100.3, +100.3], wrist_roll [-180, +180].
SAFE_JOINT_LOW_DEG = np.array([-95.0, -102.6, -85.0, -92.0, -174.5], dtype=float)
SAFE_JOINT_HIGH_DEG = np.array([+95.0, +85.0, +90.0, +95.0, +174.5], dtype=float)

# Validated (0, 0) physical grasp pose from puff_pilot_20261114_235138_trimmed (t ~ 11.25s)
ORIGIN_REST_STATE = np.array([-8.75, -102.55, 93.27, 77.19, -94.46, 0.55], dtype=float)
ORIGIN_REST_ACTION = np.array([-8.57, -102.81, 97.23, 77.36, -94.15, 0.51], dtype=float)

ORIGIN_GRASP_STATE_5 = np.array([-12.527, 4.703, 31.824, 50.462, -167.165], dtype=float)
# Physically validated wrist-camera observation pose above (0, 0) (from 20261114_231754)
ORIGIN_OBSERVE_STATE_5 = np.array([-14.0, -41.10, 51.25, 50.55, -85.14], dtype=float)
# Measured servo steady-state gravity/stiffness feedforward offset (action - state) at table grasp:
GRASP_FEEDFORWARD_DEG = np.array([-0.176, -0.879, +0.088, -0.352, 0.0], dtype=float)

# Paper frame alignment fitted on the 11 physical captures (table_20261115_010604.json):
# +X_table = robot right, +Y_table = away from base, rotated +6.34 deg in base XY.
DEFAULT_PAPER_YAW_DEG = 6.34

# Safe table workspace bounds (mm relative to validated (0, 0) puff location)
WORKSPACE_X_MM = (-100.0, +100.0)
WORKSPACE_Y_MM = (-60.0, +95.0)
WORKSPACE_Z_MM = (-6.0, +135.0)
WORKSPACE_RADIUS_BASE_MM = (125.0, 325.0)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _tf(pos: list[float], quat: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quat_to_mat(np.asarray(quat, dtype=float))
    T[:3, 3] = pos
    return T


def _rotz(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    T = np.eye(4, dtype=float)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return T


# Official SO-101 rigid body transforms (meters, radians) from so101.xml
_T_BASE_SH = _tf([0.0388353, 0.0, 0.0624], [0.0, 0.0, -1.0, 0.0])
_T_SH_UA = _tf([-0.0303992, -0.0182778, -0.0542], [1.0, -1.0, -1.0, -1.0])
_T_UA_LA = _tf([-0.11257, -0.028, 0.0], [1.0, 0.0, 0.0, 1.0])
_T_LA_WR = _tf([-0.1349, 0.0052, 0.0], [1.0, 0.0, 0.0, -1.0])
_T_WR_GR = _tf([0.0, -0.0611, 0.0181], [0.0172091, -0.0172091, 0.706897, 0.706897])
_T_GR_CAM = _tf([-0.00296, 0.05436, -0.05310], [0.958280, -0.285792, -0.001556, 0.004547])

# Tool Center Point (gripperframe between the jaws in so101.xml)
TCP_LOCAL_M = np.array([0.012, -0.000218, -0.098127, 1.0], dtype=float)


def verify_calibration_hash(path: Path = CALIBRATION_PATH) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != EXPECTED_CALIBRATION_SHA256:
        raise RuntimeError(
            f"Empreinte de calibration modifiée ({digest[:12]} != {EXPECTED_CALIBRATION_SHA256[:12]}). "
            "Arrêt par sécurité."
        )
    return digest


def fk_gripper_frame(q_deg_5: np.ndarray) -> np.ndarray:
    """Return 4x4 pose of the gripper body in SO-101 base frame (meters)."""
    q = np.deg2rad(np.asarray(q_deg_5[:5], dtype=float))
    return (
        _T_BASE_SH
        @ _rotz(q[0])
        @ _T_SH_UA
        @ _rotz(q[1])
        @ _T_UA_LA
        @ _rotz(q[2])
        @ _T_LA_WR
        @ _rotz(q[3])
        @ _T_WR_GR
        @ _rotz(q[4])
    )


def fk_tcp_mm(q_deg_5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (tcp_xyz_mm, gripper_R_3x3) in SO-101 base frame."""
    T = fk_gripper_frame(q_deg_5)
    tcp_m = (T @ TCP_LOCAL_M)[:3]
    return tcp_m * 1000.0, T[:3, :3].copy()


def fk_camera_frame(q_deg_5: np.ndarray) -> np.ndarray:
    """Return 4x4 pose of wrist camera in SO-101 base frame (meters)."""
    return fk_gripper_frame(q_deg_5) @ _T_GR_CAM


@dataclass(frozen=True)
class IKResult:
    state_deg: np.ndarray       # Expected 5-DOF measured joint angles (deg)
    command_deg: np.ndarray     # Gravity-compensated 5-DOF motor commands (deg)
    tcp_base_mm: np.ndarray     # Achieved TCP in base frame (mm)
    pos_error_mm: float         # 3D position residual (mm)
    yaw_clamped: bool           # True if wrist_roll hit soft [-174.5, +174.5] safety clamp


class SO101CalibratedIK:
    """3D Cartesian IK anchored on the validated (0, 0) physical grasp pose."""

    def __init__(self, paper_yaw_deg: float = DEFAULT_PAPER_YAW_DEG):
        self.paper_yaw_deg = float(paper_yaw_deg)
        self.origin_state_deg = ORIGIN_GRASP_STATE_5.copy()
        self.origin_tcp_mm, self.origin_R = fk_tcp_mm(self.origin_state_deg)
        self.observe_state_deg = ORIGIN_OBSERVE_STATE_5.copy()
        self.observe_tcp_mm, self.observe_R = fk_tcp_mm(self.observe_state_deg)

        # Paper-to-base 2D rotation:
        # Nominal: +Y_table (away) -> +x_base ; +X_table (right) -> -y_base
        phi = np.deg2rad(self.paper_yaw_deg)
        c, s = float(np.cos(phi)), float(np.sin(phi))
        # u_nom = [Y_table, -X_table]; d_base_xy = u_nom @ [[c, s], [-s, c]]
        self._R_paper_xy = np.array([[c, s], [-s, c]], dtype=float)

    def table_to_base_mm(self, x_mm: float, y_mm: float, z_mm: float = 0.0) -> np.ndarray:
        """Convert (X_right, Y_away, Z_up) in mm from paper frame to SO-101 base mm."""
        u_nom = np.array([float(y_mm), -float(x_mm)], dtype=float)
        d_xy = u_nom @ self._R_paper_xy
        return self.origin_tcp_mm + np.array([d_xy[0], d_xy[1], float(z_mm)], dtype=float)

    def base_to_table_mm(self, base_xyz_mm: np.ndarray) -> np.ndarray:
        """Inverse of table_to_base_mm: base mm -> (X_right, Y_away, Z_up) mm."""
        diff = np.asarray(base_xyz_mm, dtype=float) - self.origin_tcp_mm
        u_nom = diff[:2] @ self._R_paper_xy.T
        y_mm = float(u_nom[0])
        x_mm = float(-u_nom[1])
        z_mm = float(diff[2])
        return np.array([x_mm, y_mm, z_mm], dtype=float)

    def check_workspace(self, x_mm: float, y_mm: float, z_mm: float = 0.0) -> np.ndarray:
        if not np.isfinite([x_mm, y_mm, z_mm]).all():
            raise ValueError("Coordonnées cibles non finies.")
        if not (WORKSPACE_X_MM[0] <= x_mm <= WORKSPACE_X_MM[1]):
            raise ValueError(f"X={x_mm:.1f} mm hors zone autorisée {WORKSPACE_X_MM} mm.")
        if not (WORKSPACE_Y_MM[0] <= y_mm <= WORKSPACE_Y_MM[1]):
            raise ValueError(f"Y={y_mm:.1f} mm hors zone autorisée {WORKSPACE_Y_MM} mm.")
        if not (WORKSPACE_Z_MM[0] <= z_mm <= WORKSPACE_Z_MM[1]):
            raise ValueError(f"Z={z_mm:.1f} mm hors hauteur autorisée {WORKSPACE_Z_MM} mm.")
        target_base_mm = self.table_to_base_mm(x_mm, y_mm, z_mm)
        radius = float(np.hypot(target_base_mm[0], target_base_mm[1]))
        if not (WORKSPACE_RADIUS_BASE_MM[0] <= radius <= WORKSPACE_RADIUS_BASE_MM[1]):
            raise ValueError(f"Rayon polaire {radius:.1f} mm hors enveloppe sécurisée {WORKSPACE_RADIUS_BASE_MM} mm.")
        return target_base_mm

    def solve(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        initial_deg: np.ndarray | None = None,
        lock_wrist_roll_deg: float | None = None,
    ) -> IKResult:
        """Solve 5-DOF IK for table target (x_mm, y_mm, z_mm) with vertical grasp orientation.

        At (0, 0, 0, yaw_delta_deg=0), returns the exact validated reference grasp pose.
        """
        target_base_mm = self.check_workspace(x_mm, y_mm, z_mm)
        target_m = target_base_mm / 1000.0

        # Desired world orientation: rotate reference gripper orientation around world Z by yaw_delta_deg
        yaw_rad = np.deg2rad(float(yaw_delta_deg))
        Rz_world = _rotz(yaw_rad)[:3, :3]
        R_des = Rz_world @ self.origin_R

        # Smart seed: adjust pan analytically from polar angle difference
        if initial_deg is None:
            pan0 = float(np.rad2deg(np.arctan2(self.origin_tcp_mm[1], self.origin_tcp_mm[0])
                                    - np.arctan2(target_base_mm[1], target_base_mm[0])))
            q = self.origin_state_deg.copy()
            q[0] = np.clip(self.origin_state_deg[0] + pan0, SAFE_JOINT_LOW_DEG[0], SAFE_JOINT_HIGH_DEG[0])
            q[4] = np.clip(
                self.origin_state_deg[4] + (q[0] - self.origin_state_deg[0]) + yaw_delta_deg,
                SAFE_JOINT_LOW_DEG[4],
                SAFE_JOINT_HIGH_DEG[4],
            )
        else:
            q = np.asarray(initial_deg[:5], dtype=float).copy()

        if lock_wrist_roll_deg is not None:
            q[4] = float(np.clip(lock_wrist_roll_deg, SAFE_JOINT_LOW_DEG[4], SAFE_JOINT_HIGH_DEG[4]))

        eps_deg = 1e-3
        yaw_clamped = False

        # Damped Least-Squares 6D Newton-Raphson (exact 3D position + 2D verticality + yaw)
        # Note: SO-101 has 5 DOF, so roll about horizontal axis perpendicular to arm plane is structural;
        # weighting z_gr verticality (pitch) + world yaw gives clean, unique convergence in < 15 steps.
        for _ in range(45):
            T = fk_gripper_frame(q)
            pos_m = (T @ TCP_LOCAL_M)[:3]
            R = T[:3, :3]

            pos_err = target_m - pos_m  # 3D meters
            if lock_wrist_roll_deg is not None:
                # Constrain only gripper vertical axis z_gr == R_des[:, 2] (4 DOF for 4 active joints)
                ori_err = np.cross(R[:, 2], R_des[:, 2]) * 0.18
                active = [0, 1, 2, 3]
            else:
                # Full 3D orientation error (softly weighted around unreachable lateral roll axis)
                ori_err = 0.5 * (
                    np.cross(R[:, 0], R_des[:, 0]) * 0.10
                    + np.cross(R[:, 1], R_des[:, 1]) * 0.10
                    + np.cross(R[:, 2], R_des[:, 2]) * 0.22
                )
                active = [0, 1, 2, 3, 4]

            err = np.r_[pos_err, ori_err]
            if np.linalg.norm(pos_err) < 5e-5 and np.linalg.norm(ori_err) < 5e-4:
                break

            J = np.zeros((6, len(active)), dtype=float)
            for col, idx in enumerate(active):
                qp = q.copy()
                qp[idx] += eps_deg
                Tp = fk_gripper_frame(qp)
                dp = ((Tp @ TCP_LOCAL_M)[:3] - pos_m) / eps_deg
                Rp = Tp[:3, :3]
                # Numerical derivative of residual (-err)
                if lock_wrist_roll_deg is not None:
                    ori_p = np.cross(Rp[:, 2], R_des[:, 2]) * 0.18
                else:
                    ori_p = 0.5 * (
                        np.cross(Rp[:, 0], R_des[:, 0]) * 0.10
                        + np.cross(Rp[:, 1], R_des[:, 1]) * 0.10
                        + np.cross(Rp[:, 2], R_des[:, 2]) * 0.22
                    )
                do = (ori_err - ori_p) / eps_deg
                J[:, col] = np.r_[dp, do]

            # Damped least squares step in degrees
            JJt = J @ J.T + np.diag([1e-8, 1e-8, 1e-8, 2e-5, 2e-5, 2e-5])
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -8.0, +8.0)

            for col, idx in enumerate(active):
                proposed = q[idx] + dq[col]
                clipped = float(np.clip(proposed, SAFE_JOINT_LOW_DEG[idx], SAFE_JOINT_HIGH_DEG[idx]))
                if idx == 4 and abs(proposed - clipped) > 0.05:
                    yaw_clamped = True
                q[idx] = clipped

            # If wrist_roll saturated on [-174.5, +174.5], re-solve position cleanly with wrist_roll locked
            if yaw_clamped and lock_wrist_roll_deg is None:
                res = self.solve(
                    x_mm=x_mm,
                    y_mm=y_mm,
                    z_mm=z_mm,
                    yaw_delta_deg=yaw_delta_deg,
                    initial_deg=q,
                    lock_wrist_roll_deg=q[4],
                )
                return replace(res, yaw_clamped=True)

        achieved_mm, _ = fk_tcp_mm(q)
        pos_error_mm = float(np.linalg.norm(achieved_mm - target_base_mm))
        if pos_error_mm > 1.0:
            raise ValueError(
                f"IK non convergée pour ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm : "
                f"erreur résiduelle {pos_error_mm:.2f} mm."
            )

        # Blend gravity feedforward smoothly with height (full on table z=0, tapered in high hover)
        ff_scale = float(np.clip(1.0 - max(0.0, z_mm) / 140.0, 0.35, 1.0))
        cmd = np.clip(
            q + GRASP_FEEDFORWARD_DEG * ff_scale,
            SAFE_JOINT_LOW_DEG - np.array([0.0, 0.2, 0.0, 0.0, 0.3]),
            SAFE_JOINT_HIGH_DEG,
        )

        return IKResult(
            state_deg=q,
            command_deg=cmd,
            tcp_base_mm=achieved_mm,
            pos_error_mm=pos_error_mm,
            yaw_clamped=yaw_clamped,
        )

    def solve_observe(
        self,
        x_mm: float,
        y_mm: float,
        z_extra_mm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        initial_deg: np.ndarray | None = None,
    ) -> IKResult:
        """Solve 5-DOF IK for the validated wrist-camera observation pose translated by (x_mm, y_mm).

        At (0, 0), reproduces ORIGIN_OBSERVE_STATE_5 (wrist_roll ~ -85 deg, pitched ~61 deg)
        where the physical HBV HD CAMERA points directly at the puff on (0, 0).
        """
        self.check_workspace(x_mm, y_mm, max(0.0, z_extra_mm))
        u_nom = np.array([float(y_mm), -float(x_mm)], dtype=float)
        d_xy = u_nom @ self._R_paper_xy
        target_base_mm = self.observe_tcp_mm + np.array([d_xy[0], d_xy[1], float(z_extra_mm)], dtype=float)
        target_m = target_base_mm / 1000.0

        Rz_world = _rotz(np.deg2rad(float(yaw_delta_deg)))[:3, :3]
        R_des = Rz_world @ self.observe_R

        if initial_deg is None:
            pan_delta = float(
                np.rad2deg(
                    np.arctan2(self.observe_tcp_mm[1], self.observe_tcp_mm[0])
                    - np.arctan2(target_base_mm[1], target_base_mm[0])
                )
            )
            q = self.observe_state_deg.copy()
            q[0] = float(np.clip(q[0] + pan_delta, SAFE_JOINT_LOW_DEG[0], SAFE_JOINT_HIGH_DEG[0]))
            q[4] = float(np.clip(q[4] + pan_delta + yaw_delta_deg, SAFE_JOINT_LOW_DEG[4], SAFE_JOINT_HIGH_DEG[4]))
        else:
            q = np.asarray(initial_deg[:5], dtype=float).copy()

        eps_deg = 1e-3
        for _ in range(45):
            T = fk_gripper_frame(q)
            pos_m = (T @ TCP_LOCAL_M)[:3]
            R = T[:3, :3]
            pos_err = target_m - pos_m
            ori_err = 0.5 * (
                np.cross(R[:, 0], R_des[:, 0]) * 0.14
                + np.cross(R[:, 1], R_des[:, 1]) * 0.14
                + np.cross(R[:, 2], R_des[:, 2]) * 0.18
            )
            err = np.r_[pos_err, ori_err]
            if np.linalg.norm(pos_err) < 5e-5 and np.linalg.norm(ori_err) < 8e-4:
                break

            J = np.zeros((6, 5), dtype=float)
            for idx in range(5):
                qp = q.copy()
                qp[idx] += eps_deg
                Tp = fk_gripper_frame(qp)
                dp = ((Tp @ TCP_LOCAL_M)[:3] - pos_m) / eps_deg
                Rp = Tp[:3, :3]
                ori_p = 0.5 * (
                    np.cross(Rp[:, 0], R_des[:, 0]) * 0.14
                    + np.cross(Rp[:, 1], R_des[:, 1]) * 0.14
                    + np.cross(Rp[:, 2], R_des[:, 2]) * 0.18
                )
                J[:, idx] = np.r_[dp, (ori_err - ori_p) / eps_deg]

            JJt = J @ J.T + np.diag([1e-8, 1e-8, 1e-8, 3e-5, 3e-5, 3e-5])
            dq = np.clip(J.T @ np.linalg.solve(JJt, err), -8.0, +8.0)
            q = np.clip(q + dq, SAFE_JOINT_LOW_DEG, SAFE_JOINT_HIGH_DEG)

        achieved_mm, _ = fk_tcp_mm(q)
        pos_error_mm = float(np.linalg.norm(achieved_mm - target_base_mm))
        if pos_error_mm > 1.5:
            raise ValueError(f"IK observation non convergée pour ({x_mm:.1f}, {y_mm:.1f}) mm : {pos_error_mm:.2f} mm.")
        cmd = np.clip(q + GRASP_FEEDFORWARD_DEG * 0.5, SAFE_JOINT_LOW_DEG, SAFE_JOINT_HIGH_DEG)
        return IKResult(
            state_deg=q,
            command_deg=cmd,
            tcp_base_mm=achieved_mm,
            pos_error_mm=pos_error_mm,
            yaw_clamped=False,
        )


if __name__ == "__main__":
    ik = SO101CalibratedIK()
    r0 = ik.solve(0.0, 0.0, 0.0)
    print("IK (0, 0, 0) state_deg :", np.round(r0.state_deg, 3).tolist())
    print("Reference    state_deg :", np.round(ORIGIN_GRASP_STATE_5, 3).tolist())
    print("Position residual (mm) :", round(r0.pos_error_mm, 5))
