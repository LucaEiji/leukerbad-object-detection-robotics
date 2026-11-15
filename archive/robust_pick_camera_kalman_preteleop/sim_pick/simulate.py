"""Run deterministic, hardware-free SO-101 grasp trials with structured logs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from grasp_env import VapeGraspEnv

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs/best.zip"


def run_trial(seed: int, policy: str = "best", checkpoint: Path = CHECKPOINT) -> dict:
    env = VapeGraspEnv(randomize=True)
    observation, _ = env.reset(seed=seed)
    model_path = None
    if policy == "best":
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint SAC manquant : {checkpoint}")
        model = SAC.load(checkpoint, device="cpu")
        action = model.predict(observation, deterministic=True)[0]
        model_path = str(checkpoint.resolve())
    elif policy == "neutral":
        action = np.zeros(4, dtype=np.float32)
    elif policy == "random":
        action = np.random.default_rng(seed).uniform(-1, 1, 4).astype(np.float32)
    else:
        raise ValueError(f"Politique inconnue : {policy}")
    _, _, terminated, truncated, info = env.step(action)
    result = {
        "schema": "puff_sort_simulation_attempt_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_accessed": False,
        "seed": int(seed),
        "policy": policy,
        "checkpoint": model_path,
        "units": {"length": "m", "joint_angle": "rad"},
        "axes": {"simulation": ["x_forward", "y_left", "z_up"], "yaw": "world_z"},
        "initial_object": {"xy_m": env.objxy.tolist(), "yaw_rad": env.yaw},
        "action": np.asarray(action, dtype=float).tolist(),
        "planned_grasp": {"xyz_m": info["grasp_xyz"], "yaw_rad": info["grasp_yaw"]},
        "termination": {"terminated": bool(terminated), "truncated": bool(truncated)},
        "result": {key: info[key] for key in ("success", "held", "lift", "reward")},
        "trace": env.trace,
        "success_definition": "lift > 0.04 m with contact on gripper and moving jaw",
        "physical_success_proven": False,
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=("best", "neutral", "random"), default="best")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    attempt = run_trial(args.seed, args.policy, args.checkpoint)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or ROOT / "outputs/simulation_attempts" / f"seed_{args.seed:06d}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(attempt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **attempt["result"]}, indent=2))


if __name__ == "__main__":
    main()
