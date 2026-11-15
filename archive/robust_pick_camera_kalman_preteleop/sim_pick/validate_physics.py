"""Check determinism, real contacts, release/drop and numeric stability."""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from grasp_env import VapeGraspEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    env=VapeGraspEnv(randomize=False)
    report={}
    a,_=env.reset(seed=123);state=env.data.qpos.copy()
    b,_=env.reset(seed=123)
    assert np.array_equal(a,b) and np.allclose(state,env.data.qpos)
    report['deterministic_reset']=True
    assert env.model.neq==0
    assert env.model.jnt_type[6]==mujoco.mjtJoint.mjJNT_FREE
    report['no_weld_or_attachment']=True
    env.reset(seed=0)
    info=env.step([0,0,0,0])[-1]
    assert info['success'] and info['held'] and info['lift']>.04
    report['physical_contact_lift']=info
    initial=float(env.data.xpos[env.obid,2])
    env._run(env.data.ctrl[:5].copy(),.7,1.5)
    released=float(env.data.xpos[env.obid,2])
    assert released<initial-.03
    report['object_falls_when_jaws_open']={'before_height':initial,'after_height':released}
    env.reset(seed=0)
    miss=env.step([1,1,1,0])[-1]
    assert not miss['success']
    report['miss_is_not_a_success']=miss
    rng=np.random.default_rng(678)
    for seed in range(20):
        env.reset(seed=seed)
        env.step(rng.uniform(-1,1,4))
        assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
    report['finite_physics_random_trials']=20
    env.close()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.output or Path(__file__).resolve().parent/'outputs/validation_runs'/f'physics_checks_{stamp}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
