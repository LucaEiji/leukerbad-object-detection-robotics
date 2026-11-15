"""Train SAC on the physical SO-101 grasp task; no demonstrations or hardware."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from grasp_env import VapeGraspEnv

ROOT = Path(__file__).resolve().parent


def evaluate(model, seeds, random=False):
    env = VapeGraspEnv()
    rng = np.random.default_rng(777)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        action = rng.uniform(-1, 1, 4) if random else model.predict(obs, deterministic=True)[0]
        _, reward, _, _, info = env.step(action)
        rows.append(dict(seed=int(seed), yaw=env.yaw, action=np.asarray(action).tolist(), **info))
    env.close()
    return dict(success_rate=float(np.mean([r['success'] for r in rows])),
                mean_lift=float(np.mean([r['lift'] for r in rows])),
                mean_reward=float(np.mean([r['reward'] for r in rows])), rows=rows)


class Progress(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.best = -1
        self.begin = time.monotonic()
        self.scores = []

    def _on_step(self):
        if self.num_timesteps % 500 == 0:
            score = evaluate(self.model, range(10000, 10024))
            small = dict(steps=self.num_timesteps, seconds=time.monotonic()-self.begin,
                         **{k:v for k,v in score.items() if k!='rows'})
            self.scores.append(small)
            (self.output/'progress.json').write_text(json.dumps(self.scores, indent=2))
            self.model.save(self.output/'latest')
            if score['success_rate'] > self.best:
                self.best = score['success_rate']
                self.model.save(self.output/'best')
            print(json.dumps(small), flush=True)
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=8000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=ROOT/'outputs';output.mkdir(exist_ok=True)
    env=VapeGraspEnv()
    check_env(env, skip_render_check=True)
    print('Gymnasium environment checks passed.',flush=True)
    if not (output/'random_baseline.json').exists():
        baseline=evaluate(None,range(20000,20100),random=True)
        (output/'random_baseline.json').write_text(json.dumps(baseline,indent=2))
        print('Random baseline:',baseline['success_rate'],flush=True)
    monitor=Monitor(env,str(output/'training'))
    if args.resume:
        model=SAC.load(output/'latest',env=monitor,device='cpu')
    else:
        model=SAC('MlpPolicy',monitor,learning_rate=0.0005,buffer_size=30000,
                  learning_starts=256,batch_size=128,tau=.02,gamma=0.,
                  train_freq=1,gradient_steps=1,ent_coef='auto_0.1',
                  policy_kwargs=dict(net_arch=[64,64]),seed=42,device='cpu',verbose=0)
    model.learn(total_timesteps=args.steps,callback=Progress(output),reset_num_timesteps=not args.resume)
    model.save(output/'latest')
    best=SAC.load(output/'best',device='cpu')
    heldout=evaluate(best,range(50000,50200))
    (output/'heldout_evaluation.json').write_text(json.dumps(heldout,indent=2))
    print('HELD-OUT:',{k:v for k,v in heldout.items() if k!='rows'},flush=True)
    env.close()


if __name__=='__main__':main()
