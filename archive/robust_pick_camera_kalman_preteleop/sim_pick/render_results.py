"""Render held-out MuJoCo grasp attempts from a trained SAC policy."""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import SAC

from grasp_env import VapeGraspEnv

ROOT=Path(__file__).resolve().parent


def annotate(frame,text):
    im=Image.fromarray(frame)
    draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,640,48),fill=(20,24,32))
    draw.text((12,8),text,fill='white')
    return np.asarray(im)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model=SAC.load(ROOT/'outputs/best',device='cpu')
    # First six consecutive held-out seeds, not handpicked successes.
    seeds=list(range(50000,50006))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=args.output or ROOT/'outputs/simulation_attempts'/f'heldout_preview_{stamp}.mp4'
    output.parent.mkdir(parents=True,exist_ok=True)
    env=VapeGraspEnv(render_mode='rgb_array')
    with imageio.get_writer(output,fps=30,codec='libx264',quality=8) as writer:
        for index,seed in enumerate(seeds):
            env.frames=None
            obs,_=env.reset(seed=seed)
            before=env.render()
            action,_=model.predict(obs,deterministic=True)
            env.frames=[];env.frame_counter=0
            info=env.step(action)[-1]
            frames=env.frames;env.frames=None
            label=f"SIMULATION - SO-101 / SAC | test {index+1}/6 | yaw {np.degrees(env.yaw):+.0f} deg"
            for _ in range(20):writer.append_data(annotate(before,label))
            for frame in frames:writer.append_data(annotate(frame,label))
            last=env.render()
            result=f"{'SIM SUCCESS' if info['success'] else 'SIM FAILURE'} | lift {info['lift']*100:.1f} cm | simulated jaw contacts"
            for _ in range(35):writer.append_data(annotate(last,result))
            if index==0:imageio.imwrite(output.with_suffix('.png'),annotate(last,result))
            print(seed,info,flush=True)
    env.close()
    print('Video:',output,flush=True)


if __name__=='__main__':main()
