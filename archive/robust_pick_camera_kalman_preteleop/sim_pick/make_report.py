"""Plot validation progress and summarize the held-out evaluation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root=Path(__file__).resolve().parent/'outputs'
progress=json.loads((root/'progress.json').read_text())
baseline=json.loads((root/'random_baseline.json').read_text())
heldout=json.loads((root/'heldout_evaluation.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].plot([p['steps'] for p in progress],[100*p['success_rate'] for p in progress],marker='o',color='#2379ac')
axes[0].axhline(100*baseline['success_rate'],linestyle='--',color='#888',label='Random grasp baseline (100 trials)')
axes[0].set(xlabel='RL training episodes',ylabel='Validation success (%)',ylim=(0,105),title='SAC grasp-pose learning (24 validation trials)')
axes[0].legend(fontsize=8)
axes[1].bar(['Random\n100 trials','Trained SAC\n200 held-out trials'],[100*baseline['success_rate'],100*heldout['success_rate']],color=['#9aa8b0','#2379ac'])
axes[1].set(ylabel='Simulated grasp success (%)',ylim=(0,110),title='Simulation only / privileged object pose')
for i,v in enumerate([baseline['success_rate'],heldout['success_rate']]):axes[1].text(i,100*v+2,f'{100*v:.1f}%',ha='center')
fig.savefig(root/'training_summary.png',dpi=160)
summary={
    'training_episodes':progress[-1]['steps'],
    'training_seconds':progress[-1]['seconds'],
    'random_baseline_trials':len(baseline['rows']),
    'random_baseline_successes':sum(x['success'] for x in baseline['rows']),
    'heldout_trials':len(heldout['rows']),
    'heldout_successes':sum(x['success'] for x in heldout['rows']),
    'heldout_success_rate':heldout['success_rate'],
    'mean_lift_cm':100*heldout['mean_lift'],
    'limitations':['Simulation only','Privileged object position/orientation, not images',
                   'Learned grasp pose; programmed IK approach/close/lift',
                   'Object yaw between -57 and +57 degrees; x/y jitter +/-5 mm',
                   'No real-robot transfer tested'],
}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
