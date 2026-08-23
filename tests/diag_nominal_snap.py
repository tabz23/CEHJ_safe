"""Diagnose arm snaps in nominal eval: log per-tick qpos, find jumps."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.train.config import FrozenConfig
from main.train.rollout import RolloutController

cfg = FrozenConfig(task="stack_blocks_two", embodiment="franka-panda",
                   obstacle_mode="on_path", replan_k=60)

ro = RolloutController(cfg, seed=1000, mode="nominal", k=60)

qpos_log = []
orig_capture = ro._capture_tick

def capture_with_qpos():
    obs = orig_capture()
    qpos_log.append(obs["joint_state"][0, 0].numpy().copy())
    return obs

ro._capture_tick = capture_with_qpos
trace = ro.run()
q = np.array(qpos_log)          # [n_ticks, 16]
dq = np.abs(np.diff(q, axis=0)).max(axis=1)
t = np.array(trace["t"])
print("ticks:", len(q), "success:", trace["success"])
order = np.argsort(dq)[-10:]
print("largest per-tick qpos jumps (rad):")
for i in sorted(order):
    print(f"  tick {i:4d}  t={t[i]:5.2f}s  |dq|max={dq[i]:.4f}  arm={'L' if np.argmax(np.abs(q[i+1]-q[i])) < 8 else 'R'}")
