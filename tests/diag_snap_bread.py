"""Locate the arm snap in place_bread_basket: log per-physics-step qpos and
drive targets; report every discontinuity with the plan log around it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.envs.controller import TickChunkedController  # noqa: E402
from main.train.collect import make_training_env  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402

cfg = FrozenConfig(task="place_bread_basket", embodiment="franka-panda",
                   randomize_scenes=False)
env = make_training_env(cfg, seed=0)
ctrl = TickChunkedController(env)
ctrl.attach()

qpos_log = []      # measured, per physics step
tgt_log = []       # commanded drive target, per physics step
orig_step = env.task.scene.step
state = {"i": 0}

robot = env.robot

def hooked():
    orig_step()
    state["i"] += 1
    if state["i"] > 4500:
        raise RuntimeError("timeout")
    qpos_log.append(np.array(robot.get_left_arm_real_jointState()
                             + robot.get_right_arm_real_jointState()))
    tgt_log.append(np.array(robot.get_left_arm_jointState()
                            + robot.get_right_arm_jointState()))

env.task.scene.step = hooked
try:
    env.task.play_once()
except Exception as exc:
    print("play_once ended:", type(exc).__name__, exc)

q = np.array(qpos_log)
tg = np.array(tgt_log)
dq = np.abs(np.diff(q, axis=0)).max(axis=1)
dgap = np.abs(tg[1:] - q[:-1]).max(axis=1)   # target vs measured at prev step
print(f"physics steps: {len(q)}")
order = np.argsort(dq)[-5:]
for i in sorted(order):
    print(f"  measured |dq|max at step {i} (t={i/250:.2f}s): {dq[i]:.4f} rad")
order = np.argsort(dgap)[-5:]
for i in sorted(order):
    print(f"  target-measured gap at step {i} (t={i/250:.2f}s): {dgap[i]:.4f} rad")
env.close()
