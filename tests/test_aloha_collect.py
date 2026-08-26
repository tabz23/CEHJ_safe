"""Smoke: one collect-mode episode each for aloha-agilex and piper.

Validates: aloha split-URDF body features (FK consistency assert runs at
extractor init), DualArmKinematics state encoding, empirical T_base2world,
and the tick-chunked collect path end to end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from main.network.encoder import HoloBrainEncoder
from main.train.buffer import StepBuffer
from main.train.collect import CKPT_DIR, make_kinematics
from main.train.config import FrozenConfig
from main.train.rollout import RolloutController

out = Path("/tmp/aloha_smoke")
buf = StepBuffer(out / "buffer", capacity=4 * 452,
                 header=FrozenConfig().to_dict())
encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")

for ep, emb in enumerate(["aloha-agilex", "piper"]):
    cfg = FrozenConfig(embodiment=emb, task="stack_blocks_two",
                       randomize_scenes=False)
    kin = make_kinematics(CKPT_DIR, emb)
    ro = RolloutController(cfg, seed=0, mode="collect", encoder=encoder,
                           kin=kin, buf=buf, episode_id=ep,
                           max_steps=1500)
    trace = ro.run()
    print(f"{emb}: success={trace['success']} steps={trace['n_physics']} "
          f"buf={len(buf)}")

buf.flush()
arr = buf.arrays
print("state_tokens:", arr["state_tokens"].shape, " h:", arr["h"].shape)
h = np.asarray(arr["h"][: len(buf)])
assert np.isfinite(h).all(), "h must be finite on every stored step"
print("h range:", float(h.min()), float(h.max()))
print("ALOHA SMOKE PASSED")
