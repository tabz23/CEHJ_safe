"""Filter smoke on the tick-chunked vanilla nominal (Phase 4).

Filtered rollout with untrained actor/critics on one scene: verifies the
Q(s, a_nom) trigger path runs end-to-end — a_nom assembly from the live
plan, per-tick V trace, 3-tick intervention blocks, forced replan after
each intervention, and the mode-switch counter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.network.encoder import HoloBrainEncoder, RobotInjection
from main.network.heads import PolicyEncoder, TokenActor, TwinCritic
from main.network.trunk import GeometricTrunk
from main.train.collect import CKPT_DIR, make_kinematics
from main.train.config import FrozenConfig
from main.train.rollout import RolloutController

cfg = FrozenConfig(task="stack_blocks_two", embodiment="franka-panda",
                   obstacle_mode="on_path", randomize_scenes=False)

encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
kin = make_kinematics(CKPT_DIR, cfg.embodiment)
policy_enc = PolicyEncoder(
    RobotInjection().cuda().eval(), GeometricTrunk().cuda().eval()
).cuda().eval()
actor = TokenActor().cuda().eval()
critics = TwinCritic().cuda().eval()

ro = RolloutController(cfg, seed=0, mode="filtered", encoder=encoder, kin=kin,
                       policy_enc=policy_enc, actor=actor, critics=critics)
trace = ro.run()
print("success:", trace["success"])
print("ticks with V:", len(trace["V"]), "interventions:", trace.get("interventions"),
      "mode switches:", trace.get("mode_switches"))
assert len(trace["V"]) > 0, "filter never evaluated"
print("FILTER SMOKE PASSED")
