"""Repro the franka 5 Hz filter crash with a full traceback."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.network.encoder import HoloBrainEncoder, RobotInjection
from main.network.heads import PolicyEncoder, TokenActor, TwinCritic
from main.network.trunk import GeometricTrunk
from main.train.collect import CKPT_DIR, make_kinematics
from main.train.config import FrozenConfig
from main.train.rollout import RolloutController

cfg = FrozenConfig(task="place_empty_cup", embodiment="franka-panda",
                   obstacle_mode="on_path")

encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
kin = make_kinematics(CKPT_DIR, cfg.embodiment)
policy_enc = PolicyEncoder(
    RobotInjection().cuda().eval(), GeometricTrunk().cuda().eval()
).cuda().eval()
actor = TokenActor().cuda().eval()
critics = TwinCritic().cuda().eval()

ro = RolloutController(cfg, seed=0, mode="filtered", encoder=encoder, kin=kin,
                       policy_enc=policy_enc, actor=actor, critics=critics, k=50)
trace = ro.run()
print("success:", trace["success"])
print("windows:", len(trace["V"]), "interventions:", trace.get("interventions"))
