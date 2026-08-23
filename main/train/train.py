"""Stage 2 — HJ-SAC training.

Objective (h computed, never learned; V is the only learned object and
represents the backward reachable set):

    target  = (1-gamma)*h(s) + gamma*min{ h(s), Q_target(s', a') },  a' ~ pi(s')
    loss_Q  = MSE(Q1, target) + MSE(Q2, target)
    loss_pi = -Q_min(s, pi(s)) + alpha * logp

Annealing: gamma -> 1 (so V doesn't sit at h because the recursion never
looks far — watch the h - V gap open), alpha -> 0 aggressively.

NOTE on the Bellman operator: the target is deliberately the HARD Bellman
(no -alpha*logp_n) while the actor loss keeps the entropy term. alpha is an
exploration regularizer on the actor, NOT a soft-value temperature: the
soft value is alpha*logsumexp(Q/alpha) >= max_a Q, and higher Q means
safer, so a soft target would make the value function optimistic about
safety — the dangerous direction for an avoid problem. logp_n is computed
for logging and intentionally unused in the target.

Target entropy note: if alpha is ever auto-tuned, the target is per-sample
-n_act, not -dim(A).

Usage (from CEHJ_safe/):
    python main/train.py --data data/smoke --grad-steps 300 --eval-every 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# offline by default; online when the user provides WANDB_API_KEY
if not os.environ.get("WANDB_API_KEY"):
    os.environ.setdefault("WANDB_MODE", "offline")

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "main.train"
CEHJ_ROOT = Path(__file__).resolve().parents[2]

from main.train.buffer import StepBuffer  # noqa: E402
from main.train.collect import CKPT_DIR  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402
from main.network.encoder import HoloBrainEncoder, RobotInjection  # noqa: E402
from main.network.heads import PolicyEncoder, TokenActor, TwinCritic  # noqa: E402
from main.network.trunk import GeometricTrunk  # noqa: E402


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype).cuda()


class Trainer:
    def __init__(self, cfg: FrozenConfig, data_root: Path, run_dir: Path):
        self.cfg = cfg
        # resolve NOW: RoboTwin chdirs during env creation; a relative run
        # dir would scatter checkpoint/eval files under the RoboTwin root
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.json")

        self.buf = StepBuffer(Path(data_root) / "buffer")  # capacity from header
        print(f"buffer: {len(self.buf)} steps")

        self.injection = RobotInjection().cuda()
        self.trunk = GeometricTrunk().cuda()
        self.policy_enc = PolicyEncoder(self.injection, self.trunk).cuda()
        self.actor = TokenActor().cuda()
        # V lives in h*h_scale units; T is physical metres in the config, so
        # scale it to keep the softmin sharpness physically meaningful
        T = cfg.softmin_T * float(getattr(cfg, "h_scale", 1.0))
        self.critics = TwinCritic(temperature=T).cuda()
        self.critics_targ = TwinCritic(temperature=T).cuda()
        self.critics_targ.load_state_dict(self.critics.state_dict())
        # target ENCODER too: Polyak covers nothing if enc_n comes from the
        # live trunk/injection — the bootstrap target would move every step
        import copy as _copy

        self.injection_targ = _copy.deepcopy(self.injection)
        self.trunk_targ = _copy.deepcopy(self.trunk)
        self.policy_enc_targ = PolicyEncoder(self.injection_targ, self.trunk_targ)
        self.target_params = (
            list(self.critics_targ.parameters())
            + list(self.injection_targ.parameters())
            + list(self.trunk_targ.parameters())
        )
        self.live_params = (
            list(self.critics.parameters())
            + list(self.injection.parameters())
            + list(self.trunk.parameters())
        )
        for p in self.target_params:
            p.requires_grad_(False)

        self.opt_c = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_e = torch.optim.Adam(
            list(self.injection.parameters()) + list(self.trunk.parameters()),
            lr=cfg.lr,
        )

        # eval encoder built once (safetensors load + Swin init per eval
        # otherwise). Jlin/Jang/dtheta_max come from the buffer (stored per
        # step — a mixed-embodiment buffer cannot recompute them).
        self._eval_encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
        self.rng = np.random.RandomState(0)
        self.step = 0

    # ---------- schedules ----------
    def gamma(self) -> float:
        f = min(1.0, self.step / max(self.cfg.gamma_anneal_steps, 1))
        return self.cfg.gamma + f * (self.cfg.gamma_final - self.cfg.gamma)

    def alpha(self) -> float:
        f = min(1.0, self.step / max(self.cfg.alpha_anneal_steps, 1))
        return self.cfg.alpha_start + f * (self.cfg.alpha_final - self.cfg.alpha_start)

    # ---------- batch ----------
    def _encode(self, b: dict, suffix: str = "", target: bool = False):
        g = lambda k: b[k + suffix]
        enc_mod = self.policy_enc_targ if target else self.policy_enc
        enc = enc_mod(
            _t(g("state_tokens")), _t(g("body_tokens")), _t(g("link_pos")),
            _t(g("body_mask")).bool(), _t(g("joint_index"), torch.int64),
            _t(g("scene_tokens")), _t(g("scene_pos")),
        )
        # stored per-step (mixed embodiments cannot recompute from qpos)
        Jlin = _t(g("Jlin"))
        Jang = _t(g("Jang"))
        return enc, Jlin, Jang, _t(g("joint_index"), torch.int64)

    def grad_step(self) -> dict:
        cfg = self.cfg
        b = self.buf.sample(cfg.batch_size, self.rng)
        enc, Jlin, Jang, ji = self._encode(b)
        enc_n, Jlin_n, Jang_n, ji_n = self._encode(b, "_next", target=True)
        h = _t(b["h"])                                    # [B]
        dtheta = _t(b["dtheta"])                          # executed action
        dtheta_max = _t(b["dtheta_max"])                  # per-sample, padded
        dtheta_max_n = _t(b["dtheta_max_next"])
        gamma, alpha = self.gamma(), self.alpha()

        with torch.no_grad():
            a_n, logp_n, _ = self.actor(enc_n.body, ji_n, dtheta_max_n)
            q1_t, q2_t, _, _ = self.critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
            q_next = torch.minimum(q1_t, q2_t)
            target = (1 - gamma) * h + gamma * torch.minimum(h, q_next)
        # critic update
        q1, q2, v1, v2 = self.critics(enc, dtheta, Jlin, Jang, ji)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(set_to_none=True)
        self.opt_e.zero_grad(set_to_none=True)
        loss_q.backward()
        gn_c = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        gn_e = torch.nn.utils.clip_grad_norm_(
            list(self.injection.parameters()) + list(self.trunk.parameters()), 10.0
        )
        self.opt_c.step()
        self.opt_e.step()

        # actor update (encoder/critics frozen for this step; enc detached —
        # the critic-loss backward already consumed its graph)
        for p in list(self.injection.parameters()) + list(self.trunk.parameters()):
            p.requires_grad_(False)
        for p in self.critics.parameters():
            p.requires_grad_(False)
        from main.network.heads import Encoded as _Encoded

        enc_det = _Encoded(
            enc.body.detach(), enc.pos.detach(), enc.mask,
            enc.scene.detach(), enc.scene_pos.detach(), enc.scene_mask,
        )
        a_pi, logp, n_act = self.actor(enc_det.body, ji, dtheta_max)
        q1_pi, q2_pi, _, _ = self.critics(enc_det, a_pi, Jlin, Jang, ji)
        loss_pi = (-torch.minimum(q1_pi, q2_pi) + alpha * logp).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_pi.backward()
        gn_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        for p in list(self.injection.parameters()) + list(self.trunk.parameters()):
            p.requires_grad_(True)
        for p in self.critics.parameters():
            p.requires_grad_(True)

        # Polyak target update — critic heads AND the target encoder
        with torch.no_grad():
            for tp, sp in zip(self.target_params, self.live_params):
                tp.mul_(1 - cfg.tau).add_(cfg.tau * sp)

        self.step += 1
        qmin = torch.minimum(q1, q2)
        return {
            "loss_q": float(loss_q), "loss_pi": float(loss_pi),
            "alpha": alpha, "gamma": gamma,
            "mean_q": float(qmin.mean()), "mean_h": float(h.mean()),
            "mean_gap_h_q": float((h - qmin).mean()),
            "entropy": float(-logp.mean()),
            "gn_critic": float(gn_c), "gn_actor": float(gn_a),
            "gn_encoder": float(gn_e),
            "buffer": len(self.buf),
        }

    # ---------- eval ----------
    # NOTE: no global no_grad here — the eval probe runs cuRobo planning,
    # which needs autograd. run_eval_episode wraps model calls locally.
    def evaluate(self, wandb_run=None) -> dict:
        """Eval = the filtered trajectory only (nominal + safety filter),
        on a fresh random (task, embodiment, obstacle) scene per eval."""
        from main.train.eval_utils import run_eval_episode, save_eval_figure

        # a different scene per eval (deterministic sequence): fixed-seed
        # evals made every video identical, hiding placement sensitivity
        n_evals = getattr(self, "_n_evals", 0)
        self._n_evals = n_evals + 1
        metrics = {}
        for mode in ("filtered",):
            trace = run_eval_episode(
                self.cfg, n_evals, mode, self._eval_encoder, self.policy_enc,
                self.actor, self.critics, tag=f"_step{self.step}",
            )
            metrics[mode] = trace["metrics"]
            fig_path = save_eval_figure(trace, self.run_dir / "eval", mode, self.step)
            if wandb_run is not None:
                import wandb

                logs = {f"eval/{mode}_hv_fig": wandb.Image(str(fig_path))}
                # panel video (observer + white stats panel) — the filtered
                # trajectory is the nominal controller + safety filter
                if trace.get("video_path"):
                    logs[f"eval/{mode}_video"] = wandb.Video(
                        trace["video_path"], fps=5, format="mp4"
                    )
                wandb_run.log(logs, step=self.step)
        if wandb_run is not None:
            import wandb

            flat = {}
            for mode, m in metrics.items():
                for k, v in m.items():
                    flat[f"eval/{mode}/{k}"] = v
            wandb_run.log(flat, step=self.step)
        return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=CEHJ_ROOT / "data" / "smoke")
    p.add_argument("--run", type=Path, default=CEHJ_ROOT / "runs" / "smoke")
    p.add_argument("--grad-steps", type=int, default=204800,
                   help="200 epochs x 1024 by default")
    p.add_argument("--eval-every", type=int, default=1024,
                   help="grad steps per epoch (test runs: 50)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    args.data = args.data.resolve()
    args.run = args.run.resolve()  # RoboTwin chdirs during env creation

    cfg = FrozenConfig.load(args.data / "config.json")
    cfg.grad_steps = args.grad_steps
    cfg.eval_every = args.eval_every

    trainer = Trainer(cfg, args.data, args.run)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="cehj-hjsac", config=cfg.to_dict(), dir=str(args.run)
        )

    t0 = time.time()
    for step in range(cfg.grad_steps):
        m = trainer.grad_step()
        if step % 20 == 0:
            sps = (step + 1) / (time.time() - t0)
            print(
                f"step {step}  loss_q {m['loss_q']:.4f}  loss_pi {m['loss_pi']:.4f}  "
                f"q {m['mean_q']:.3f}  h {m['mean_h']:.3f}  gap {m['mean_gap_h_q']:.3f}  "
                f"{sps:.1f} steps/s"
            )
            if run is not None:
                run.log({**m, "steps_per_sec": sps}, step=trainer.step)
        if (step + 1) % cfg.eval_every == 0:
            trainer.evaluate(run)

    torch.save(
        {
            "injection": trainer.injection.state_dict(),
            "trunk": trainer.trunk.state_dict(),
            "actor": trainer.actor.state_dict(),
            "critics": trainer.critics.state_dict(),
        },
        args.run / "checkpoint.pt",
    )
    trainer.evaluate(run)
    if run is not None:
        run.finish()
    print("TRAIN DONE")


if __name__ == "__main__":
    main()
