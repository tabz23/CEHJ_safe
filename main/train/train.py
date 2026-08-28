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
from main.train.config import FrozenConfig, apply_embodiment_selection  # noqa: E402
from main.network.encoder import HoloBrainEncoder, RobotInjection  # noqa: E402
from main.network.heads import PolicyEncoder, TokenActor, TwinCritic  # noqa: E402
from main.network.trunk import GeometricTrunk  # noqa: E402


def _safe_wandb_log(wandb_run, logs: dict, step: int) -> None:
    """wandb logging must never crash training — the service process times
    out occasionally on slow networks (observed killing an eval sweep)."""
    try:
        wandb_run.log(logs, step=step)
    except Exception as exc:
        print(f"[train] wandb log failed at step {step} ({exc}); continuing")


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype).cuda()


class Trainer:
    def __init__(self, cfg: FrozenConfig, data_root: Path, run_dir: Path,
                 create_capacity: int | None = None):
        self.cfg = cfg
        # resolve NOW: RoboTwin chdirs during env creation; a relative run
        # dir would scatter checkpoint/eval files under the RoboTwin root
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.json")

        buf_dir = Path(data_root) / "buffer"
        if not (buf_dir / "header.json").exists():
            # DAgger from scratch: size the buffer across ALL rounds up front
            # (open_memmap allocates at fixed capacity; append asserts on it)
            if create_capacity is None:
                raise ValueError(
                    f"{buf_dir}: no buffer yet — collect first, or pass "
                    f"create_capacity for a DAgger run"
                )
            StepBuffer(buf_dir, capacity=create_capacity, header=cfg.to_dict())
        self.buf = StepBuffer(buf_dir)  # capacity from header
        # leave-one-out: the warmup buffer is SHARED across splits — restrict
        # sampling to the selected embodiment pool instead of collecting a
        # per-split buffer (phase 1 --leave-out X -> pool = all minus X;
        # phase 2 --only-embodiment X -> pool = {X})
        from main.train.rollout import EMBODIMENT_IDS

        allowed = {EMBODIMENT_IDS[e] for e in cfg.embodiment_choices
                   if e in EMBODIMENT_IDS}
        if 0 < len(allowed) < len(EMBODIMENT_IDS):
            self.buf.set_embodiment_filter(allowed)
            n_valid = len(self.buf._valid_transitions())
            if n_valid == 0:
                raise RuntimeError(
                    f"embodiment filter {sorted(allowed)} leaves no "
                    f"transitions in {buf_dir}"
                )
            print(f"buffer: {len(self.buf)} steps, "
                  f"{n_valid} valid transitions after embodiment filter")
        else:
            print(f"buffer: {len(self.buf)} steps")
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
        out = {
            "loss_q": float(loss_q), "loss_pi": float(loss_pi),
            "alpha": alpha, "gamma": gamma,
            "mean_q": float(qmin.mean()), "mean_h": float(h.mean()),
            "mean_gap_h_q": float((h - qmin).mean()),
            "entropy": float(-logp.mean()),
            "gn_critic": float(gn_c), "gn_actor": float(gn_a),
            "gn_encoder": float(gn_e),
            "buffer": len(self.buf),
        }
        # mode-collapse diagnostics (does the actor output the same action
        # everywhere?): diversity of the deterministic action across states,
        # and how hard it leans on the tanh bound. Masked to real joints.
        with torch.no_grad():
            a_det, _, _ = self.actor(enc_det.body, ji, dtheta_max,
                                     deterministic=True)
            jmask = (dtheta_max > 0).float()
            out["actor/action_std"] = float(
                (a_det.std(dim=0) * jmask).sum() / jmask.sum().clamp_min(1)
            )
            out["actor/sat_frac"] = float(
                ((a_det.abs() / dtheta_max.clamp_min(1e-9)) > 0.95
                 ).float().mul(jmask).sum() / jmask.sum().clamp_min(1)
            )
            out["data/action_std"] = float(
                (dtheta.std(dim=0) * jmask).sum() / jmask.sum().clamp_min(1)
            )
        # collision precision/recall: predicted unsafe = Q(s,a) below the
        # filter margin; ground truth = next state actually violated (h < 0,
        # h is stored in h_scale units like the margin). The filter's core
        # quality signal — moves long before success rate does.
        h_next = _t(b["h_next"])
        margin = float(cfg.filter_margin) * float(cfg.h_scale)
        pred_unsafe = qmin < margin
        true_unsafe = h_next < 0
        n_pred = float(pred_unsafe.sum())
        n_true = float(true_unsafe.sum())
        out["q/precision"] = float(
            (pred_unsafe & true_unsafe).sum() / max(n_pred, 1e-9)
        )
        out["q/recall"] = float(
            (pred_unsafe & true_unsafe).sum() / max(n_true, 1e-9)
        )
        # per-embodiment / per-task splits: free — both ids ride in the batch
        emb_ids = np.asarray(b["embodiment_id"])
        task_ids = np.asarray(b["task_id"])
        gap = (h - qmin).detach().cpu().numpy()
        q_np = qmin.detach().cpu().numpy()
        h_np = h.detach().cpu().numpy()
        for eid in np.unique(emb_ids):
            m = emb_ids == eid
            out[f"emb{eid}/mean_q"] = float(q_np[m].mean())
            out[f"emb{eid}/mean_gap"] = float(gap[m].mean())
            out[f"emb{eid}/mean_h"] = float(h_np[m].mean())
        for tid in np.unique(task_ids):
            m = task_ids == tid
            out[f"task{tid}/mean_q"] = float(q_np[m].mean())
            out[f"task{tid}/mean_gap"] = float(gap[m].mean())
            out[f"task{tid}/mean_h"] = float(h_np[m].mean())
        return out

    # ---------- eval ----------
    # NOTE: no global no_grad here — the eval probe runs cuRobo planning,
    # which needs autograd. run_eval_episode wraps model calls locally.
    def evaluate(self, wandb_run=None) -> dict:
        """Filtered-only rollout sweep: one episode per task, the embodiment
        rotating across sweeps so all (task, embodiment) combinations are
        covered every len(embodiments) sweeps."""
        from main.train.eval_utils import run_eval_episode, save_eval_figure

        cfg = self.cfg
        n_evals = getattr(self, "_n_evals", 0)
        self._n_evals = n_evals + 1
        n_emb = len(cfg.embodiment_choices)
        metrics = {}
        for i, task in enumerate(cfg.task_choices):
            embodiment = cfg.embodiment_choices[(n_evals + i) % n_emb]
            seed = cfg.eval_seeds[0] + n_evals * len(cfg.task_choices) + i
            trace = run_eval_episode(
                cfg, seed, task, embodiment, self._eval_encoder,
                self.policy_enc, self.actor, self.critics,
                tag=f"_step{self.step}",
            )
            key = f"{task}/{embodiment}"
            metrics[key] = trace["metrics"]
            fig_path = save_eval_figure(
                trace, self.run_dir / "eval", key.replace("/", "_"), self.step
            )
            if wandb_run is not None:
                import wandb

                logs = {f"eval/{key}_hv_fig": wandb.Image(str(fig_path))}
                # panel video (observer + white stats panel)
                if trace.get("video_path"):
                    logs[f"eval/{key}_video"] = wandb.Video(
                        trace["video_path"], format="mp4"
                    )
                _safe_wandb_log(wandb_run, logs, self.step)
        if wandb_run is not None:
            import wandb

            flat = {}
            for key, m in metrics.items():
                for k, v in m.items():
                    flat[f"eval/{key}/{k}"] = v
            # aggregate across the sweep
            import numpy as _np

            flat["eval/success_rate"] = float(
                _np.mean([m["task_success"] for m in metrics.values()])
            )
            flat["eval/violation_rate"] = float(
                _np.mean([m["violation_rate"] for m in metrics.values()])
            )
            flat["eval/intervention_rate"] = float(
                _np.mean([m["intervention_rate"] for m in metrics.values()])
            )
            # EMA per (task, embodiment): sweeps cover each combo only every
            # len(embodiments) sweeps — the EMA keeps the wandb panels
            # populated instead of one-episode samples reading as noise
            ema = getattr(self, "_eval_ema", None) or {}
            beta = float(getattr(self.cfg, "eval_ema", 0.7))
            for key, m in metrics.items():
                slot = ema.setdefault(key, {})
                for k, v in m.items():
                    if isinstance(v, bool):
                        v = float(v)
                    if not isinstance(v, (int, float)) or not np.isfinite(v):
                        continue
                    slot[k] = beta * slot.get(k, v) + (1 - beta) * v
                    flat[f"eval_ema/{key}/{k}"] = slot[k]
            self._eval_ema = ema
            _safe_wandb_log(wandb_run, flat, self.step)
        return metrics

    # ---------- training loop ----------
    def train_steps(self, n: int, wandb_run=None) -> None:
        """n gradient steps; a rollout sweep fires at training start and
        every eval_every * eval_sweep_every_epochs steps (a sweep is 5 full
        episodes — running it per epoch would dominate wall time)."""
        if not getattr(self, "_swept_at_start", False):
            self._swept_at_start = True
            self.evaluate(wandb_run)
        # async online: if the warmup collector hasn't written enough for a
        # batch yet, wait for it rather than crashing on an empty buffer
        import time as _time

        while len(self.buf) < self.cfg.batch_size:
            print(f"[train] waiting for data ({len(self.buf)} steps)...")
            _time.sleep(5)
            self.buf.refresh()
        sweep_every = self.cfg.eval_every * self.cfg.eval_sweep_every_epochs
        t0 = time.time()
        for _ in range(n):
            m = self.grad_step()
            if self.step % 100 == 0:
                self.buf.refresh()  # async collector appends between sweeps
            if self.step % self.cfg.checkpoint_every == 0:
                # publish weights so a --follow collector picks them up
                self.save_checkpoint(self.run_dir / "checkpoint.pt")
            if self.step % 20 == 0:
                sps = (self.step + 1) / (time.time() - t0 + 1e-9)
                print(
                    f"step {self.step}  loss_q {m['loss_q']:.4f}  "
                    f"loss_pi {m['loss_pi']:.4f}  q {m['mean_q']:.3f}  "
                    f"h {m['mean_h']:.3f}  gap {m['mean_gap_h_q']:.3f}  "
                    f"{sps:.1f} steps/s"
                )
                if wandb_run is not None:
                    _safe_wandb_log(
                        wandb_run, {**m, "steps_per_sec": sps}, self.step
                    )
            if self.step % sweep_every == 0 and self.step > 0:
                self.evaluate(wandb_run)

    def save_checkpoint(self, path: Path) -> Path:
        path = Path(path)
        # atomic publish: the --follow collector stats mtime per episode and
        # would read a half-written file if we saved in place
        tmp = path.parent / (path.name + ".tmp")
        torch.save(
            {
                "injection": self.injection.state_dict(),
                "trunk": self.trunk.state_dict(),
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
            },
            tmp,
        )
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load weights from a previous run (e.g. LOO phase 1 -> finetune).
        Shape-compatible across embodiments by design (no per-joint
        parameters). Polyak targets are re-synced to the loaded weights."""
        ckpt = torch.load(Path(path), map_location="cuda", weights_only=True)
        self.injection.load_state_dict(ckpt["injection"])
        self.trunk.load_state_dict(ckpt["trunk"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critics.load_state_dict(ckpt["critics"])
        self.critics_targ.load_state_dict(self.critics.state_dict())
        self.injection_targ.load_state_dict(self.injection.state_dict())
        self.trunk_targ.load_state_dict(self.trunk.state_dict())
        print(f"loaded checkpoint {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=CEHJ_ROOT / "data" / "smoke")
    p.add_argument("--run", type=Path, default=CEHJ_ROOT / "runs" / "smoke")
    p.add_argument("--grad-steps", type=int, default=204800,
                   help="200 epochs x 1024 by default")
    p.add_argument("--eval-every", type=int, default=1024,
                   help="grad steps per epoch (test runs: 50)")
    p.add_argument("--eval-sweep-every-epochs", type=int, default=10,
                   help="rollout sweep every N epochs (a sweep is 5 full "
                        "episodes); set 1 for short test runs")
    p.add_argument("--collect-rounds", type=int, default=0,
                   help=">0: DAgger loop — collect N rounds (round 0 "
                        "nominal-only, later rounds cfg.filter_episode_frac "
                        "with current weights), training between rounds")
    p.add_argument("--episodes-per-round", type=int, default=0,
                   help="per DAgger round; 0 = len(tasks) x len(embodiments) "
                        "so every round covers the full product")
    p.add_argument("--grad-steps-per-round", type=int, default=1024)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--leave-out", default=None,
                   help="leave-one-out phase 1: exclude this embodiment")
    p.add_argument("--only-embodiment", default=None,
                   help="leave-one-out phase 2: train ONLY this embodiment")
    p.add_argument("--init-from", type=Path, default=None,
                   help="load weights from a previous checkpoint "
                        "(e.g. the 4-embodiment phase-1 run) before training")
    args = p.parse_args()
    args.data = args.data.resolve()
    args.run = args.run.resolve()  # RoboTwin chdirs during env creation

    run = None
    if not args.no_wandb:
        import wandb

        try:
            run = wandb.init(project="cehj-hjsac", dir=str(args.run))
        except Exception as exc:
            # wandb being flaky must never take training down
            print(f"[train] wandb.init failed ({exc}); continuing offline")
            run = None

    if args.collect_rounds > 0:
        # DAgger loop: train on the current buffer, then collect the next
        # round with the UPDATED weights (filter episodes per
        # cfg.filter_episode_frac), then train again. If the data dir already
        # holds a warmup buffer (collect.py --success-only), it is reused
        # directly and round 0 trains on it first; otherwise round 0
        # collects a nominal-only warmup first. Buffer capacity is sized up
        # front for all rounds when created here.
        import dataclasses

        from main.train.collect import collect

        cfg = FrozenConfig()
        cfg.eval_every = args.eval_every
        cfg.eval_sweep_every_epochs = args.eval_sweep_every_epochs
        apply_embodiment_selection(cfg, args.leave_out, args.only_embodiment)
        n_ep = args.episodes_per_round or (
            len(cfg.task_choices) * len(cfg.embodiment_choices)
        )
        warmup_exists = (args.data / "buffer" / "header.json").exists()
        n_collects = args.collect_rounds + (0 if warmup_exists else 1)
        capacity = (n_collects * n_ep
                    * (cfg.max_steps_per_episode + 2)) if not warmup_exists else None
        trainer = Trainer(cfg, args.data, args.run, create_capacity=capacity)
        if args.init_from is not None:
            trainer.load_checkpoint(args.init_from)
        if run is not None:
            run.config.update(cfg.to_dict())
        # warmup ids were assigned by the standalone collector; keep round
        # ids clear of them (ids are diagnostic only)
        id_base = 10_000 if warmup_exists else 0
        if not warmup_exists:
            cfg_w = dataclasses.replace(cfg, n_episodes=n_ep)
            collect(cfg_w, args.data, buf=trainer.buf, episode_offset=0,
                    encoder=trainer._eval_encoder, models=None,
                    round_index=0)
        for r in range(args.collect_rounds):
            trainer.train_steps(args.grad_steps_per_round, run)
            trainer.save_checkpoint(args.run / "checkpoint.pt")
            models = (trainer.policy_enc, trainer.actor, trainer.critics)
            cfg_r = dataclasses.replace(cfg, n_episodes=n_ep)
            collect(cfg_r, args.data, buf=trainer.buf,
                    episode_offset=id_base + r * n_ep,
                    encoder=trainer._eval_encoder, models=models,
                    round_index=r + 1)
        trainer.evaluate(run)
    else:
        cfg = FrozenConfig.load(args.data / "config.json")
        cfg.grad_steps = args.grad_steps
        cfg.eval_every = args.eval_every
        cfg.eval_sweep_every_epochs = args.eval_sweep_every_epochs
        apply_embodiment_selection(cfg, args.leave_out, args.only_embodiment)
        if run is not None:
            run.config.update(cfg.to_dict())
        trainer = Trainer(cfg, args.data, args.run)
        if args.init_from is not None:
            trainer.load_checkpoint(args.init_from)
        trainer.train_steps(cfg.grad_steps, run)
        trainer.save_checkpoint(args.run / "checkpoint.pt")
        trainer.evaluate(run)
    if run is not None:
        try:
            run.finish()
        except Exception as exc:
            print(f"[train] wandb finish failed ({exc})")
    print("TRAIN DONE")


if __name__ == "__main__":
    main()
