# CEHJ_safe — HJ-SAC safety filter on RoboTwin

Cross-embodiment HJ reachability safety filter: a frozen HoloBrain-0 GD encoder
(vision + robot-state) feeds a geometric trunk; a token actor and twin critics
learn V(s) ≈ min future clearance h, and a least-restrictive filter overrides
the nominal planner when Q(s, a_nom) < 0.

## Environment

- conda env `RoboTwin` (sapien 3.0.0b1, RTX ray tracing). RoboTwin lives at
  `../RoboTwin` relative to the repo root and is on the python path.
- HoloBrain checkpoint: `HOLOBRAIN_CKPT` env var, default
  `/root/autodl-tmp/HoloBrain/HoloBrain_v0.0_GD` (see `main/train/collect.py:CKPT_DIR`).
- wandb project `cehj-hjsac`; online only when `WANDB_API_KEY` is set
  (train.py:41). Set `WANDB__SERVICE_WAIT=300` on slow networks.
- No tmux on autodl hosts — use `screen` (`screen -dmS name ...`, `screen -r`).

## Core semantics (do not change casually — they define the MDP)

- Physics 250 Hz, control 25 Hz, one control tick = exactly 10 physics steps.
- Actor action = per-tick joint displacement δθ, bounded by
  `dtheta_max = URDF velocity limit × dt × kappa`, **kappa = 1.0** (config.py) —
  a saturated actor tick and a full-speed cuRobo tick share the same box.
  The actor never drives grippers (prismatic columns stay zero).
- **h = min signed distance (robot ∪ held payload) → spawned obstacle**, metres.
  The table is NOT in h (cuRobo keeps it in its world model). No obstacle →
  h = +inf → spawn retries are mandatory (see below).
- **h_scale = 20**: h/V stored and trained in h×20 units; anything human-facing
  (panel, figures, eval metrics) converts back to cm via ×(100/h_scale).
- **Filter trigger**: engage when `Q(s, a_nom) < 0` (filter_margin = 0, scaled
  by h_scale at use), 3-tick intervention blocks (`hj_hold_ticks`), actor
  re-queried every tick, release tested at each block boundary. No hysteresis.
- a_nom = pos[min(i_end, T-1)] − qpos_live, per arm, from the live joint state
  (never the plan's own start) — rollout.py `_anom`.
- Buffer actions are the COMMANDED displacement per tick, tagged
  action_source (0 planner / 1 actor / 2 perturbation).

## Layout

- `main/envs/` — `env.py` (Env wrapper, `get_encoder_obs` HoloBrain-parity
  bundle, `step_dtheta` velocity-feedforward stepping), `controller.py`
  (TickChunkedController: per-tick hook, deferred replan after interventions),
  `obstacle.py` (spawn: on/off-path, keepaways, tiered fallback, corridor
  t∈U[0.22,0.48], model `068_boxdrink`), `distance.py` (sphere/OBB distances,
  held-object detection latch, `obstacle_contact(env) -> (force_N, touched)`),
  `record.py`, `run.py` (standalone sweep runner — the reference pipeline).
- `main/network/` — `encoder.py` (HoloBrainEncoder: frozen backbone/neck/
  spatial_enhancer + robot-state encoder), `body_features.py`
  (BodyTokenExtractor: per-link tokens, Jacobians, dtheta_max), `heads.py`
  (PolicyEncoder/TokenActor/TwinCritic — softmin temperature is `softmin_T ×
  h_scale`, NOT in the state_dict), `trunk.py`.
- `main/train/` — see below.

## Training pipeline

Warmup (standalone, reusable, once per dataset — collect ALL embodiments,
success-only, nominal-only):

```
python main/train/collect.py --episodes 250 --success-only --record-video \
    --capacity 250000 --out <warmup_dir>          # buffer in <warmup_dir>/buffer
```

Round-based training (train → collect with current weights → train):

```
python main/train/train.py --collect-rounds 10 --episodes-per-round 3 \
    --grad-steps-per-round 1024 --eval-every 1024 \
    --init-from <prev checkpoint.pt> --init-buffer <warmup_dir>/buffer \
    --buffer-dir <LOCAL disk path> --data <data_dir> --run <run_dir>
```

- Round 0 trains on the warmup buffer first; each later round collects with
  80% filter / 20% nominal-only episodes (`filter_episode_frac`).
- `main/train/run_online.py` — async alternative (separate watch collector
  process + trainer, checkpoint-follow). Faster wall-clock (collection hides
  behind training); round mode is the debuggable default.

## Standard training (production defaults)

```
python main/train/train.py --collect-rounds 40 --grad-steps-per-round 1024 \
    --eval-every 1024 --leave-out franka-panda \
    --init-from <ckpt> --buffer-dir <local>/buffer \
    --data <data_dir> --run <run_dir>
```

40 rounds x 1024 grad steps, full 5x4 (pool) episode product per round,
cross-eval on the held-out embodiment every 10 epochs.

## Ablations

`--ablate-geometry` — no (log-dist, direction) attention bias and no
attention-weighted direction channel, in the trunk AND the critic's late
blocks (post_action_geometry follows automatically). `--ablate-injection` —
body tokens are the frozen HoloBrain state tokens alone, no analytic
feature columns. `--vanilla` = both: plain cross-attention over frozen
state + scene tokens with the SAME actor/critic heads (the baseline).
Ablated runs need their own checkpoints — logit shapes differ.

## Cross-validation (LOO and leave-4-out)

ONE shared warmup buffer for everything — collect the full 5x5 pool once,
never per split (the trainer filters samples by embodiment_id at sampling
time; see buffer.set_embodiment_filter):

```
python main/train/collect.py --episodes 250 --success-only --record-video \
    --capacity 250000 --out <warmup_dir>
```

Leave-one-out (train on 4, finetune the 5th):

```
# phase 1: exclude piper; cross-eval on piper runs every 10 epochs
python main/train/train.py --collect-rounds N ... --leave-out piper \
    --init-buffer <warmup_dir>/buffer --buffer-dir <local> --run <run_p1>

# phase 2: finetune the phase-1 checkpoint on piper only
python main/train/train.py --collect-rounds M ... --only-embodiment piper \
    --init-from <run_p1>/checkpoint.pt --init-buffer <warmup_dir>/buffer ...
```

Leave-4-out (train on 1, eval the other 4): just drop --init-from —

```
python main/train/train.py --collect-rounds N ... --only-embodiment piper \
    --init-buffer <warmup_dir>/buffer --buffer-dir <local> --run <run_l4o>
```

The cross-embodiment sweep fires every 10 epochs on ALL embodiments outside
the training pool (1 for LOO, 4 for L4O): 3 filtered episodes per sweep,
tasks rotating through the 5, under <run>/eval/cross_embodiment/ and wandb
cross_overview/ + cross_heatmap/ + cross_videos.

### Running several LOO/L4O splits in parallel

Each run needs its OWN memmap buffer — two processes appending to the same
memmap corrupt it. The default buffer location is <run_dir>/buffer (unique
per run); with an explicit --buffer-dir, give every split its own local dir:

```
python main/train/train.py --leave-out piper        --run runs/loo_piper   --buffer-dir /local/buf_piper  --init-buffer <warmup>/buffer ...
python main/train/train.py --leave-out franka-panda --run runs/loo_franka  --buffer-dir /local/buf_franka --init-buffer <warmup>/buffer ...
```

The warmup buffer is only ever READ (seeded via --init-buffer at startup), so
all splits share it safely. run_online.py's async mode is the exception: its
collector and trainer intentionally share one buffer inside a single job.


  behind training) but round mode is the debuggable default.

## Test training (fast pipeline check)

Reuse the existing warmup buffer and resume from any checkpoint; small
rounds keep the loop tight (~5 min/round):

```
python main/train/train.py --collect-rounds 10 --episodes-per-round 3 \
    --grad-steps-per-round 256 --eval-every 256 --leave-out franka-panda \
    --init-from <run_dir>/checkpoint.pt --buffer-dir <local>/buffer \
    --data <data_dir> --run <run_dir>
```

(franka-panda is left out BY DEFAULT for test training: the buffer filter
drops its samples, and the cross-embodiment sweep on franka runs every 10
epochs — see below.)

From scratch (no warmup buffer): add `--warmup-episodes 20` and drop
`--init-buffer`/`--init-from` — the trainer first collects 20 success-only
nominal episodes with videos, then starts the rounds. Buffer capacity is
sized for warmup + all rounds up front.

Per round you get: filter-episode videos (`round_videos` carousel in wandb +
`eval/videos/`), per-metric heatmaps + trend.png under `eval/<metric>/`, and
`round_overview/*` scalars in wandb. Watch violation_* rates, v_le_h_frac,
q/precision-recall, actor/action_std, and the videos for the filter engaging.

## Buffer (`main/train/buffer.py`)

- Flat per-step memmap, ~630 KB/step (dominated by fp16 scene tokens).
- **Ring semantics**: at capacity, oldest transitions are overwritten
  (warmup ages out first). `head` persists in header.json; sampling excludes
  pairs crossing the write head / wrap seam. `discard_to` (success-only
  rollback) works only pre-wrap.
- Keep the memmap on LOCAL disk (`--buffer-dir`): ~80 MB random reads per grad
  step stall badly on network FS (autodl-fs is also quota-limited beyond what
  `df` shows). Buffer capacity is fixed at creation — size generously.

## Metrics (wandb + `eval/<metric>/roundXXX.png` heatmaps + `trend.png`)

- `violation_rate`: h < 0 (analytic model). `violation_contact`: real surface
  touch (PhysX separation ≤ 0, obstacle vs anything non-table).
  `violation_force`: contact force ≥ 20 N (contact_force_threshold; measured
  real contacts are ≥ 79 N). `violation_any`: h<0 OR force violation.
- `v_le_h_frac`: fraction of ticks with V ≤ h — HJ consistency; must → 1.0.
- `q/precision`, `q/recall`: predicted-unsafe (Q < margin) vs actual h<0 next.
- `actor/action_std`, `actor/sat_frac`, `data/action_std`, `entropy`:
  mode-collapse diagnostics.
- `mean_realized_ratio` / `realized_cmd_ratio`: ‖achieved Δq‖/‖commanded δθ‖.
- `min_h`, `mean_gap` (h−V), per-combo heatmaps are filter-episodes only.
- eval sweeps are OFF by default (`eval_sweep_every_epochs=0`) — metrics come
  from the collection episodes themselves via RoundMetrics.

## Videos

- Panel video per episode attempt (warmup: all; rounds: filter episodes only).
  Observer frame + bbox overlay (obstacle/held) + fixed-height panel:
  h/Q on one line (cm), control+|a| on one line, intervention counter, block
  counter (always present), force line (red + TOUCH on contact), h/V/F
  mini-plot (F normalized — checks alignment with h dips).
- wandb logging: one `round_videos` carousel per round (list under one key),
  filtered by file MTIME — filenames collide across restarts, never use
  name-based dedup.

## Gotchas (learned the hard way)

- Ray tracing: call `task._update_render()` before any `take_picture` or the
  RT scene is stale. OIDN "invalid handle" errors are harmless spam (~10
  ms/tick). `RenderCameraGroup` renders WRONG poses for mounted cameras on
  CPU sim — do not use it.
- RT Color pictures are HDR float; alpha is NOT 0/1 — depth mask is
  clamp(alpha,0,1) × (Position.w < 1), see env.py get_encoder_obs.
- Tick accumulator: 10 × float(0.1) < 1.0 — rollout.py uses a 1e-9 epsilon,
  keep it.
- PhysX contact points are speculative (1–2 cm, zero impulse) — real touches
  are separation ≤ 0.
- wandb calls must stay wrapped in _safe_wandb_log (CommError killed a run).
- Checkpoints publish atomically (tmp + os.replace); the --follow collector
  tolerates load failures.
- git push: ghfast.top is read-only; push direct to github.com with the
  token. If direct times out, use the AutoDL turbo proxy:
  `https_proxy=http://10.37.1.23:12798 git push ...`
- Never commit `data/` (memmaps) or `wandb/` — gitignored.
- cuRobo init fails with Warp NVRTC "cannot open temporary file" when TMPDIR
  is missing/unwritable — `export TMPDIR=/tmp`. The mplib RRT fallback then
  silently takes over (slower, no constraints) — grep the log for
  "CuRobo init failed".

## Quick checks

- `python main/train/buffer.py`-level unit snippets for ring/sampling live in
  git history; `bench_warmup_cost.py` profiles a collection episode;
  `profile_sim.py` / `bench_step_cost.py` profile the vanilla pipeline.
