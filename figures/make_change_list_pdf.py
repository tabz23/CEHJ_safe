"""Render the HJ-SAC change list to a wide landscape PDF.

A3 landscape keeps the two description columns long, so rows wrap less and
the document stays short vertically.

    PYTHONPATH=/tmp/pdflibs python figures/make_change_list_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    LongTable,
    PageTemplate,
    Paragraph,
    Spacer,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "hjsac_change_list.pdf"

PAGE = landscape(A3)
MARGIN = 12 * mm
USABLE = PAGE[0] - 2 * MARGIN
COLS = [0.115 * USABLE, 0.425 * USABLE, 0.460 * USABLE]
COLS4 = [0.105 * USABLE, 0.295 * USABLE, 0.295 * USABLE, 0.305 * USABLE]

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b5b5b")
RULE = colors.HexColor("#c8c8c8")
HEAD_BG = colors.HexColor("#ececec")
ROW_BG = colors.HexColor("#f7f7f7")
ACCENT = colors.HexColor("#7a2718")


def _register_fonts() -> tuple[str, str, str]:
    """DejaVu from the matplotlib data dir: Helvetica cannot render Greek."""
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    faces = {
        "DJV": "DejaVuSans.ttf",
        "DJV-Bold": "DejaVuSans-Bold.ttf",
        "DJV-Oblique": "DejaVuSans-Oblique.ttf",
        "DJVMono": "DejaVuSansMono.ttf",
    }
    for name, fn in faces.items():
        pdfmetrics.registerFont(TTFont(name, str(ttf / fn)))
    pdfmetrics.registerFontFamily(
        "DJV", normal="DJV", bold="DJV-Bold", italic="DJV-Oblique"
    )
    return "DJV", "DJV-Bold", "DJVMono"


BODY, BOLD, MONO = _register_fonts()

st_title = ParagraphStyle(
    "title", fontName=BOLD, fontSize=17, leading=21, textColor=INK, spaceAfter=3
)
st_sub = ParagraphStyle(
    "sub", fontName=BODY, fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=11
)
st_sec = ParagraphStyle(
    "sec", fontName=BOLD, fontSize=11.5, leading=14, textColor=ACCENT,
    spaceBefore=13, spaceAfter=4,
)
st_part = ParagraphStyle(
    "part", fontName=BOLD, fontSize=14, leading=17.5, textColor=INK,
    spaceBefore=16, spaceAfter=5, borderPadding=4,
    borderWidth=0, backColor=colors.HexColor("#e6e0dd"),
)
st_lede = ParagraphStyle(
    "lede", fontName=BODY, fontSize=8.0, leading=11, textColor=MUTED, spaceAfter=5
)
st_note = ParagraphStyle(
    "note", fontName=BODY, fontSize=8.2, leading=11.4, textColor=INK, spaceAfter=7
)
st_cell = ParagraphStyle(
    "cell", fontName=BODY, fontSize=7.4, leading=9.9, textColor=INK
)
st_item = ParagraphStyle(
    "item", fontName=BOLD, fontSize=7.4, leading=9.9, textColor=INK
)
st_hdr = ParagraphStyle(
    "hdr", fontName=BOLD, fontSize=8.2, leading=10.5, textColor=INK
)


def fmt(text: str) -> str:
    """Escape XML, then map `code` -> mono and **bold** -> bold."""
    out = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font face="{MONO}" size="6.9">{m.group(1)}</font>',
        out,
    )
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out


def cell(text: str, style=st_cell) -> Paragraph:
    return Paragraph(fmt(text), style)


def _styled(data, widths) -> LongTable:
    t = LongTable(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, MUTED),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_BG))
    t.setStyle(TableStyle(style))
    return t


def table(rows: list[tuple[str, str, str]]) -> LongTable:
    head = ("Item", "Previously", "To be done")
    data = [[Paragraph(fmt(h), st_hdr) for h in head]]
    data += [[cell(a, st_item), cell(b), cell(c)] for a, b, c in rows]
    return _styled(data, COLS)


def table4(rows: list[tuple[str, str, str, str]]) -> LongTable:
    head = ("Item", "HoloBrain reference (what it expects)", "CEHJ today (what we send)",
            "Why it matters / what to do")
    data = [[Paragraph(fmt(h), st_hdr) for h in head]]
    data += [[cell(a, st_item), cell(b), cell(c), cell(d)] for a, b, c, d in rows]
    return _styled(data, COLS4)


HFUNC = [
    (
        "h definition",
        "`h = min(d_block, d_table)`. `d_block = distance_info(env)[\"d_system\"]` is arms plus held payload against the obstacle OBB. `d_table` is a separate half-space term: the minimum over arm collision spheres of `center_z - radius - table_z - table_margin`, with `table_z = table_height + task.table_z_bias`.",
        "**`h = d_block` only.** Because `d_system == d_min == min(d_left, d_right, d_left_held, d_right_held)` in `distance.py`, h becomes exactly the `d_min` that `run_safe_unsafe_bimanual_plank.sh` already reports. Training h and the sweep's distance are then the same number, directly comparable.",
    ),
    (
        "Table term removal and the spawn guarantee",
        "`compute_h` builds `left_spheres` / `right_spheres` from `_arm_spheres`, computes `d_table`, and folds it into both `h` and every entry of `per_link`; `main/envs/distance.py` has no table logic at all, which is why the sweep never saw this term. Separately, `d_block = inf` when `env.obstacle is None`, but `d_table` was always finite so `h` stayed finite. `geometric_pose` tries the corridor `t`, then perpendicular nudges of 0.03 to 0.16 m, then a best-line-distance fallback; if all fail it prints \"no spawn pose clears pick/target keepaway\" and returns `(None, None)`, `choose_and_spawn` passes that up as `(None, None, None, arm)`, and the episode runs obstacle-free.",
        "Delete the table computation from `compute_h`. `table_margin` and `table_height` stay in `FrozenConfig` because `update_curobo_world` still pushes the table into cuRobo's world model, so the planner keeps avoiding it; only the value function stops scoring it. **That makes the spawn path safety-critical:** with nothing else bounding h, an obstacle-free episode gives `h = +inf` on every tick, which reaches the memmap and turns the Bellman target into NaN on the first batch that samples it. So spawning becomes total, in three tiers. **Tier 1** is the existing chain, unchanged. **Tier 2 is new:** if on_path placement fails, place the obstacle anywhere feasible on the table by scanning the existing `_table_grid(block_r)` lattice \u2014 11 x 9 points already inset by `block_r + 0.02` \u2014 keeping candidates that pass `_in_table` and `_keepaway_ok` and taking the one with maximum keepaway clearance. **Tier 3:** if that set is empty, skip the episode. `collect.py` tests the returned actor for None and re-draws from the schedule instead of proceeding.",
    ),
    (
        "per_link and true_argmin",
        "`per_link[label] = min(obstacle_distance, d_tab)`, so `true_argmin` could name a link that is merely low over the table rather than near the obstacle.",
        "Obstacle distance only, so `true_argmin` always names the link genuinely closest to the box. That makes it a usable check on the learned per-token argmin in the softmin readout.",
    ),
    (
        "Arm-to-arm distance",
        "`d_arm_arm` computed over all left/right sphere pairs and reported in `diag`, but deliberately excluded from h; self-collision is out of scope.",
        "Unchanged.",
    ),
    (
        "Accepted risk",
        "The table term was the only thing preventing the actor from treating the tabletop as free space.",
        "The HJ actor has no collision model and is bounded only by d\u03b8_max, so nothing now penalizes driving a link into the table, particularly in early rounds when the policy is close to random. Accepted on the grounds that the obstacle is lateral and descending does not increase obstacle clearance. Tracked by the diagnostic in section 8 rather than by h.",
    ),
]

CONFIG = [
    (
        "control_freq",
        "`Env(control_freq=20.0)`. 250 / 20 = 12.5, so `_physics_substeps` carries `_substep_acc` and emits 12 or 13 physics steps per tick, alternating; `RolloutController.hooked_step` mirrors it with `_tick_acc += 0.08`. Every index calculation has to consult the accumulator to know how long the current tick is. A 2720-physics-step episode is 218 control ticks.",
        "`25.0` Hz. 250 / 25 = 10 exactly, so a control tick is always exactly 10 physics steps and neither accumulator ever carries a fraction. **That is the main reason to do it:** a_nom spans a fixed 10 plan rows, tick m starts at physics step `10 * m`, and the 12-or-13 ambiguity leaves the action-index logic entirely. Cost: the same episode is 272 ticks, so 25% more encoder forwards and 25% more stored transitions \u2014 at roughly 630 KB per step, about 171 MB per episode rather than 137 MB.",
    ),
    (
        "The 20 Hz literals that `control_freq` does NOT reach",
        "Seven call sites hardcode the old rate instead of reading it from the env, so `control_freq` is **not** a single knob. Defaults `control_freq: float = 20.0` at `main/train/collect.py:113`, `main/train/nominal.py:129` (`probe`) and `:184` (`cached`), and `main/envs/env.py:337`. Literal conversions `int(cfg_ep.max_steps_per_episode * 12.5)` at `main/train/collect.py:185` and `main/train/eval_utils.py:63`, and `round(int(n_chunk) / 12.5)` at `main/train/rollout.py:251`. Worst of the set is `main/train/nominal.py:220`, `step_advance = physics_freq / 20.0`, which **ignores the `control_freq` argument its own function accepts** \u2014 latent today only because the default happens to be 20.",
        "Change all seven to derive from `env.control_freq` (or `cfg.control_dt`); the three `12.5` literals become `PHYSICS_FREQ / control_freq`, which evaluates to exactly 10. **Grep for `12.5` and `20.0` before declaring the rate change done** \u2014 editing `FrozenConfig.control_dt` alone leaves the planner advancing 12.5 path rows per 10-step tick, i.e. the nominal would run 25% fast against a correct clock and a_nom would no longer match what the arm was commanded. Two sites are already correct and need no edit: `main/train/rollout.py:317` (`env.control_freq / env.PHYSICS_FREQ`) and `main/envs/env.py:512` (`PHYSICS_FREQ / control_freq`).",
    ),
    (
        "control_dt and delta_theta_max",
        "`control_dt = 0.05` s, matching 20 Hz. `delta_theta_max = joint_vel_limits * control_dt * kappa` with `kappa = 0.25`, so 0.0125 x the velocity limit per tick.",
        "`control_dt = 0.04` s, which must move in the same commit as `control_freq` since it feeds `delta_theta_max` in `body_features.py` and the softmin / Jacobian scaling \u2014 a mismatch would silently misbound the actor. The formula is unchanged, so the bound shrinks 20% to 0.01 x the velocity limit at unchanged kappa. Either accept the smaller per-tick action \u2014 finer control, but about 25% more ticks to cover the same motion \u2014 or raise `kappa` to 0.3125 to hold physical displacement constant. Recommend accepting the smaller bound, since the extra ticks are already being paid for.",
    ),
    (
        "max_steps_per_episode",
        "`400` control ticks, converted at the call site as `int(400 * 12.5) = 5000` physics steps, or 20 s of simulated time.",
        "`450` control ticks, giving `450 * 10 = 4500` physics steps, or 18 s \u2014 now an exact integer conversion. Enforced in `RolloutController.hooked_step`, which counts `scene.step()` calls and raises `RolloutTimeout`.",
    ),
    (
        "hj_hold_ticks",
        "Did not exist as a field. Intervention length was implicitly `round(replan_k / 12.5) = 5` control ticks, derived from the 60-step window rather than chosen.",
        "New, `3`. At 25 Hz that is 0.12 s, exactly 30 physics steps.",
    ),
    (
        "obstacle_model / obstacle_choices",
        "`\"086_woodenblock\"` default, sampled per episode from `(\"086_woodenblock\", \"059_pencup\")` \u2014 a 10.3 cm cube and a 9.8 x 11.7 x 9.8 cm cup. Two geometries mean h is measured against a different OBB from episode to episode.",
        "Fixed `\"068_boxdrink\"`, a 11.0 x 15.4 x 11.6 cm box, with `obstacle_choices = (\"068_boxdrink\",)`. Matches `--obstacle-model 068_boxdrink` in the sweep script, so training and the reference runs measure against identical geometry. It is taller than either previous option at 15.4 cm standing, so the arm can no longer pass over it as easily.",
    ),
    (
        "task_choices",
        "Eight tasks: `stack_blocks_two`, `stack_bowls_two`, `grab_roller`, `pick_dual_bottles`, `place_bread_basket`, `place_burger_fries`, `place_can_basket`, `place_cans_plasticbox`.",
        "Five tasks: `stack_blocks_two`, `stack_bowls_two`, `place_burger_fries`, `place_bread_basket`, `place_can_basket`. All five have explicit `corridors` entries in `TASK_SPECS`, so on/off-path spawning is well defined. Note the dropped `grab_roller` has `object == target == \"roller\"`, making its corridor degenerate and forcing `geometric_pose` onto a hardcoded fallback direction.",
    ),
    (
        "embodiment_choices",
        "Four: `piper`, `franka-panda`, `ARX-X5`, `ur5-wsg`. Three are 6-DOF per arm; `franka-panda` is 7-DOF, which is what drives the buffer padding.",
        "Unchanged.",
    ),
    (
        "obstacle_t",
        "Hardcoded `0.6`, so the obstacle always sits at the same 60% point of the pick-to-place corridor. Some incidental spread exists when `_t_bounds` snaps t away from a keepaway, but the preferred value never varies.",
        "Sampled per episode with `_corridor_t(seed)` from `main/envs/run.py`: `np.random.RandomState(seed + 917).uniform(0.30, 0.70)`. Reusing that exact function keeps parity with the sweep, where corridor position produces the t37 / t42 / t52 / t63 spread in the output directory names. Bounds match `CORRIDOR_T_LO = 0.30` and `CORRIDOR_T_HI = 0.70`.",
    ),
    (
        "obstacle_mode",
        "`\"on_path\"` for every episode.",
        "80% `on_path`, 20% `off_path`, drawn per episode. `off_path` places the obstacle roughly 0.22 m lateral to the corridor, so the expert usually stays clear and h stays large \u2014 those episodes supply the safe side of the value function. Expect the unsafe fraction of the buffer to drop by about a fifth.",
    ),
    (
        "h_scale",
        "`20.0`. h is stored and trained in units of 20 x metres.",
        "`100.0`, so h and V read directly in centimetres. `softmin_T` and the filter margin stay in physical metres in the config and are multiplied by `h_scale` at the point of use, so physical sharpness is preserved automatically. **Watch the gradients:** Bellman targets grow 5x, so critic gradients grow roughly 5x against the fixed `clip_grad_norm_(..., 10.0)` and that clip will bind far more often. Monitor `gn_critic`.",
    ),
    (
        "Filter margin",
        "Two values: `margin_on = 0.0` and `margin_off = 0.005` metres, scaled to 0.0 and 0.1 in trained units. Engage when V < 0, release when V > 0.5 cm. `margin_on = 0` means the filter waits until the safe policy itself can no longer hold clearance, leaving zero tolerance for critic error.",
        "One margin at `0.01` m, i.e. 1.0 in the new cm units. Engage when Q(s, a_nom) < 1 cm. The `margin_off` field is deleted rather than set equal, so the code does not imply a hysteresis band that no longer exists.",
    ),
    (
        "replan_k",
        "`60` physics steps, which is 0.24 s at 250 Hz, per receding-horizon window. Part of the nominal's definition.",
        "Removed; the nominal becomes vanilla `play_once`. Planning was never the cost driver \u2014 a vanilla episode spends 1.24 s planning out of 53.5 s total, against 43 s of rendering. The real saving is that vanilla episodes are shorter in physics steps, and the dominant per-episode cost in training is one encoder forward per control tick.",
    ),
    (
        "n_episodes",
        "`25` per run. With 20 (task, embodiment) combinations that gives 20 combos once and 5 twice.",
        "A multiple of 20 per collection round so the cyclic schedule closes evenly. 25 was also thin for 20 combinations; sizing is your call, but 200 episodes is roughly 57 GB at the new tick rate.",
    ),
    (
        "Other config fields",
        "`kappa = 0.25`, `softmin_T = 0.02` m, gamma annealing 0.9 to 0.999 over 50k steps, alpha 0.2 to 0.0 over 20k, `perturb_prob = 0.05`, `batch_size = 64`, `lr = 3e-4`, `tau = 0.005`, `rgbsd_archive_every = 25`.",
        "Unchanged. Note gamma's effective horizon of about 1000 decision steps is 40 s at 25 Hz rather than 50 s at 20 Hz, still comfortably longer than an 18 s episode.",
    ),
]

ACTOR_EXEC = [
    (
        "Timescales",
        "Physics is 250 Hz (`Env.PHYSICS_FREQ`, `main/envs/env.py:324`). The control loop ran at 20 Hz, so one tick was 12 or 13 physics steps depending on `_substep_acc`.",
        "Physics 250 Hz, actor 25 Hz, so one action lasts exactly **N = 250 / 25 = 10 physics steps = 40 ms**. The network is never asked for a 4 ms twitch: it outputs a displacement spanning the whole 40 ms.",
    ),
    (
        "What the actor predicts",
        "`TokenActor` head is Gaussian + tanh, and \u03b4\u03b8 = tanh(\u00b7) \u00d7 d\u03b8_max with d\u03b8_max = q\u0307_lim \u00b7 \u0394t_ctrl \u00b7 \u03ba, \u03ba = 0.25 (`main/network/body_features.py`). The bound is structural \u2014 applied inside the head \u2014 not a clip bolted on afterwards.",
        "Unchanged in form; only \u0394t_ctrl moves 0.05 \u2192 0.04 s. So **|\u03b4\u03b8| \u2264 25% of the URDF joint speed limit, sustained over 40 ms**. That 25% figure is the actor's entire authority budget, which is why losing 80% of it to the drive (next row) is not survivable.",
    ),
    (
        "Expanding \u03b4\u03b8 to the 250 Hz drives",
        "`Env.step()` writes \u03b8_d = q + \u03b4\u03b8 **once** with \u03b8\u0307_d = 0 \u2014 the hardcoded `zeros` handed to `set_arm_joints` at `main/envs/env.py:563` and `:565` \u2014 then runs all 10 physics steps without rewriting the target (`env.py:567-569`). Drive gains are kp = 1000, kd = 200 (`RoboTwin/assets/embodiments/*/config.yml`), applied in `RoboTwin/envs/robot/robot.py:625-633`. The dominant pole is kp/kd = 5 rad/s (\u03c4 \u2248 0.2 s), so the damper fights the spring and only **~20% of \u03b4\u03b8 is realized per tick**.",
        "Interpolate the same \u03b4\u03b8 across the 10 steps instead of holding it. For k = 1 \u2026 N: **\u03b8_d(k) = q + \u03b4\u03b8 \u00b7 k/N** and **\u03b8\u0307_d = \u03b4\u03b8 / \u0394t_ctrl = 25\u00b7\u03b4\u03b8**, the same velocity on all 10 steps; each `scene.step()` receives the pair (\u03b8_d(k), \u03b8\u0307_d). The PD then tracks a short straight-line joint move and **commanded \u03b4\u03b8 \u2248 achieved \u0394q**. Implementation: `Env.step()` must move `set_arm_joints` inside the substep loop rather than calling it once before it.",
    ),
    (
        "Consistency with the nominal",
        "The nominal executor already passes CuRobo's planned velocity: `task.robot.set_arm_joints(pos[i], vel[i], \"left\")` at `main/envs/controller.py:817` and `:820`. So the actor path is the **only** place in either codebase that sends a zero velocity target, and a_nom and \u03c0(s) are not the same physical quantity today.",
        "After the fix both controllers issue (position, velocity) pairs at 250 Hz with identical semantics, so a single critic can score both. This is a **precondition** for the Q(s, a_nom) trigger and for storing commanded actions in the buffer \u2014 without it the critic learns a 5x-wrong action-to-effect map on the actor branch only.",
    ),
    (
        "Why the network stays at 25 Hz",
        "The 20 Hz rate was inherited from the old receding-horizon window rather than derived from anything.",
        "Q(s, a) and \u03ba are both **defined on the control tick**, so the tick is the MDP timestep. Predict \u03b4\u03b8 at 25 Hz and expand it to 250 Hz with (\u03b8_d, \u03b8\u0307_d). Do **not** make the actor emit 250 Hz waypoints \u2014 \u03ba and d\u03b8_max lose their meaning. Do **not** run TOPP to the goal: variable-length actions break the fixed \u0394t, and with it \u03b3, the a_nom index math, and uniform h sampling.",
    ),
    (
        "Reference: how RoboTwin expands actions",
        "Not consulted previously.",
        "For comparison, `RoboTwin/envs/_base_task.py:1536-1560` builds the 2-point path `[q_current, q_target]`, runs `TOPP(path, 1/250)`, and executes **every** sample with `set_arm_joints(pos[i], vel[i])` until the segment completes (`:1633-1660`). Same principle \u2014 never a zero velocity target \u2014 but variable duration, which is exactly why we interpolate a fixed 10 steps instead of adopting TOPP. Note their silent failure path: if TOPP raises, `topp_flag` goes False, `n_step` is forced to 50 and `set_arm_joints` is never called, so the arm freezes for 50 physics steps.",
    ),
]

SWITCHING = [
    (
        "Trigger quantity",
        "Q(s, \u03c0_safe(s)) \u2014 the critic evaluated at the safe actor's own deterministic action. Both `SafetyFilter.value` and `.action` call `self.actor(..., deterministic=True)` and score that. Semantically this asks whether the safe policy can rescue the robot, which only becomes false once that policy is already losing.",
        "Q(s, a_nom) \u2014 the critic evaluated at the planner's action for the upcoming tick. Semantically: if the planner does what it wants and the robot then behaves optimally safely forever, does it stay clear. This is the textbook least-restrictive filter and overrides only when the nominal action specifically is the problem. `SafetyFilter.action` already accepts an `a_nom` argument; it is currently passed `None` and only echoed back, never scored.",
    ),
    (
        "Threshold and release",
        "Two thresholds forming a latch: set `engaged = True` when `v < margin_on`, clear it when `v > margin_off`, both scored on Q(s, \u03c0_safe). The 0.5 cm dead band stopped the filter flipping at every decision point when V hovered at the boundary.",
        "One threshold at 1 cm with the latch removed: engage when Q(s, a_nom) < 1 cm, and at each 3-tick block boundary resume the nominal if Q(s, a_nom) \u2265 1 cm, otherwise run another block. Chattering is prevented structurally by the block rather than by hysteresis, but at 3 ticks that block is only 0.12 s and is a weak substitute \u2014 log mode-switch count per episode as a chatter check, and lengthen the block rather than reinstating a second margin if it is high. **Cost caveat worth designing around:** the release test needs a_nom, hence a plan, and a real cuRobo plan is about 96 ms of wall time, so replanning at every boundary during a sustained intervention is roughly 8 plans per simulated second. Cheaper scheme \u2014 score the release test against a_nom formed from the existing plan, stale though it is, and pay for a genuine replan only once the test passes and control is actually handed back.",
    ),
    (
        "Intervention block",
        "5 control ticks, 0.25 s, computed as `round(n_chunk / 12.5)` from the 60-step window, so the length was tied to the nominal's replan period rather than chosen. One \u03b4\u03b8 was computed at the window start and reissued for all 5 ticks via `for _ in range(...): self.env.step_dtheta(a)`.",
        "3 control ticks, 0.12 s, exactly 30 physics steps, repeated until Q(s, a_nom) clears the margin. No cap, per your decision, so a stuck actor holds control until the 4500-step timeout fires; that outcome is logged distinctly so it does not masquerade as an ordinary timeout. The actor is re-queried every tick from a fresh observation \u2014 new encoder forward, new trunk pass, new \u03b4\u03b8 \u2014 since reissuing one displacement across the block would be 3 x d\u03b8_max of open-loop motion with no feedback. Three encoder forwards per block is cheap, and each one also yields the Q value the video panel needs.",
    ),
    (
        "Vanilla nominal: hook point and stale plans",
        "All three modes \u2014 `_filtered_play_seq`, `_shadow_play_seq`, `_collect_play_seq` \u2014 hook in by overriding `PlanEveryKController._play_seq`, which fires once per 60-step window. Staleness never arose, because the receding-horizon controller called `_after_chunk` after every window, replanning from the live `qpos` toward the stored `_cehj_target_pose`, so any divergence was absorbed within 0.24 s.",
        "`VanillaPlayOnceController` is a bare subclass of `ResidualController` with no `_play_seq`; stock `take_dense_action` plays an entire spline in one call with no window boundaries. A new tick-chunked wrapper around it becomes the hook: it walks the spline in fixed 10-step chunks, forms a_nom, evaluates Q, and either continues or diverts. All three modes migrate onto it. The wrapper also has to handle the staleness vanilla now creates: it plans once per Action and reports `n_replans: 0`, so after the actor moves the arm the remaining waypoints are stale and `set_arm_joints(pos[i], ...)` **snaps** the drive targets back to a configuration the arm has left \u2014 undoing the avoidance maneuver, and entering the buffer as an action many multiples of d\u03b8_max. The fix reuses existing machinery: treat an intervention as a forced `_after_chunk`, discarding the remaining waypoints and replanning the same target from live `qpos`, carrying `grip_i` across rather than resetting it.",
    ),
    (
        "Shadow-V mode",
        "`_shadow_play_seq` runs the filter as a passive observer in nominal mode: it evaluates `self.flt.value(...)`, records V, hardcodes `intervened = False`, and always falls through to the nominal. This preserves the counterfactual \u2014 a filter that acts prevents the collision it was supposed to predict.",
        "Switch `filter.value` to Q(s, a_nom) as well, so the eval figure plots exactly the quantity being thresholded. Leaving it alone would silently validate a different number. The two are not interchangeable: \u03c0_safe approximately maximizes Q, so Q(s, a_nom) is uniformly the smaller, and the filter fires more often at the same 1 cm margin than the old curve suggests. a_nom is available for free here since the plan dicts are already passed into the hook.",
    ),
]

COLLECTION = [
    (
        "Behavior policy",
        "The RoboTwin scripted `play_once` stage machine with all motion planned by cuRobo, plus a 5% chance per window of one control tick of uniform random \u03b4\u03b8 within \u00b1d\u03b8_max. The HJ actor is **never constructed** \u2014 `collect()` builds `RolloutController` with no actor and no critics, so `self.flt` is `None`. Nothing learned influences the data and the buffer contains zero on-policy samples.",
        "80% of episodes run the full deployment stack, vanilla nominal plus HJ switching at 1 cm; 20% run nominal only. The dataset then covers the state distribution the deployed filter actually induces, which fixes the coverage gap where the critic must currently extrapolate to the actor's action region from planner-scale data.",
    ),
    (
        "Pipeline structure",
        "Strictly sequential and single-pass: `collect.py` runs all episodes to completion and writes the memmap, then `train.py` reads that fixed buffer and never touches the simulator except at eval.",
        "Alternating rounds. Round 0 must be nominal-only since the actor is randomly initialized at that point. Then train, then each subsequent round collects with current weights and appends. This turns `train.py` from a single offline loop into an outer DAgger-style loop and requires the encoder, `policy_enc`, actor and critics to be live during collection.",
    ),
    (
        "Action stored, and why it matters",
        "Achieved displacement: `qpos_raw[t+1] - qpos_raw[t]`, read back from the simulator at consecutive captures. Applied uniformly \u2014 `_collect_tick` never asks which controller drove the tick, so planner ticks, actor ticks and perturbation ticks are all handled by the same three lines. On contact the achieved displacement collapses toward zero while the command stays large, so the buffer records a near-zero action ending in collision and the critic learns that holding still is dangerous there \u2014 the wrong causal attribution, in exactly the transitions carrying the safety signal. In the reference sweep 53 of 64 vanilla episodes make contact, so this is the dominant regime.",
        "Commanded displacement, per source. Planner ticks store the net delta-joints the spline asks for over the tick; HJ ticks store exactly the \u03b4\u03b8 handed to `step_dtheta`; perturbation ticks store the sampled \u03b4\u03b8, which already exists in a local variable and is currently discarded. Requires the executor to report its command, which the new tick wrapper is positioned to do. This attributes the collision to the action that caused it, and matches what Q is queried with at deployment for both Q(s, a_nom) in the filter and Q(s, \u03c0(s)) in the actor loss.",
    ),
    (
        "Building a_nom: reference point and indices",
        "Neither arose. Achieved displacement is measured-to-measured by construction, and at 20 Hz any index arithmetic would have been awkward anyway, since a tick spanned 12 or 13 plan rows depending on where `_tick_acc` happened to land.",
        "`a_nom = pos[i_end] - qpos_now`, referenced against the robot's **live** joint state rather than the plan's own starting waypoint `pos[i_start]`. `step_dtheta` defines the actor's action as \u03b8_now + \u03b4\u03b8, so referencing the nominal identically makes the two controllers' actions the same physical quantity \u2014 a prerequisite for one Q scoring both. **Under vanilla this is not cosmetic:** an entire Action runs open-loop, so plan-versus-actual drift accumulates monotonically and `pos[i_end] - pos[i_start]` would bake that growing error into every stored action. For the indices, `result[\"position\"]` is `[T, n_joints]` and the executor consumes exactly one row per `scene.step()`, so row index and physics step are 1:1; **at 25 Hz a tick is always 10 rows,** so for tick m within an Action, `i_start = 10 * m` and `i_end = i_start + 9`, with no accumulator involved. T covers one `play_once` Action, not the episode, so under vanilla i runs 0 .. T-1 across the whole Action and does reach the thousands. Two guards: **clamp i_end to T-1**, since the executor's `i < _n_wp(...)` check means it holds the last target once the plan runs out; and **compute per arm**, since `left_arm` and `right_arm` are independent plan dicts with different T.",
    ),
    (
        "Action vector layout: padding and prismatic columns",
        "`MAX_ACTION = 18`. Width is 2 x `spec.n_joints`, so 16 for the three 6-DOF embodiments and 18 for franka-panda, right-padded with zeros, built by differencing already-padded `qpos_raw` and writing only at `joint_index` columns. Prismatic gripper columns stay exactly zero: `joint_index` is -1 for the gripper token so the actor never writes them, and `step_dtheta` ignores them since grippers hold their current opening. The HJ actor therefore never opens or closes the gripper; grasp timing stays with the task script.",
        "Same layout, but a_nom must respect both properties. Because the padding applies to the concatenated two-arm vector, the right arm begins at column 8 for 6-DOF embodiments and column 9 for franka \u2014 **there is no fixed per-arm stride**, so assemble using the embodiment's own `spec.n_joints` and then pad. A hardcoded stride would silently write the right arm into the wrong columns for three of the four embodiments. The prismatic columns must also stay zero: the critic forms each link's Cartesian effect as `Jlin @ dtheta` summed over all columns and the prismatic Jacobian columns are non-zero, so a stray value there would make the critic attribute real motion to a component the environment will not execute. The planner's `position` array is arm-only, since grippers go through a separate `set_gripper` stream, so this is safe by default \u2014 the risk is only if a_nom is built by differencing `qpos_raw`, which does include gripper channels.",
    ),
    (
        "Episode scheduling",
        "Three independent `rng.choice` calls in `sample_scene` for task, embodiment and obstacle; IID sampling over 25 episodes leaves the 20 (task, embodiment) combinations badly unbalanced. Every episode was on_path with no filter, so neither 80/20 split existed. A spawn failure returned `(None, None, None, arm)` and the episode simply ran with `env.obstacle = None`.",
        "A fixed cyclic schedule over the shuffled 5 x 4 product, so task and embodiment counts are exactly equal; the obstacle axis disappears since there is now one model. Two further draws are made independently per episode \u2014 obstacle placement (80% on_path / 20% off_path) and behavior policy (80% with switching / 20% nominal only) \u2014 giving four episode types. Keeping them independent avoids confounding the filter engaging with the obstacle being in the way. Episodes skipped by the tier-3 spawn failure in section 1 must be re-drawn into the same schedule slot or the balance quietly breaks, and skips should be counted per round split by mode, since off_path is far more prone to failure than on_path given the 0.22 m lateral offset against `TABLE_XYLIM` bounds of [-0.48, 0.48] x [-0.32, 0.28].",
    ),
    (
        "Planner's world model",
        "`make_training_env` spawns the obstacle physically, then calls `update_curobo_world(env.robot)` with no obstacle argument, so cuRobo's world contains only the table. The expert plans a collision-free path as if the obstacle were absent and drives into it. There is no termination on violation, so the rollout keeps recording after penetration.",
        "Unchanged.",
    ),
]

BUFFER = [
    (
        "New fields: task_id and action_source",
        "`FIELDS` stores `embodiment_id` as an int8 but nothing identifying the task, so per-task metrics cannot be computed offline at all. Action origin is likewise unrecorded, and since every action was an achieved displacement they are indistinguishable by source anyway.",
        "Two int8 fields. `task_id` indexes the 5-task list, enabling free per-task splits of h, Q and the h - Q gap at every gradient step since it rides along in the sampled batch. `action_source` marks planner / actor / perturbation, letting us audit how much of the buffer sits in the actor's action box versus planner scale, and diagnose distribution shift as the DAgger rounds progress.",
    ),
    (
        "Capacity and schema",
        "`cfg.n_episodes * (cfg.max_steps_per_episode + 2)`, sized for one pass: 25 x 402 \u2248 10,050 steps \u2248 6.3 GB. The header records `shapes` and `capacity`, and `append` refuses a shape change after step 0.",
        "Same formula with the new tick budget, `n_episodes * 452`, but sized across all collection rounds up front since `open_memmap` allocates at fixed capacity and `append` asserts against it. At ~630 KB per step \u2014 dominated by the `[1200, 256]` fp16 scene tokens at 614 KB \u2014 200 episodes is roughly 57 GB. Adding fields, changing `h_scale`, redefining h, changing the control rate and changing the action convention each invalidate every existing buffer, so this starts from fresh collection regardless.",
    ),
]

LOGGING = [
    (
        "Video panel",
        "`PanelVideoWriter` renders a 320 x 480 panel beside the observer frame, 960 x 480 output, showing t, step index, h, V, h - V, control mode as NOMINAL / FILTER, |\u03b4\u03b8| as a fraction of max, and a scrolling h/V mini-plot of the last ~100 samples, at `video_fps = 10.0` frames per simulated second \u2014 described in a code comment as every 2nd tick, true only at 20 Hz.",
        "Adds, in cm: `dL`, `dR`, `dLh`, `dRh` formatted exactly as `record.py` does (`f\"{val*100:.1f}cm\"`, `\"inf\"` when non-finite); the `true_argmin` link name; per-arm holding state with provenance (`L=bread(contact) R=none(arm_open)`); the plan stage from `task._cehj_skill`; the HJ block counter as tick-in-block out of 3 plus cumulative seconds since engagement; and |a_nom| / d\u03b8_max. The frame rate also needs fixing: at 25 Hz every 2nd tick is 12.5 fps, so the constant and the comment disagree \u2014 replace with an explicit integer stride, every 2 ticks for 12.5 fps or every 5 for 5 fps. Recommend every 2, since intervention blocks are only 3 ticks long and a coarser stride could skip an entire engagement.",
    ),
    (
        "Distance plumbing and panel self-consistency",
        "`compute_h` already calls `distance_info(env)` once per control tick and receives all these fields, but its returned `diag` keeps only h, d_block, d_table, d_held, d_arm_arm, per_link, true_argmin, holding and contact \u2014 the per-arm and per-held-object terms are dropped on the floor. Separately, because h = min(d_block, d_table) while the panel showed only obstacle distances, h sat well below all four displayed numbers whenever the gripper descended toward the table and the panel appeared to contradict itself; a `d_table` readout was going to be added for that reason.",
        "Forward `d_left`, `d_right`, `d_left_held`, `d_right_held`, `holding_left`, `holding_right`, and the `hold_debug` fields `L_source` / `R_source` / `L_fail` / `R_fail`. **Zero extra simulator cost** \u2014 the expensive part, contact queries and hold detection, is already paid every tick. No `d_table` readout is needed once the table term is gone: h = d_block = min(dL, dR, dLh, dRh) is exactly the minimum of the four displayed distances at every frame, so the panel becomes self-consistent by construction and doubles as a check that the plumbing is correct.",
    ),
    (
        "Sampling rate of h",
        "In the sweeps `distance_info` ran on every one of the 250 Hz `scene.step()` calls, about 2720 per episode at 2.6 ms each, so `d_min` was a minimum over the full trajectory.",
        "Now once per 25 Hz control tick, about 272 per episode \u2014 a 10x reduction rather than the 12.5x it would have been at 20 Hz. At 0.3 m/s the arm covers 12 mm between samples instead of 15 mm, so the upward bias in recorded minimum h shrinks slightly but does not go away. If it matters, keep a min-tracker on the 250 Hz hook and have the tick read the running minimum instead of an instantaneous sample.",
    ),
    (
        "Plan stage",
        "Already live and unused. `ResidualController.attach()` calls `_install_skill_tracker(self.env.task)`, which stamps every `play_once` Action and keeps `task._cehj_skill` in sync, producing strings like \"L pre_grasp bread | R lift dz=0.1\".",
        "Read it in `_capture_tick` and display it. No new machinery.",
    ),
    (
        "Camera streams",
        "Observer camera only, via `cameras.observer_camera`.",
        "Unchanged.",
    ),
    (
        "Training-loop metrics",
        "`grad_step` returns global `mean_q`, `mean_h`, `mean_gap_h_q`, plus losses, grad norms, alpha, gamma and buffer size, with no per-embodiment or per-task breakdown.",
        "The same three quantities additionally split by `embodiment_id` and the new `task_id`. Free, since both fields arrive in the sampled batch \u2014 a masked mean over existing tensors, no extra forward passes. **What cannot go here:** per-episode intervention, success and collision rates. Training is fully offline \u2014 `grad_step` samples a static memmap and never touches the simulator \u2014 so there are no episodes in the gradient loop to measure, and an offline \"collision rate\" would just be the fraction of dataset transitions with h < 0: a fixed property of the buffer that never improves, and actively misleading as a progress signal. Those ride the rollout-sweep cadence instead.",
    ),
    (
        "Rollout sweeps and eval videos",
        "No debug rollouts during training. The only simulator contact is `evaluate()`, which runs one filtered episode on a random scene every `eval_every = 1024` grad steps, writing a single video to `runs/videos/eval_{mode}_{emb}_{task}_s{seed}_step{N}.mp4`.",
        "5 rollouts per sweep, one per task, with the embodiment rotating across sweeps so all 20 combinations are covered every 4 sweeps. Single camera. Triggered at training start and every 10 epochs. Budget roughly 10-15 minutes per sweep \u2014 the earlier 8-12 minute estimate scaled up for 25% more ticks per episode. Eval matches: one video per task, same panel, same per-(task, embodiment) metrics.",
    ),
    (
        "Success / collision / intervention",
        "Eval only, one random (task, embodiment, obstacle) triple per eval, reported as flat aggregates in `trace[\"metrics\"]`.",
        "Computed per (task, embodiment) from the rollout sweeps and held as an EMA, so the wandb panels stay populated between sweeps instead of going sparse. Calibration note: the obstacle-blind expert baseline is 17/64 success and 53/64 contact, so success rate is a noisy, low-baseline signal needing many episodes before it means anything.",
    ),
]


# Part II: frozen-encoder input parity. Paths relative to
# /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/ ; OURS = CEHJ/, HB = RoboOrchardLab/.
ENCODER = [
    (
        "Image resolution",
        "`HB/projects/holobrain/configs/config_holobrain_gd_common.py:43` sets `dst_wh=(320, 256)`. The deploy transform `Resize` (`HB/projects/holobrain/configs/config_robotwin_dataset.py:171-174`) resizes every camera to it and **rescales the intrinsics** by the same factors (`HB/robo_orchard_lab/dataset/robotwin/transforms.py:493-500`).",
        "`OURS/main/envs/env.py` `get_encoder_obs()` uses the D435 native 320x240 (`OURS/RoboTwin/env_cfg/task_config/_camera_config.yml:11-14`) with **no resize** and no intrinsic rescale. `image_wh` is taken from the raw frame at `env.py:470`.",
        "Height is 240 where the model expects 256, so fy/cy are off by 256/240 and the feature grid has the wrong number of rows. **Add the resize + intrinsic rescale to `get_encoder_obs`.** Verified directly.",
    ),
    (
        "Scene-token count (proof of the above)",
        "The four pyramid levels are strides **(8, 16, 32, 64)**, stated outright as `depth_gt_stride` under `multi_task=True` (`HB/.../config_holobrain_gd_common.py:131-136`) and consistent with Swin `out_indices=(1,2,3)` plus `ChannelMapper(num_outs=4)` (`:171`, `:183`). Both sides select `feature_level = [1, 2]` (`HB/.../config_holobrain_gd_common.py:372`), i.e. **strides 16 and 32**. At 320x256 that is 20x16 + 10x8 = 400 tokens per camera; with 3 cameras, exactly **1200**.",
        "`OURS/main/train/buffer.py:37-38` hardcodes `scene_tokens (1200, 256)` and `scene_pos (1200, 3)` \u2014 no padding constant, unlike `MAX_STATE_TOKENS` / `MAX_ACTION`. `OURS/main/network/encoder.py:387` uses the same `feature_level=(1, 2)`.",
        "1200 is only reachable at 320x**256** with 3 cameras. At our 320x240 the count is 20x15 + 10x8 = 380 per cam = **1140**, and `append` would assert. Collection has never run (CKPT_DIR unreachable), so this has stayed latent. **This constant is the proof that the resize is missing.** Verified directly.",
    ),
    (
        "projection_mat coordinate frame",
        "`GetProjectionMat(target_coordinate=\"ego\")` (`HB/.../config_robotwin_dataset.py:178`) builds P = K @ T_world2cam @ T_base2world @ inv(T_base2ego) (`HB/robo_orchard_lab/dataset/robotwin/transforms.py:847-871`). `MoveEgoToCam` sets T_base2ego from the **last** camera in `cam_names` (`transforms.py:63-65`), i.e. the head. So 3D lifting happens in **head-camera (ego) frame**.",
        "`OURS/main/envs/env.py:453-457` builds P = K @ T_world2cam and stops, so lifting is in **world frame**. The `get_encoder_obs` docstring states this is deliberate (\"the encoders and body features stay in the world frame\").",
        "**Highest-risk item.** The spatial enhancer back-projects depth with P and then embeds the resulting 3D coordinates with weights trained on ego-frame values; world coordinates are offset by the base pose, so the learned position embedding runs far out of distribution \u2014 silently. **Fix that keeps both properties:** feed the encoder ego-frame P (matching training), then transform the returned positions to world frame ourselves for the trunk. Verified directly.",
    ),
    (
        "Extrinsic direction and naming",
        "Field is `t_world2cam` (`HB/robo_orchard_lab/models/holobrain/processor.py:58-60`). The raw RoboTwin path uses `extrinsic_cv` **directly**, no inverse (`HB/projects/holobrain/policy/robotwin_policy.py:111-115`); the formatted path inverts a camera-in-world pose (`robotwin_policy.py:159-190`).",
        "`OURS/main/envs/env.py:391-402` `camera_extrinsic()` returns `cam.get_extrinsic_matrix()`, which is also world\u2192camera. The obs key is named `T_cam_world` (`env.py:472`).",
        "**Direction MATCHES** \u2014 this is the one geometry convention we got right. Only the *name* is inverted relative to its semantics; rename to `T_world2cam` to stop it misleading the next reader. Verified directly.",
    ),
    (
        "Camera set and ordering",
        "Two processors exist: a 4-camera list `[front, left, right, head]` (`HB/.../config_robotwin_dataset.py:55-60`) and a 3-camera `robotwin2_0_ur5_wsg` list `[left_camera, right_camera, head_camera]` (`:112`). The processor stacks strictly in that order (`HB/robo_orchard_lab/models/holobrain/processor.py:118-143`).",
        "`OURS/main/envs/env.py:441-445` uses `[head, left_wrist, right_wrist]`, labelled at `env.py:477`. ARX-X5 / piper configs have `front_camera` commented out (`OURS/RoboTwin/assets/embodiments/ARX-X5/config.yml:27-40`), so only 3 exist.",
        "The 1200-token arithmetic implies **3 cameras**, so the ur5_wsg processor is our reference \u2014 but its order puts head **last** and ours puts it **first**. Camera index is preserved through backbone and spatial enhancer with no name-based reordering, and the ego anchor is the last camera. **Reorder to [left, right, head].** Arithmetic verified; which processor the v0.0_GD checkpoint exports is undetermined without the checkpoint on disk.",
    ),
    (
        "feature_enhancer / text branch",
        "GD config sets `multi_task=True` (`:45`), which builds a `TextImageDeformable2DEnhancer` (`:202`) and a text encoder. `BIP3D._forward` runs text extraction \u2192 **feature_enhancer** \u2192 spatial_enhancer \u2192 decoder (`HB/robo_orchard_lab/models/bip3d/structure.py:131-166`). The processor maps a missing instruction to `\"\"`, which is still tokenized (`processor.py:145-147`).",
        "`OURS/main/network/encoder.py:246-260` loads only `backbone`, `neck`, `backbone_3d`, `neck_3d`, `spatial_enhancer` and `decoder.robot_encoder`. There is **no** `feature_enhancer`, no `text_encoder`, and no instruction anywhere in `get_encoder_obs`.",
        "We run correctly-loaded spatial-enhancer weights on inputs that never passed the fusion stage they were trained after. Dropping language from a safety filter is defensible; feeding a frozen module out-of-distribution inputs is a separate question. **Decide empirically:** either load and run `feature_enhancer` with a fixed instruction, or measure the token-distribution shift and accept it. Verified directly.",
    ),
    (
        "Depth acquisition and masking",
        "`HB/projects/holobrain/policy/robotwin_policy.py:109` takes `camera_data[\"depth\"] / 1000`. That observation is built by `OURS/RoboTwin/envs/camera/camera.py:412-418`, `depth_image = -position[...,2] * 1000.0`, and every consumer line then applies `res[cam][\"depth\"] *= rgba[cam][\"rgba\"][:,:,3] / 255` (`:433-434`, `:441`, `:445`). Net input to the model is therefore **-pos[...,2] x (alpha/255)** in metres.",
        "`OURS/main/envs/env.py:451-452` reads `cam.get_picture(\"Position\")` directly and takes `-pos[:,:,2]` in metres. No `/1000` needed, but also **no alpha mask**.",
        "Units, sign and shape agree \u2014 the round trip x1000 then /1000 cancels, so no scale bug and no uint16 quantization (they cast to float64). The single difference is the **alpha factor we omit**: background pixels are exactly 0 for them and whatever the position buffer holds for us, and partially transparent materials get their depth scaled down. Depth feeds `backbone_3d` and hence `feature_3d`, so this shifts the enhancer's inputs rather than being cosmetic. **Multiply by `alpha/255` in `get_encoder_obs`**, taking alpha from the same `Color` picture as the RGB. Verified directly on both sides.",
    ),
    (
        "Normalization and channel order",
        "`ImageChannelFlip([2,1,0])` in deploy (`config_robotwin_dataset.py:175`) then `BaseDataPreprocessor(channel_flip=True, mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375])` (`config_holobrain_gd_common.py:140-157`), applied on float32 **0-255** with HWC\u2192CHW (`HB/robo_orchard_lab/models/layers/data_preprocessors.py:116-157`).",
        "`OURS/main/envs/env.py:386-388` defines the same mean/std and applies them at `env.py:463-464` on float32 0-255, permuting to `[B, n_cam, 3, H, W]` at `:468`. No channel flips.",
        "**Effectively MATCHES** for live SAPIEN RGB: their two flips cancel, so both end up RGB with identical mean/std and layout. No change needed \u2014 but do not feed BGR anywhere, since we have no flip to compensate. Verified directly.",
    ),
    (
        "joint_state source",
        "`robotwin_policy.py:123-128` passes `joint_action[\"vector\"]`, which RoboTwin builds from `get_left_arm_jointState()` / `get_right_arm_jointState()` \u2014 these read **drive targets** (`OURS/RoboTwin/envs/robot/robot.py:510-514`).",
        "`OURS/main/envs/env.py:465-475` uses `get_left_arm_real_jointState()` / `get_right_arm_real_jointState()`, which read **measured qpos** (`RoboTwin/envs/robot/robot.py:524-531`).",
        "The robot-state encoder was trained on commanded values; we feed achieved ones. Same commanded-vs-achieved theme as the action side, and largest exactly during contact. Layout, radians and the trailing normalized gripper scalar all match. **Switch to the drive-target getters.** Verified directly.",
    ),
    (
        "joint_scale_shift / joint_mask / embodiedment_mat",
        "The decoder applies `apply_scale_shift(hist_robot_state, joint_scale_shift)` (`HB/robo_orchard_lab/models/holobrain/action_decoder.py:445-447`); deploy sets `joint_mask = ([True]*6 + [False])*2` so gripper joints are masked in the robot encoder; FK uses `embodiedment_mat` = T_base2ego from `GetProjectionMat`.",
        "`OURS/main/network/encoder.py:308-311` calls `encode_robot_state(robot_state, kinematics.joint_relative_pos, joint_mask)` with raw angles, **no** scale/shift, and `joint_mask` is never populated by the rollout. No `embodiedment_mat` is used.",
        "Three separate omissions on the same path, all silent. Couples to the ego-frame item above. **Verify against the checkpoint's exported processor JSON before changing anything** \u2014 reported by scan, not re-verified by me.",
    ),
    (
        "Model loading and invocation",
        "`HoloBrainInferencePipeline.load_pipeline(..., load_impl=\"native\", model_prefix=\"model\")` (`HB/projects/holobrain/scripts/robotwin_eval.py:163-170`) builds the whole graph from `model.config.json`, then `pipeline.model.eval()` and `@torch.inference_mode()`.",
        "`OURS/main/network/encoder.py:246-260` calls `safetensors.torch.load_file` and strips prefixes into a partial wrapper, then `.to(device, dtype).eval()`; every `encode_*` is `@torch.no_grad()`. Default dtype fp32.",
        "eval mode, no-grad and fp32 all **MATCH** (their GD path has no autocast; `drop_rate = 0` so no stochastic layers). The only real difference is the subset of modules loaded, already covered by the feature_enhancer row. Verified directly.",
    ),
]


def build() -> Path:
    doc = BaseDocTemplate(
        str(OUT), pagesize=PAGE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 6,
        title="HJ-SAC Safety Filter - Change List",
        author="CEHJ",
    )
    frame = Frame(
        MARGIN, MARGIN + 6, USABLE, PAGE[1] - 2 * MARGIN - 6, id="body",
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(BODY, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, MARGIN - 4, "CEHJ \u2014 HJ-SAC safety filter change list")
        canvas.drawRightString(
            PAGE[0] - MARGIN, MARGIN - 4, f"page {canvas.getPageNumber()}"
        )
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])

    story = [
        Paragraph("HJ-SAC Safety Filter \u2014 Change List", st_title),
        Paragraph(
            "cross_embodiment/CEHJ &nbsp;\u00b7&nbsp; every row states the current behaviour "
            "and the intended replacement. Line references are to the code as it stands today. "
            "<b>Part I</b> is the training-pipeline change list. <b>Part II</b> is a separate set of "
            "frozen-encoder input mismatches found later by scanning HoloBrain in RoboOrchardLab; those are "
            "silent and more damaging, and should be fixed first even though they appear second.<br/>"
            "Headline settings: h = obstacle distance only, never infinite \u00b7 25 Hz control "
            "(exactly 10 physics steps per tick) \u00b7 single 1 cm margin \u00b7 3-tick HJ blocks "
            "(0.12 s) \u00b7 4500 physics steps before timeout \u00b7 5 tasks x 4 embodiments \u00b7 "
            "actor bounded to 25% of joint speed limit (\u03ba = 0.25) over each 40 ms tick.",
            st_sub,
        ),
        Paragraph(
            "<b>Prerequisite (not a code change).</b> " + fmt(
                "`CKPT_DIR` in `main/train/collect.py` and `tests/test_encoder_step.py` points at "
                "`/root/autodl-tmp/HoloBrain/HoloBrain_v0.0_GD`, an AutoDL path that returns permission "
                "denied on this cluster; likewise `/root/autodl-tmp/RoboTwin/...` in "
                "`tests/gen_franka_dualarm_urdf.py`. These need real paths before collection can run."
            ),
            st_note,
        ),
        Paragraph(
            "PART I \u2014 Training-pipeline changes (h, control rate, actor execution, "
            "switching, collection, buffer, logging)",
            st_part,
        ),
        Paragraph(fmt("1. Hazard function \u2014 main/train/hfunc.py, main/envs/obstacle.py"), st_sec),
        Paragraph(
            "h is the safety specification the value function learns, so this is the change "
            "everything else is calibrated against. It must be finite on every stored transition.",
            st_lede,
        ),
        table(HFUNC),
        Paragraph(fmt("2. Config \u2014 main/train/config.py"), st_sec),
        Paragraph(
            "The first two rows are coupled: control_freq, control_dt and delta_theta_max "
            "have to move together.",
            st_lede,
        ),
        table(CONFIG),
        Paragraph(fmt("3. Actor execution: turning one 25 Hz \u03b4\u03b8 into 10 physics steps \u2014 main/envs/env.py"), st_sec),
        Paragraph(
            "The actor predicts a displacement over a 40 ms tick, but the environment currently applies it as a "
            "held position target with zero commanded velocity, so only about a fifth of it happens. This section "
            "is self-contained and can be implemented independently of everything else.",
            st_lede,
        ),
        table(ACTOR_EXEC),
        Paragraph(fmt("4. Switching \u2014 main/train/filter.py, main/train/rollout.py"), st_sec),
        table(SWITCHING),
        Paragraph(fmt("5. Collection \u2014 main/train/collect.py, main/train/rollout.py"), st_sec),
        table(COLLECTION),
        Paragraph(fmt("6. Buffer \u2014 main/train/buffer.py"), st_sec),
        table(BUFFER),
        Paragraph(fmt("7. Logging \u2014 main/train/hfunc.py, eval_utils.py, train.py"), st_sec),
        table(LOGGING),
        Paragraph("8. Diagnostics to add while we are in there", st_sec),
        Paragraph(
            "<b>Realized versus commanded displacement, per tick, as a ratio against d\u03b8_max.</b> "
            + fmt(
                "We are now deliberately excluding tracking error from the stored action, so we should "
                "measure how large it is rather than assume it is small. This also quantifies the one "
                "inconsistency that survives the change: the two controllers have different intra-tick "
                "profiles for the same net command \u2014 cuRobo ramps through 10 intermediate targets "
                "while `step_dtheta` sets a single target and holds it \u2014 so identical \u03b4\u03b8 values "
                "produce slightly different realized motion depending on who issued them. If that turns "
                "out to be large, Q(s, a) is mildly controller-dependent and we would want to build a_nom "
                "from a one-tick forward simulation of the drives rather than the raw spline."
            ),
            st_note,
        ),
        Paragraph(
            "<b>Mode-switch count per episode.</b> " + fmt(
                "Hysteresis is gone and the commitment block is only 3 ticks, so the filter can legally "
                "flip every 0.12 s. This is the number that tells us whether that matters in practice. If "
                "it is high, lengthen `hj_hold_ticks` rather than reintroducing a second margin \u2014 the "
                "block is doing the same job with one fewer parameter."
            ),
            st_note,
        ),
        Paragraph(
            "<b>Action-magnitude histogram by action_source.</b> " + fmt(
                "The actor is bounded to \u03ba = 0.25 of each joint's velocity limit per tick, while cuRobo "
                "plans against the full limit, so a_nom can exceed d\u03b8_max several times over by "
                "construction. Tracking the distribution per source shows how much of the critic's "
                "evaluation region is covered by data versus extrapolated, and whether the DAgger rounds "
                "are filling in the actor's small-action region."
            ),
            st_note,
        ),
        Paragraph(
            "<b>Fraction of ticks with an arm link below the table plane.</b> " + fmt(
                "Cheap to compute and no longer part of h, but it is the direct measurement of the risk "
                "accepted by removing the table term. If the actor starts driving links through the tabletop "
                "this is the number that will show it first, well before it shows up as a strange-looking video."
            ),
            st_note,
        ),
        Spacer(1, 10),
        Paragraph(
            "PART II \u2014 Frozen-encoder parity with HoloBrain (found in the "
            "RoboOrchardLab scan, after Part I was written)",
            st_part,
        ),
        Paragraph(
            "Part I assumes the frozen HoloBrain-0 encoder is being driven correctly. This part checks that "
            "assumption and finds it is not. We reuse HoloBrain-0 as a frozen encoder, so our observation dict must "
            "be constructed the way their inference pipeline constructs it; every mismatch below is silent \u2014 no "
            "exception, just degraded scene tokens feeding the trunk, which the geometric trunk then treats as "
            "metrically true. Nothing in Part I is worth running until these are settled. "
            "All paths are relative to <b>/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/</b>, "
            "with <b>OURS = CEHJ/</b> and <b>HB = RoboOrchardLab/</b>. Each row ends with whether I verified it "
            "directly in the source or am relaying it from the automated scan.",
            st_lede,
        ),
        Paragraph(fmt("9. Observation and encoder-input mismatches"), st_sec),
        table4(ENCODER),
        Paragraph("10. How to settle all of Part II at once", st_sec),
        Paragraph(
            "<b>Write one parity test.</b> " + fmt(
                "Build a single observation two ways from the same simulator state \u2014 once through HoloBrain's own "
                "processor (`HB/robo_orchard_lab/models/holobrain/processor.py` plus the deploy transform chain in "
                "`HB/projects/holobrain/configs/config_robotwin_dataset.py`) and once through our "
                "`OURS/main/envs/env.py::get_encoder_obs` \u2014 then assert tensor-by-tensor agreement on `imgs`, "
                "`depths`, `image_wh`, `projection_mat` and `joint_state`. That converts every row above from an "
                "argument into a pass/fail, including the three I could not verify statically, and it will catch "
                "whichever processor variant the v0.0_GD checkpoint actually exports. It needs a reachable "
                "`CKPT_DIR`, which is the blocker already listed at the top of Part I."
            ),
            st_note,
        ),
        Paragraph(
            "<b>Priority order.</b> " + fmt(
                "projection_mat frame first (it corrupts every token position, and the trunk consumes those as "
                "metric truth); then image resolution together with the 1200-token constant, since they are the "
                "same bug and one of them will hard-fail on the first `append`; then the missing `feature_enhancer`; "
                "then camera ordering; then the joint-state source. The zero-velocity actor bug in Part I section 3 "
                "is real but sits below all of these \u2014 it corrupts the action, whereas these corrupt the state."
            ),
            st_note,
        ),
        Spacer(1, 4),
    ]
    doc.build(story)
    return OUT


if __name__ == "__main__":
    print(build())
