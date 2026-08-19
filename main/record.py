"""Record observer / wrist videos with distance overlay and optional debug bboxes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from distance import (
    detect_held_object,
    distance_info,
    is_gripper_link,
    load_collision_spheres,
    obstacle_corners,
    sphere_aabb_corners,
    spheres_with_names,
)


def observer_rgb(task) -> np.ndarray:
    cam = task.cameras.observer_camera
    cam.take_picture()
    rgba = cam.get_picture("Color")
    return (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]


def _put_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = np.ascontiguousarray(frame.copy())
    try:
        import cv2

        y = 18
        for line in lines:
            cv2.putText(
                img,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 16
        return img
    except Exception:
        return img


def _project(cam, pts: np.ndarray) -> np.ndarray:
    """Project world points with SAPIEN OpenCV extrinsics (z forward)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    if hasattr(cam, "get_extrinsic_matrix"):
        ext = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
        if ext.shape == (3, 4):
            pc = (ext @ hom.T).T
        else:
            pc = (ext @ hom.T).T[:, :3]
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        depth = z
        u_n = x / np.maximum(depth, 1e-8)
        v_n = y / np.maximum(depth, 1e-8)
    else:
        pose = cam.entity.get_pose() if hasattr(cam, "entity") else cam.get_pose()
        T_inv = np.linalg.inv(pose.to_transformation_matrix())
        pc = (T_inv @ hom.T).T[:, :3]
        # Entity pose is OpenGL: +X forward, +Y left, +Z up.
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        depth = x
        u_n = -y / np.maximum(depth, 1e-8)
        v_n = -z / np.maximum(depth, 1e-8)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = depth > 1e-4
    uv[valid, 0] = K[0, 0] * u_n[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * v_n[valid] + K[1, 2]
    return uv


_CUBE_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


def _draw_cube(frame: np.ndarray, uv: np.ndarray, color) -> np.ndarray:
    img = np.ascontiguousarray(frame.copy())
    try:
        import cv2

        h, w = img.shape[:2]
        for i, j in _CUBE_EDGES:
            if i >= len(uv) or j >= len(uv):
                continue
            u0, v0 = uv[i]
            u1, v1 = uv[j]
            if not (np.isfinite(u0) and np.isfinite(v0) and np.isfinite(u1) and np.isfinite(v1)):
                continue
            if (u0 < -w and u1 < -w) or (u0 > 2 * w and u1 > 2 * w):
                continue
            if (v0 < -h and v1 < -h) or (v0 > 2 * h and v1 > 2 * h):
                continue
            p0 = (int(round(u0)), int(round(v0)))
            p1 = (int(round(u1)), int(round(v1)))
            cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def draw_debug_bboxes(env, frame: np.ndarray) -> np.ndarray:
    cam = env.task.cameras.observer_camera
    img = frame
    if getattr(env, "obstacle", None) is not None:
        corners = obstacle_corners(env.obstacle, getattr(env, "obstacle_half", None))
        img = _draw_cube(img, _project(cam, corners), (0, 255, 0))
    held_actor, _holding = detect_held_object(env)
    if held_actor is not None:
        img = _draw_cube(img, _project(cam, obstacle_corners(held_actor)), (255, 0, 255))
    try:
        spheres = load_collision_spheres(env.embodiment)
    except Exception:
        return img
    arm_i = 0
    for entity in (env.robot.left_entity, env.robot.right_entity):
        for center, radius, link_name in spheres_with_names(entity, spheres):
            if is_gripper_link(link_name):
                img = _draw_cube(img, _project(cam, sphere_aabb_corners(center, radius)), (0, 220, 255))
                continue
            if arm_i % 3 == 0:
                img = _draw_cube(img, _project(cam, sphere_aabb_corners(center, radius)), (255, 80, 80))
            arm_i += 1
    return img


HOLD_CSV_FIELDS = [
    "step",
    "recorded",
    "holding",
    "hold_source",
    "hold_fail",
    "hold_side",
    "near",
    "near_reason",
    "contact_grip_obj",
    "left_grip",
    "right_grip",
    "left_holding",
    "right_holding",
    "origin_dist_ee",
    "origin_dist_tcp",
    "ee_obb",
    "tcp_obb",
    "grip_obb",
    "L_origin_dist_ee",
    "L_origin_dist_tcp",
    "L_tcp_obb",
    "L_grip_obb",
    "L_near",
    "R_origin_dist_ee",
    "R_origin_dist_tcp",
    "R_tcp_obb",
    "R_grip_obb",
    "R_near",
    "d_robot",
    "d_held",
    "d_sys",
    "contact_obstacle",
    "closest",
    "obj_x",
    "obj_y",
    "obj_z",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "ee_x",
    "ee_y",
    "ee_z",
]


def _hold_csv_row(step: int, recorded: int, dbg: dict, info: dict | None = None) -> dict:
    row = {key: "" for key in HOLD_CSV_FIELDS}
    row["step"] = step
    row["recorded"] = recorded
    row["holding"] = dbg.get("holding") or (info or {}).get("holding") or ""
    row["hold_source"] = dbg.get("source", "")
    row["hold_fail"] = dbg.get("fail", "")
    row["hold_side"] = dbg.get("side", "")
    row["near"] = int(bool(dbg.get("near")))
    row["near_reason"] = dbg.get("near_reason", "")
    row["contact_grip_obj"] = int(bool(dbg.get("contact_grip_obj")))
    row["left_grip"] = dbg.get("left_grip", "")
    row["right_grip"] = dbg.get("right_grip", "")
    row["left_holding"] = int(bool(dbg.get("left_holding")))
    row["right_holding"] = int(bool(dbg.get("right_holding")))
    for key in (
        "origin_dist_ee",
        "origin_dist_tcp",
        "ee_obb",
        "tcp_obb",
        "grip_obb",
        "L_origin_dist_ee",
        "L_origin_dist_tcp",
        "L_tcp_obb",
        "L_grip_obb",
        "L_near",
        "R_origin_dist_ee",
        "R_origin_dist_tcp",
        "R_tcp_obb",
        "R_grip_obb",
        "R_near",
        "obj_x",
        "obj_y",
        "obj_z",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "ee_x",
        "ee_y",
        "ee_z",
    ):
        val = dbg.get(key, "")
        if isinstance(val, bool):
            row[key] = int(val)
        else:
            row[key] = val
    if info is not None:
        row["holding"] = info.get("holding") or row["holding"]
        row["d_robot"] = info.get("d_robot", "")
        row["d_held"] = info.get("d_held", "")
        row["d_sys"] = info.get("d_system", "")
        row["contact_obstacle"] = int(bool(info.get("contact")))
        row["closest"] = info.get("closest", "")
    return row


def write_hold_trace(rows: list[dict], out_path: Path) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HOLD_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HOLD_CSV_FIELDS})
    print(f"Saved {len(rows)} hold-trace rows to {out_path}")
    return out_path


def attach_recorder(env, streams: dict, record_every: int, overlay: dict, draw_bbox: bool):
    state = {
        "n": 0,
        "d_min": [],
        "d_robot": [],
        "d_system": [],
        "d_held": [],
        "holding": [],
        "contact": [],
        "csv_rows": [],
    }
    orig_step = env.task.scene.step
    state["orig_step"] = orig_step

    def step():
        orig_step()
        state["n"] += 1
        recorded = state["n"] % max(record_every, 1) == 0
        if not recorded:
            try:
                detect_held_object(env)
                dbg = dict(getattr(env, "_cehj_hold_debug", {}) or {})
                state["csv_rows"].append(_hold_csv_row(state["n"], 0, dbg))
            except Exception as exc:
                if state["n"] <= 2 or state["n"] % 50 == 0:
                    print(f"\n[record] hold debug failed at step {state['n']}: {exc}")
            return
        try:
            info = distance_info(env)
        except Exception as exc:
            print(f"\n[record] distance_info failed at frame {state['n']}: {exc}")
            return
        d_robot = info["d_robot"]
        d_sys = info["d_system"]
        holding = info["holding"]
        contact = info["contact"]
        state["d_min"].append(d_sys)
        state["d_robot"].append(d_robot)
        state["d_system"].append(d_sys)
        state["d_held"].append(info["d_held"])
        state["holding"].append(holding)
        state["contact"].append(contact)
        dbg = dict(info.get("hold_debug") or getattr(env, "_cehj_hold_debug", {}) or {})
        state["csv_rows"].append(_hold_csv_row(state["n"], 1, dbg, info))
        obs = env.task.get_obs()["observation"]
        rgb = observer_rgb(env.task)

        def _cm(val):
            return "inf" if not np.isfinite(val) else f"{val * 100:.1f}cm"

        plan = overlay.get("plan_success")
        if plan is None:
            plan = getattr(env.task, "plan_success", None)
        hold_txt = holding if holding else "none"
        src = dbg.get("source") or "none"
        fail = dbg.get("fail") or ""
        hold_extra = f"src={src}"
        if fail:
            hold_extra += f" fail={fail}"
        tcp_obb = dbg.get("tcp_obb")
        if tcp_obb is not None and np.isfinite(tcp_obb):
            hold_extra += f" tcp_obb={_cm(tcp_obb)}"
        d_line = f"d_robot={_cm(d_robot)}  d_sys={_cm(d_sys)}"
        if holding:
            d_line += f"  d_held={_cm(info['d_held'])}"
        lines = [
            f"HOLDING={hold_txt}  {hold_extra}  contact={int(contact)} {info['closest']}",
            d_line,
            f"mode={overlay.get('obstacle_mode')} plan={overlay.get('plan_mode')} seed={overlay.get('seed')} expert={plan}",
        ]
        streams["agent_rgb"].append(_put_text(rgb, lines))
        if draw_bbox:
            streams["debug_bbox"].append(_put_text(draw_debug_bboxes(env, rgb), lines))
        streams["left_wrist_rgb"].append(np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8))
        streams["right_wrist_rgb"].append(np.asarray(obs["right_camera"]["rgb"], dtype=np.uint8))
        print(
            f"recorded {len(streams['agent_rgb'])} frames  "
            f"HOLDING={hold_txt} src={src} d_robot={_cm(d_robot)} d_sys={_cm(d_sys)}",
            end="\r",
        )

    env.task.scene.step = step
    return state


def detach_recorder(env, state: dict) -> None:
    orig = state.get("orig_step")
    if orig is not None:
        env.task.scene.step = orig


def save_video(frames, out_path: Path, fps: float) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"no frames for {out_path}")
    video = np.stack(frames, axis=0)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        for frame in video:
            writer.append_data(frame)
        writer.close()
        print(f"Saved {len(video)} frames to {out_path}")
        return out_path
    except Exception as exc:
        print(f"imageio failed ({exc}); trying OpenCV")

    import cv2

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (video.shape[2], video.shape[1]),
    )
    for frame in video[:, :, :, ::-1]:
        writer.write(frame)
    writer.release()
    print(f"Saved {len(video)} frames to {out_path}")
    return out_path
