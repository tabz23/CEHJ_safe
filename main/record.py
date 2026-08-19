"""Record observer / wrist videos with distance overlay and optional debug bboxes."""

from __future__ import annotations

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


def attach_recorder(env, streams: dict, record_every: int, overlay: dict, draw_bbox: bool):
    state = {
        "n": 0,
        "d_min": [],
        "d_robot": [],
        "d_system": [],
        "d_held": [],
        "holding": [],
        "contact": [],
    }
    orig_step = env.task.scene.step
    state["orig_step"] = orig_step

    def step():
        orig_step()
        state["n"] += 1
        if state["n"] % max(record_every, 1) != 0:
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
        obs = env.task.get_obs()["observation"]
        rgb = observer_rgb(env.task)

        def _cm(val):
            return "inf" if not np.isfinite(val) else f"{val * 100:.1f}cm"

        plan = overlay.get("plan_success")
        if plan is None:
            plan = getattr(env.task, "plan_success", None)
        hold_txt = holding if holding else "none"
        d_line = f"d_robot={_cm(d_robot)}  d_sys={_cm(d_sys)}"
        if holding:
            d_line += f"  d_held={_cm(info['d_held'])}"
        lines = [
            f"HOLDING={hold_txt}  contact={int(contact)} {info['closest']}",
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
            f"HOLDING={hold_txt} d_robot={_cm(d_robot)} d_sys={_cm(d_sys)}",
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
