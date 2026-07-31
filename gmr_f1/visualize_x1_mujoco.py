#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay an X1 retargeted NPZ clip in the interactive MuJoCo viewer.

The clip convention matches ``humanoid.algo.amp.motion_lib``:

* ``root_translation``: ``(N, 3)`` in metres
* ``root_rotation``: ``(N, 4)`` quaternion in xyzw order
* ``joint_positions``: ``(N, 12)`` in ``dof_names`` order
* ``foot_contact``: optional ``(N, 2)`` left/right contact flags
* ``fps``: scalar playback rate

On macOS the script automatically re-launches itself with the bundled
``mjpython`` interpreter required by ``mujoco.viewer.launch_passive``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_PYLIBS = REPO_ROOT / "gmr_f1" / "pylibs"
if str(VENDORED_PYLIBS) not in sys.path:
    sys.path.insert(0, str(VENDORED_PYLIBS))

import glfw
import mujoco
import numpy as np


DEFAULT_CLIP = REPO_ROOT / "data" / "retarget_gmr" / "x1_walk_retargeted.npz"
DEFAULT_MODEL = (
    REPO_ROOT / "resources" / "robots" / "x1" / "mjcf" / "xyber_x1_flat.xml"
)

X1_DOF_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an X1 retargeted NPZ motion in MuJoCo."
    )
    parser.add_argument(
        "--clip",
        type=Path,
        default=DEFAULT_CLIP,
        help=f"retargeted NPZ clip (default: {DEFAULT_CLIP})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"X1 MJCF model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="loop at the last frame (off by default because this clip is not loopable)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="initial playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="first frame to display (default: 0)",
    )
    parser.add_argument(
        "--no-follow-camera",
        action="store_true",
        help="do not keep the camera centred on the floating base",
    )
    parser.add_argument(
        "--hide-contact-markers",
        action="store_true",
        help="hide stance anchors and foot-slip connector lines",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.2,
        help="viewer camera distance in metres (default: 2.2)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=135.0,
        help="initial camera azimuth in degrees (default: 135)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-15.0,
        help="initial camera elevation in degrees (default: -15)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the clip/model mapping and exit without opening a window",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    return args


def ensure_mjpython_on_macos() -> None:
    """Re-exec under mjpython before importing mujoco.viewer on macOS."""
    if sys.platform != "darwin" or os.environ.get("MJPYTHON_BIN"):
        return

    mjpython = VENDORED_PYLIBS / "bin" / "mjpython"
    if not mjpython.is_file():
        raise RuntimeError(
            "MuJoCo passive viewer requires mjpython on macOS, but the bundled "
            f"launcher was not found: {mjpython}"
        )

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{VENDORED_PYLIBS}{os.pathsep}{old_pythonpath}"
        if old_pythonpath
        else str(VENDORED_PYLIBS)
    )
    os.execve(
        str(mjpython),
        [str(mjpython), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def load_clip(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion clip not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "root_translation",
            "root_rotation",
            "joint_positions",
            "fps",
            "dof_names",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")

        clip = {
            "path": path,
            "root_translation": np.asarray(
                archive["root_translation"], dtype=np.float64
            ),
            "root_rotation": np.asarray(archive["root_rotation"], dtype=np.float64),
            "joint_positions": np.asarray(
                archive["joint_positions"], dtype=np.float64
            ),
            "fps": float(archive["fps"]),
            "dof_names": [str(value) for value in archive["dof_names"]],
            "foot_contact": (
                np.asarray(archive["foot_contact"], dtype=np.float64)
                if "foot_contact" in archive.files
                else None
            ),
        }

    root_translation = clip["root_translation"]
    root_rotation = clip["root_rotation"]
    joint_positions = clip["joint_positions"]
    foot_contact = clip["foot_contact"]
    n_frames = root_translation.shape[0]

    expected_shapes = {
        "root_translation": (n_frames, 3),
        "root_rotation": (n_frames, 4),
        "joint_positions": (n_frames, len(X1_DOF_NAMES)),
    }
    for name, expected in expected_shapes.items():
        if clip[name].shape != expected:
            raise ValueError(
                f"{name} shape is {clip[name].shape}; expected {expected}"
            )
    if foot_contact is not None and foot_contact.shape != (n_frames, 2):
        raise ValueError(
            f"foot_contact shape is {foot_contact.shape}; expected {(n_frames, 2)}"
        )
    if clip["dof_names"] != X1_DOF_NAMES:
        raise ValueError(
            "dof_names do not match the X1 AMP joint order:\n"
            f"  clip:     {clip['dof_names']}\n"
            f"  expected: {X1_DOF_NAMES}"
        )
    if clip["fps"] <= 0 or not np.isfinite(clip["fps"]):
        raise ValueError(f"fps must be finite and positive; got {clip['fps']}")

    numeric_arrays = [root_translation, root_rotation, joint_positions]
    if foot_contact is not None:
        numeric_arrays.append(foot_contact)
    if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
        raise ValueError("clip contains NaN or Inf")

    quaternion_norm = np.linalg.norm(root_rotation, axis=1, keepdims=True)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError("root_rotation contains a zero-length quaternion")
    clip["root_rotation"] = root_rotation / quaternion_norm
    clip["n_frames"] = n_frames
    return clip


def build_joint_qpos_map(
    model: mujoco.MjModel, dof_names: list[str]
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for name in dof_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' was not found in the MJCF model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint '{name}' is not a hinge joint")
        mapping[name] = int(model.jnt_qposadr[joint_id])
    return mapping


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: dict[str, Any],
    qpos_by_name: dict[str, int],
    frame: int,
) -> None:
    data.qpos[:] = 0.0
    data.qpos[:3] = clip["root_translation"][frame]

    # Expert/MotionLib stores xyzw; MuJoCo free joints require wxyz.
    qx, qy, qz, qw = clip["root_rotation"][frame]
    data.qpos[3:7] = [qw, qx, qy, qz]
    for index, name in enumerate(clip["dof_names"]):
        data.qpos[qpos_by_name[name]] = clip["joint_positions"][frame, index]

    data.time = frame / clip["fps"]
    mujoco.mj_forward(model, data)


def compute_foot_positions_and_anchors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: dict[str, Any],
    qpos_by_name: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in FOOT_BODY_NAMES
    ]
    missing = [
        name for name, body_id in zip(FOOT_BODY_NAMES, foot_ids) if body_id < 0
    ]
    if missing:
        raise ValueError(f"Foot bodies were not found in the MJCF model: {missing}")

    foot_positions = np.zeros((clip["n_frames"], 2, 3), dtype=np.float64)
    for frame in range(clip["n_frames"]):
        apply_frame(model, data, clip, qpos_by_name, frame)
        foot_positions[frame] = data.xpos[foot_ids]

    anchors = np.full_like(foot_positions, np.nan)
    contact = clip["foot_contact"]
    if contact is None:
        return foot_positions, anchors

    for side in range(2):
        anchor = None
        previous_contact = False
        for frame in range(clip["n_frames"]):
            current_contact = bool(contact[frame, side] > 0.5)
            if current_contact and not previous_contact:
                anchor = foot_positions[frame, side].copy()
            if current_contact and anchor is not None:
                anchors[frame, side] = anchor
            if not current_contact:
                anchor = None
            previous_contact = current_contact
    return foot_positions, anchors


def add_contact_markers(
    viewer: Any,
    foot_positions: np.ndarray,
    anchors: np.ndarray,
    frame: int,
) -> None:
    viewer.user_scn.ngeom = 0
    colors = [
        np.array([0.10, 0.55, 1.00, 0.90], dtype=np.float32),
        np.array([1.00, 0.45, 0.10, 0.90], dtype=np.float32),
    ]
    identity = np.eye(3, dtype=np.float64).ravel()

    for side in range(2):
        if not np.all(np.isfinite(anchors[frame, side])):
            continue

        current = foot_positions[frame, side].copy()
        anchor = anchors[frame, side].copy()
        # Keep the connector horizontal: it represents stance-foot xy slip.
        anchor[2] = current[2]

        anchor_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            anchor_geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0.025, 0.025]),
            pos=anchor,
            mat=identity,
            rgba=colors[side],
        )
        viewer.user_scn.ngeom += 1

        slip_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            slip_geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.006, 0.006, 0.006]),
            pos=current,
            mat=identity,
            rgba=colors[side],
        )
        mujoco.mjv_connector(
            slip_geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=0.006,
            from_=anchor,
            to=current,
        )
        viewer.user_scn.ngeom += 1


def print_summary(
    clip: dict[str, Any],
    model_path: Path,
    foot_positions: np.ndarray,
) -> None:
    duration = clip["n_frames"] / clip["fps"]
    distance = float(
        np.linalg.norm(
            clip["root_translation"][-1, :2] - clip["root_translation"][0, :2]
        )
    )
    print("X1 MuJoCo playback check")
    print(f"  clip:      {clip['path']}")
    print(f"  model:     {model_path}")
    print(
        f"  motion:    {clip['n_frames']} frames @ {clip['fps']:.2f} Hz "
        f"({duration:.2f} s)"
    )
    print(f"  root path: {distance:.3f} m")
    print(
        "  foot z:    "
        f"left [{foot_positions[:, 0, 2].min():.3f}, "
        f"{foot_positions[:, 0, 2].max():.3f}] m, "
        f"right [{foot_positions[:, 1, 2].min():.3f}, "
        f"{foot_positions[:, 1, 2].max():.3f}] m"
    )
    print("  mapping:   12/12 X1 hinge joints found; all arrays finite")


def print_controls() -> None:
    print("\nViewer controls")
    print("  Space       pause/resume")
    print("  Left/Right  pause and step one frame")
    print("  Up/Down     double/halve playback speed")
    print("  R           restart from frame 0")
    print("  L           toggle looping")
    print("  C           toggle follow camera")
    print("  M           toggle contact anchors/slip lines")
    print("  H           print this help")
    print("  Q           quit\n")


def run_viewer(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: dict[str, Any],
    qpos_by_name: dict[str, int],
    foot_positions: np.ndarray,
    anchors: np.ndarray,
) -> None:
    import mujoco.viewer

    start_frame = int(np.clip(args.start_frame, 0, clip["n_frames"] - 1))
    state: dict[str, Any] = {
        "frame": start_frame,
        "paused": False,
        "loop": bool(args.loop),
        "speed": float(args.speed),
        "follow_camera": not args.no_follow_camera,
        "show_markers": not args.hide_contact_markers,
        "step": 0,
        "dirty": True,
        "quit": False,
    }

    def keyboard_callback(keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            state["dirty"] = True
            print(f"{'paused' if state['paused'] else 'playing'} at frame {state['frame']}")
        elif keycode == glfw.KEY_LEFT:
            state["paused"] = True
            state["step"] = -1
        elif keycode == glfw.KEY_RIGHT:
            state["paused"] = True
            state["step"] = 1
        elif keycode == glfw.KEY_UP:
            state["speed"] = min(8.0, state["speed"] * 2.0)
            print(f"speed: {state['speed']:.3g}x")
        elif keycode == glfw.KEY_DOWN:
            state["speed"] = max(0.125, state["speed"] * 0.5)
            print(f"speed: {state['speed']:.3g}x")
        elif keycode == glfw.KEY_R:
            state["frame"] = 0
            state["dirty"] = True
            print("restarted at frame 0")
        elif keycode == glfw.KEY_L:
            state["loop"] = not state["loop"]
            print(f"loop: {state['loop']}")
        elif keycode == glfw.KEY_C:
            state["follow_camera"] = not state["follow_camera"]
            state["dirty"] = True
            print(f"follow camera: {state['follow_camera']}")
        elif keycode == glfw.KEY_M:
            state["show_markers"] = not state["show_markers"]
            state["dirty"] = True
            print(f"contact markers: {state['show_markers']}")
        elif keycode == glfw.KEY_H:
            print_controls()
        elif keycode == glfw.KEY_Q:
            state["quit"] = True

    def advance(delta: int, *, mark_dirty: bool = True) -> None:
        candidate = state["frame"] + delta
        if state["loop"]:
            state["frame"] = candidate % clip["n_frames"]
        else:
            state["frame"] = int(np.clip(candidate, 0, clip["n_frames"] - 1))
            if candidate >= clip["n_frames"]:
                state["paused"] = True
                print("reached final frame; paused (press R or enable loop with L)")
        if mark_dirty:
            state["dirty"] = True

    print_controls()
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=keyboard_callback,
    ) as viewer:
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        next_frame_time = time.monotonic()

        while viewer.is_running() and not state["quit"]:
            if state["step"]:
                advance(state["step"])
                state["step"] = 0

            now = time.monotonic()
            if not state["paused"] and now >= next_frame_time:
                state["dirty"] = True

            if state["dirty"]:
                frame = state["frame"]
                apply_frame(model, data, clip, qpos_by_name, frame)
                if state["show_markers"]:
                    add_contact_markers(viewer, foot_positions, anchors, frame)
                else:
                    viewer.user_scn.ngeom = 0
                if state["follow_camera"]:
                    viewer.cam.lookat[:] = clip["root_translation"][frame]
                viewer.sync()
                state["dirty"] = False

                if not state["paused"]:
                    # Queue the next frame, but render it only after its deadline.
                    advance(1, mark_dirty=False)
                    frame_dt = 1.0 / (clip["fps"] * state["speed"])
                    next_frame_time += frame_dt
                    if next_frame_time < now - frame_dt:
                        next_frame_time = now + frame_dt
                continue

            # Keep input responsive while paused or waiting for the next frame.
            viewer.sync()
            if state["paused"]:
                next_frame_time = time.monotonic()
                time.sleep(0.01)
            else:
                time.sleep(max(0.0, min(0.005, next_frame_time - now)))


def main() -> int:
    args = parse_args()
    if not args.check_only:
        ensure_mjpython_on_macos()

    clip = load_clip(args.clip)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MJCF model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos_by_name = build_joint_qpos_map(model, clip["dof_names"])
    foot_positions, anchors = compute_foot_positions_and_anchors(
        model, data, clip, qpos_by_name
    )
    print_summary(clip, model_path, foot_positions)

    if args.check_only:
        return 0

    run_viewer(
        args,
        model,
        data,
        clip,
        qpos_by_name,
        foot_positions,
        anchors,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
