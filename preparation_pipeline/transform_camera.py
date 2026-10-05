#!/usr/bin/env python3
"""Convert replayed LIBERO static-camera poses to ReRoPE calibrations.

The replay files contain two fixed cameras per demonstration.  Their RGB
observations are stored as ``camera_1_rgb`` and ``camera_2_rgb`` and their
sampled MuJoCo poses are stored in the ``camera_pose_info`` demo attribute.
This script follows exactly the episode ordering used by
``franka_to_lerobot.py`` and writes one calibration file per processed
episode::

    processed_data/libero_10/camera_params/episode_000000.pth

Each output contains OpenCV intrinsics and world-to-camera extrinsics for the
two fixed cameras.  The moving eye-in-hand camera is intentionally omitted:
the LingBot-VA ReRoPE implementation treats its token region as identity.

No replay or image decoding is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("camera_1", "camera_2")
DEFAULT_FOVY_DEGREES = 45.0  # MuJoCo's default camera vertical field of view.


def as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sorted_demos(data_group: h5py.Group) -> list[str]:
    """Match the ordering used by franka_to_lerobot.py exactly."""
    return sorted(data_group.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))


def parse_vector(text: str, length: int, description: str) -> np.ndarray:
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {description}: {text!r}")
    return values


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the local-to-world rotation for a MuJoCo wxyz quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"Invalid wxyz quaternion: {quaternion}")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Camera quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def opencv_intrinsic(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    """Build K for the vertically corrected RGB images in the LeRobot data."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {width}x{height}")
    if not 0.0 < fovy_degrees < 180.0:
        raise ValueError(f"Camera fovy must be in (0, 180), got {fovy_degrees}")
    focal = 0.5 * height / math.tan(math.radians(fovy_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def opencv_world_to_camera(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo camera pose to an OpenCV world-to-camera matrix.

    A MuJoCo camera looks along local -Z with local +Y pointing up.  OpenCV
    looks along +Z with +Y pointing down, hence diag(1, -1, -1).
    """
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"Invalid camera position: {position}")
    rotation_camera_to_world = quaternion_wxyz_to_matrix(quaternion)
    rotation_world_to_gl_camera = rotation_camera_to_world.T
    gl_to_cv = np.diag([1.0, -1.0, -1.0])
    rotation = gl_to_cv @ rotation_world_to_gl_camera
    translation = -rotation @ position
    extrinsic = np.concatenate((rotation, translation[:, None]), axis=1)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("Converted camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("Converted camera rotation does not have determinant +1")
    return extrinsic


def camera_xml_metadata(model_xml: str) -> dict[str, dict[str, Any]]:
    """Read the replay camera pose/FOV stored in a demo's complete MJCF."""
    root = ET.fromstring(model_xml)
    cameras = {camera.get("name"): camera for camera in root.iter("camera")}
    result: dict[str, dict[str, Any]] = {}
    for name in CAMERA_NAMES:
        camera = cameras.get(name)
        if camera is None:
            raise KeyError(f"model_file has no camera named {name!r}")
        if camera.get("mode", "fixed") != "fixed":
            raise ValueError(f"{name} is not a fixed camera")
        if camera.get("pos") is None or camera.get("quat") is None:
            raise ValueError(f"{name} does not have explicit pos and quat")
        result[name] = {
            "position": parse_vector(camera.get("pos", ""), 3, f"{name} position"),
            "quaternion": parse_vector(camera.get("quat", ""), 4, f"{name} quaternion"),
            "fovy": float(camera.get("fovy", DEFAULT_FOVY_DEGREES)),
        }
    return result


def replay_image_size(data_group: h5py.Group) -> tuple[int, int]:
    config_raw = data_group.attrs.get("replay_camera_config")
    if config_raw is None:
        raise KeyError("HDF5 /data has no replay_camera_config attribute")
    config = json.loads(as_text(config_raw))
    return int(config["width"]), int(config["height"])


def calibration_for_demo(
    demo: h5py.Group, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, Any]]:
    if "camera_pose_info" not in demo.attrs:
        raise KeyError(f"{demo.name} has no camera_pose_info attribute")
    if "model_file" not in demo.attrs:
        raise KeyError(f"{demo.name} has no model_file attribute")

    pose_info = json.loads(as_text(demo.attrs["camera_pose_info"]))
    xml_info = camera_xml_metadata(as_text(demo.attrs["model_file"]))
    intrinsics = []
    extrinsics = []
    fovys = []
    compact_poses: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        if name not in pose_info:
            raise KeyError(f"{demo.name} camera_pose_info has no {name!r}")
        position = np.asarray(pose_info[name]["position"], dtype=np.float64)
        quaternion = np.asarray(pose_info[name]["quaternion"], dtype=np.float64)
        # The two independently stored representations should describe the
        # exact same sampled camera.  This catches corrupt or mismatched files.
        if not np.allclose(position, xml_info[name]["position"], atol=1e-7):
            raise ValueError(f"{demo.name}/{name}: pose-info/XML position mismatch")
        q_xml = xml_info[name]["quaternion"]
        q_normalized = quaternion / np.linalg.norm(quaternion)
        q_xml_normalized = q_xml / np.linalg.norm(q_xml)
        if not (
            np.allclose(q_normalized, q_xml_normalized, atol=1e-7)
            or np.allclose(q_normalized, -q_xml_normalized, atol=1e-7)
        ):
            raise ValueError(f"{demo.name}/{name}: pose-info/XML quaternion mismatch")

        fovy = float(xml_info[name]["fovy"])
        intrinsics.append(opencv_intrinsic(width, height, fovy))
        extrinsics.append(opencv_world_to_camera(position, quaternion))
        fovys.append(fovy)
        compact_poses[name] = {
            "position": position.tolist(),
            "quaternion_wxyz": q_normalized.tolist(),
        }

    return (
        np.stack(intrinsics).astype(np.float32),
        np.stack(extrinsics).astype(np.float32),
        fovys,
        compact_poses,
    )


def atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data/libero_10_replay",
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Several replay directories in the same order as passed to "
        "franka_to_lerobot.py --input-dirs (overrides --input-dir)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPO_ROOT / "processed_data/libero_10",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <processed-dir>/camera_params",
    )
    parser.add_argument(
        "--limit-episodes", type=int, default=None, help="Limit per input directory"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in (args.input_dirs or [args.input_dir])]
    processed_dir = args.processed_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else processed_dir / "camera_params"
    )
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        raise ValueError("--limit-episodes must be positive")

    # Episode indices follow franka_to_lerobot.py: input directories are
    # concatenated in order, each limited to --limit-episodes.
    episode_sources: list[tuple[Path, str, int, int]] = []
    for input_dir in input_dirs:
        input_files = sorted(input_dir.glob("*.hdf5"))
        if not input_files:
            raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
        dir_sources: list[tuple[Path, str, int, int]] = []
        for hdf5_path in input_files:
            with h5py.File(hdf5_path, "r") as source:
                width, height = replay_image_size(source["data"])
                for demo_name in sorted_demos(source["data"]):
                    dir_sources.append((hdf5_path, demo_name, width, height))
        if args.limit_episodes is not None:
            dir_sources = dir_sources[: args.limit_episodes]
        episode_sources.extend(dir_sources)

    expected_total = None
    chunk_size = None
    info_path = processed_dir / "meta/info.json"
    if info_path.is_file():
        processed_info = json.loads(info_path.read_text())
        expected_total = int(processed_info["total_episodes"])
        chunk_size = int(processed_info["chunks_size"])
        if chunk_size < 1:
            raise ValueError(f"Invalid chunks_size in {info_path}: {chunk_size}")
        if args.limit_episodes is None and len(episode_sources) != expected_total:
            raise ValueError(
                "Replay/processed episode-count mismatch: "
                f"{len(episode_sources)} != {expected_total}"
            )

    written = 0
    skipped = 0
    if chunk_size is None:
        raise FileNotFoundError(
            f"Processed dataset metadata is missing: {info_path}"
        )
    for episode_index, (hdf5_path, demo_name, width, height) in enumerate(
        episode_sources
    ):
        output_path = output_dir / f"episode_{episode_index:06d}.pth"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        parquet_path = (
            processed_dir
            / "data"
            / f"chunk-{episode_index // chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Processed episode is missing for source mapping: {parquet_path}"
            )

        with h5py.File(hdf5_path, "r") as source:
            demo = source["data"][demo_name]
            intrinsics, extrinsics, fovys, poses = calibration_for_demo(
                demo, width, height
            )
        payload = {
            "format_version": 1,
            "episode_index": episode_index,
            "camera_names": list(CAMERA_NAMES),
            "camera_intrinsics": torch.from_numpy(intrinsics),
            "camera_extrinsics": torch.from_numpy(extrinsics),
            "image_width": width,
            "image_height": height,
            "fovy_degrees": fovys,
            "extrinsic_convention": "world_to_camera_opencv",
            "image_convention": "opencv_top_left_origin",
            "source_hdf5": str(hdf5_path),
            "source_demo": demo_name,
            "camera_poses_mujoco": poses,
        }
        atomic_torch_save(payload, output_path)
        written += 1
        print(
            f"[{episode_index + 1}/{len(episode_sources)}] "
            f"{hdf5_path.name}/{demo_name} -> {output_path.name}",
            flush=True,
        )

    print(
        f"Camera calibration conversion complete: written={written}, "
        f"skipped={skipped}, output={output_dir}, "
        f"processed_total={expected_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
