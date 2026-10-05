#!/usr/bin/env python3
"""Convert replayed LIBERO-10 HDF5 files to LingBot-VA LeRobot v2.1.

The output layout follows ``robbyant/libero-long-lerobot``::

    processed_data/libero_10/
      data/chunk-000/episode_XXXXXX.parquet
      videos/chunk-000/observation.images.agentview_rgb/episode_XXXXXX.mp4
      videos/chunk-000/observation.images.agentview2_rgb/episode_XXXXXX.mp4
      videos/chunk-000/observation.images.eye_in_hand_rgb/episode_XXXXXX.mp4
      meta/info.json
      meta/episodes.jsonl
      meta/episodes_stats.jsonl
      meta/tasks.jsonl

The replayed static cameras and LIBERO wrist camera are exported as:

    camera_1_rgb -> observation.images.agentview_rgb
    camera_2_rgb -> observation.images.agentview2_rgb
    eye_in_hand_rgb -> observation.images.eye_in_hand_rgb
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.agentview2_rgb",
    "observation.images.eye_in_hand_rgb",
)
CAMERA_MAP = {
    "camera_1_rgb": VIDEO_KEYS[0],
    "camera_2_rgb": VIDEO_KEYS[1],
    "eye_in_hand_rgb": VIDEO_KEYS[2],
}
STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def json_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sorted_demos(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))


def read_task(data_group: h5py.Group) -> str:
    problem_info = json.loads(json_text(data_group.attrs["problem_info"]))
    instruction = problem_info.get("language_instruction")
    if isinstance(instruction, list):
        instruction = instruction[0]
    if not instruction:
        raise ValueError("HDF5 problem_info has no language_instruction")
    return str(instruction)


def read_state_and_action(demo: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    obs = demo["obs"]
    ee = np.asarray(obs["ee_states"], dtype=np.float32)
    gripper = np.asarray(obs["gripper_states"], dtype=np.float32)
    action = np.asarray(demo["actions"], dtype=np.float32)
    if ee.ndim != 2 or ee.shape[1] != 6:
        raise ValueError(f"expected ee_states shape (T, 6), got {ee.shape}")
    if gripper.ndim != 2 or gripper.shape[1] != 2:
        raise ValueError(f"expected gripper_states shape (T, 2), got {gripper.shape}")
    state = np.concatenate((ee, gripper), axis=1).astype(np.float32, copy=False)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"expected actions shape (T, 7), got {action.shape}")
    if not (len(state) == len(action)):
        raise ValueError(f"state/action length mismatch: {len(state)} != {len(action)}")
    return state, action


def noop_keep_mask(action: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Return a mask that drops no-op actions, following OpenVLA's LIBERO rule.

    A step is a no-op when its 6-DoF arm delta is ~zero and its gripper command
    equals the previous step's command.  The replayed videos come from recorded
    MuJoCo states, so frames can be dropped without re-simulating.
    """
    still = np.linalg.norm(action[:, :-1], axis=1) < threshold
    same_gripper = np.ones(len(action), dtype=bool)
    same_gripper[1:] = action[1:, -1] == action[:-1, -1]
    return ~(still & same_gripper)


def read_frames(demo: h5py.Group, dataset_name: str, flip_vertical: bool) -> np.ndarray:
    obs = demo["obs"]
    if dataset_name not in obs:
        raise KeyError(
            f"{demo.name}/obs/{dataset_name} is missing; run replay_camera.py first "
            "and make sure it preserves the original eye_in_hand_rgb dataset"
        )
    frames = np.asarray(obs[dataset_name], dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected {dataset_name} shape (T,H,W,3), got {frames.shape}")
    # LIBERO HDF5 records raw OpenGL images. The reference dataset and the
    # LingBot-VA evaluation client use vertically corrected RGB images.
    if flip_vertical:
        frames = frames[:, ::-1]
    return np.ascontiguousarray(frames)


def write_video(
    frames: np.ndarray,
    output: Path,
    fps: int,
    crf: int,
    ffmpeg: str,
    encoder: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if encoder == "libx264":
        command.extend(["-crf", str(crf), "-preset", "medium"])
    else:
        # OpenH264 does not implement x264's CRF / preset options.
        command.extend(["-b:v", "2M"])
    command.extend(["-movflags", "+faststart", str(output)])
    result = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {output}: {result.stderr.decode(errors='replace')}"
        )


def select_h264_encoder(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Could not query ffmpeg encoders:\n{result.stdout}")
    for encoder in ("libx264", "libopenh264"):
        if encoder in result.stdout:
            return encoder
    raise RuntimeError("ffmpeg has neither libx264 nor libopenh264 H.264 encoder")


def write_parquet(
    output: Path,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    task_index: int,
    global_index: int,
    fps: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    length = len(action)
    table = pa.table(
        {
            "observation.state": pa.array(
                state.tolist(), type=pa.list_(pa.float32(), 8)
            ),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
            "timestamp": pa.array(
                (np.arange(length, dtype=np.float32) / fps), type=pa.float32()
            ),
            "frame_index": pa.array(np.arange(length), type=pa.int64()),
            "episode_index": pa.array(
                np.full(length, episode_index), type=pa.int64()
            ),
            "index": pa.array(
                np.arange(global_index, global_index + length), type=pa.int64()
            ),
            "task_index": pa.array(np.full(length, task_index), type=pa.int64()),
        }
    )
    pq.write_table(table, output)


def vector_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
    }


def scalar_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def image_stats(frames: np.ndarray, stride: int) -> dict[str, list[Any]]:
    sampled = frames[::stride].astype(np.float32) / 255.0

    def channel_shape(values: np.ndarray) -> list[list[list[float]]]:
        return [[[float(value)]] for value in values]

    return {
        "min": channel_shape(sampled.min(axis=(0, 1, 2))),
        "max": channel_shape(sampled.max(axis=(0, 1, 2))),
        "mean": channel_shape(sampled.mean(axis=(0, 1, 2))),
        "std": channel_shape(sampled.std(axis=(0, 1, 2))),
        "count": [int(len(sampled))],
    }


def build_info(
    episodes: int,
    frames: int,
    tasks: int,
    image_shapes: dict[str, tuple[int, int, int]],
    fps: int,
    chunk_size: int,
) -> dict[str, Any]:
    # One chunk per input directory; with a single input every episode lands
    # in chunk-000.
    chunk_size = max(chunk_size, 1)
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": {"motors": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": {"motors": ACTION_NAMES},
        },
    }
    for video_key in VIDEO_KEYS:
        height, width, channels = image_shapes[video_key]
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": channels,
                "has_audio": False,
            },
        }
    for name, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    return {
        "codebase_version": "v2.1",
        "robot_type": "Franka",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "total_videos": episodes * len(VIDEO_KEYS),
        "total_chunks": -(-episodes // chunk_size),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data/libero_10_replay")
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Several replay directories; the i-th one is written to chunk-00i "
        "(overrides --input-dir)",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "processed_data/libero_10")
    parser.add_argument("--fps", type=int, default=60, help="Reference dataset uses 60 FPS")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG_BINARY", "ffmpeg"),
        help="ffmpeg executable (or set FFMPEG_BINARY)",
    )
    parser.add_argument("--stats-stride", type=int, default=10)
    parser.add_argument("--no-flip-vertical", action="store_true")
    parser.add_argument(
        "--limit-episodes", type=int, default=None, help="Limit per input directory"
    )
    parser.add_argument(
        "--keep-noops",
        action="store_true",
        help="Keep no-op actions (they are filtered by default, as in OpenVLA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in (args.input_dirs or [args.input_dir])]
    output_dir = args.output_dir.resolve()
    if any(output_dir.iterdir()) if output_dir.exists() else False:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new/empty directory."
        )
    if args.fps <= 0 or args.stats_stride <= 0:
        raise ValueError("--fps and --stats-stride must be positive")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {args.ffmpeg!r}; pass --ffmpeg /path/to/ffmpeg"
        )
    h264_encoder = select_h264_encoder(ffmpeg)
    print(f"Using ffmpeg H.264 encoder: {h264_encoder}", flush=True)

    # Downstream code locates files with episode_index // chunks_size, so
    # "one chunk per input directory" needs every directory (except the last)
    # to contribute exactly chunks_size episodes.
    episode_sources: list[tuple[Path, str, str]] = []
    chunk_size = 0
    for chunk, input_dir in enumerate(input_dirs):
        input_files = sorted(input_dir.glob("*.hdf5"))
        if not input_files:
            raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
        dir_sources: list[tuple[Path, str, str]] = []
        for hdf5_path in input_files:
            with h5py.File(hdf5_path, "r") as source:
                task = read_task(source["data"])
                for demo_name in sorted_demos(source["data"]):
                    dir_sources.append((hdf5_path, demo_name, task))
        if args.limit_episodes is not None:
            dir_sources = dir_sources[: args.limit_episodes]
        if not dir_sources:
            raise ValueError(f"No episodes selected in {input_dir}")
        if chunk == 0:
            chunk_size = len(dir_sources)
        elif len(episode_sources) != chunk * chunk_size or len(dir_sources) > chunk_size:
            raise ValueError(
                f"{input_dir} has {len(dir_sources)} episodes but chunk size is "
                f"{chunk_size} (set by {input_dirs[0]}); every input directory "
                "except the last must have exactly that many episodes"
            )
        print(
            f"chunk-{chunk:03d}: {input_dir} ({len(dir_sources)} episodes, "
            f"{len(input_files)} files)",
            flush=True,
        )
        episode_sources.extend(dir_sources)
    task_names = list(dict.fromkeys(task for _, _, task in episode_sources))

    tasks_to_index = {task: index for index, task in enumerate(task_names)}
    episodes_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    image_shapes: dict[str, tuple[int, int, int]] = {}
    global_index = 0
    total_noops = 0

    for episode_index, (hdf5_path, demo_name, task) in enumerate(episode_sources):
        chunk = episode_index // chunk_size
        with h5py.File(hdf5_path, "r") as source:
            demo = source["data"][demo_name]
            state, action = read_state_and_action(demo)
            source_length = len(action)
            keep = (
                np.ones(source_length, dtype=bool)
                if args.keep_noops
                else noop_keep_mask(action)
            )
            if not keep.any():
                raise ValueError(f"{hdf5_path.name}/{demo_name}: every action is a no-op")
            state, action = state[keep], action[keep]
            length = len(action)
            total_noops += source_length - length
            task_index = tasks_to_index[task]
            write_parquet(
                output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
                state,
                action,
                episode_index,
                task_index,
                global_index,
                args.fps,
            )

            frame_index = np.arange(length, dtype=np.int64)
            episode_stats = {
                "observation.state": vector_stats(state),
                "action": vector_stats(action),
                "timestamp": scalar_stats(frame_index.astype(np.float32) / args.fps),
                "frame_index": scalar_stats(frame_index),
                "episode_index": scalar_stats(np.full(length, episode_index)),
                "index": scalar_stats(np.arange(global_index, global_index + length)),
                "task_index": scalar_stats(np.full(length, task_index)),
            }
            for dataset_name, video_key in CAMERA_MAP.items():
                frames = read_frames(demo, dataset_name, not args.no_flip_vertical)
                if len(frames) != source_length:
                    raise ValueError(
                        f"{hdf5_path.name}/{demo_name}: {dataset_name} has "
                        f"{len(frames)} frames, expected {source_length}"
                    )
                frames = frames[keep]
                shape = tuple(int(value) for value in frames.shape[1:])
                if video_key in image_shapes and image_shapes[video_key] != shape:
                    raise ValueError(
                        f"inconsistent shape for {video_key}: {shape} != {image_shapes[video_key]}"
                    )
                image_shapes[video_key] = shape
                write_video(
                    frames,
                    output_dir / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
                    args.fps,
                    args.video_crf,
                    ffmpeg,
                    h264_encoder,
                )
                episode_stats[video_key] = image_stats(frames, args.stats_stride)

        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": length,
                        "action_text": task,
                        "skill": "",
                    }
                ],
            }
        )
        stats_rows.append({"episode_index": episode_index, "stats": episode_stats})
        global_index += length
        print(
            f"[{episode_index + 1}/{len(episode_sources)}] "
            f"{hdf5_path.name}/{demo_name}: {length} frames "
            f"({source_length - length} no-ops removed)",
            flush=True,
        )

    meta_dir = output_dir / "meta"
    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows)
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": index, "task": task}
            for task, index in tasks_to_index.items()
        ],
    )
    info = build_info(
        len(episodes_rows),
        global_index,
        len(tasks_to_index),
        image_shapes,
        args.fps,
        chunk_size,
    )
    with (meta_dir / "info.json").open("w", encoding="utf-8") as output:
        json.dump(info, output, ensure_ascii=False, indent=4)
        output.write("\n")
    print(
        f"Converted {len(episodes_rows)} episodes ({global_index} frames, "
        f"{total_noops} no-ops removed) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
