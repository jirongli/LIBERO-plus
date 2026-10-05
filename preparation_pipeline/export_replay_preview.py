#!/usr/bin/env python3
"""Export replayed external-camera observations as task-organized MP4 previews."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("camera_1_rgb", "camera_2_rgb")


def write_video(frames: np.ndarray, output: Path, fps: int, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [
        ffmpeg,
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    count = 0
    for hdf5_path in sorted(args.input_dir.glob("*.hdf5")):
        task_name = hdf5_path.stem.removesuffix("_demo")
        with h5py.File(hdf5_path, "r") as source:
            demos = sorted(
                source["data"], key=lambda name: int(name.rsplit("_", 1)[-1])
            )
            for demo_name in demos:
                obs = source["data"][demo_name]["obs"]
                for camera in CAMERAS:
                    frames = np.asarray(obs[camera], dtype=np.uint8)[:, ::-1]
                    output = args.output_dir / task_name / f"{demo_name}_{camera}.mp4"
                    write_video(np.ascontiguousarray(frames), output, args.fps, args.ffmpeg)
                count += 1
                print(f"exported {task_name}/{demo_name}", flush=True)
    print(f"Exported {count} demonstrations ({count * len(CAMERAS)} videos)")


if __name__ == "__main__":
    main()
