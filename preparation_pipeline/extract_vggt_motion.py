#!/usr/bin/env python3
"""Extract native layer-4 VGGT-Omega raw motion for LIBERO.

This matches RoboTwin ``extract_vggt.py --raw-motion-only
--motion-layer-index 4``. Three synchronized camera views are processed jointly
and each output stores ``raw_motion`` with shape ``[F,Hp,Wp]``. L2 feature
normalization and temporal differencing are computed offline on VGGT's native
patch grid (normally 32x32 at resolution 512). Spatial resizing and q50/q90
normalization are left to the downstream training pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import imageio_ffmpeg
import torch

try:
    from .extract_vggt import (
        DEFAULT_CAMERAS,
        DEFAULT_DATASET,
        DEFAULT_VGGT_ROOT,
        FEATURE_DIM,
        PATCH_TOKEN_START,
        VGGT_PATCH_SIZE,
        EpisodeWork,
        atomic_torch_save,
        discover_work,
        load_model,
        load_temporal_metadata,
        preprocess_time_batch,
        read_selected_frames,
    )
except ImportError:
    from extract_vggt import (
        DEFAULT_CAMERAS,
        DEFAULT_DATASET,
        DEFAULT_VGGT_ROOT,
        FEATURE_DIM,
        PATCH_TOKEN_START,
        VGGT_PATCH_SIZE,
        EpisodeWork,
        atomic_torch_save,
        discover_work,
        load_model,
        load_temporal_metadata,
        preprocess_time_batch,
        read_selected_frames,
    )


LOGGER = logging.getLogger("extract_vggt_motion")
DEFAULT_MOTION_LAYER_INDEX = 4
DEFAULT_OUTPUT_DIRNAME = "vggt_raw_motion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_VGGT_ROOT / "ckpts/vggt_omega_1b_512.pt",
    )
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument(
        "--motion-layer-index",
        type=int,
        default=DEFAULT_MOTION_LAYER_INDEX,
        help="Zero-based aggregator layer index (default 4 means block 5)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--time-batch-size", type=int, default=1)
    parser.add_argument("--output-dirname", default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.image_resolution <= 0 or args.image_resolution % VGGT_PATCH_SIZE:
        parser.error("--image-resolution must be positive and divisible by 16")
    if args.time_batch_size <= 0:
        parser.error("--time-batch-size must be positive")
    if not 0 <= args.motion_layer_index < 24:
        parser.error("--motion-layer-index must be between 0 and 23")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if not args.camera_keys or len(set(args.camera_keys)) != len(args.camera_keys):
        parser.error("--camera-keys must contain unique camera names")
    return args


def compute_raw_motion(feature: torch.Tensor) -> torch.Tensor:
    """Compute RoboTwin-style unnormalized motion from [F,H,W,C]."""
    feature = torch.nn.functional.normalize(feature.float(), dim=-1, eps=1e-6)
    motion = feature.new_zeros(feature.shape[:-1])
    if feature.shape[0] > 1:
        delta = (feature[1:] - feature[:-1]).norm(dim=-1)
        motion[1:] += delta
        motion[:-1] += delta
        if feature.shape[0] > 2:
            motion[1:-1] *= 0.5
    return motion


@torch.inference_mode()
def extract_motion_episode(
    model,
    work: EpisodeWork,
    cameras: Sequence[str],
    device: torch.device,
    motion_layer_index: int,
    image_resolution: int,
    time_batch_size: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    frame_ids, temporal_metadata, latent_grids = load_temporal_metadata(
        work.latent_paths, cameras
    )
    camera_frames = [
        read_selected_frames(path, frame_ids) for path in work.video_paths
    ]
    camera_chunks: list[list[torch.Tensor]] = [[] for _ in cameras]
    patch_grid: tuple[int, int] | None = None

    for start in range(0, len(frame_ids), time_batch_size):
        time_indices = list(range(start, min(start + time_batch_size, len(frame_ids))))
        images = preprocess_time_batch(
            camera_frames, time_indices, image_resolution
        ).to(device, non_blocking=True)
        amp_dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            layer_outputs, patch_token_start = model.aggregator(images)
        if patch_token_start != PATCH_TOKEN_START:
            raise RuntimeError(
                f"Expected patch_token_start={PATCH_TOKEN_START}, got "
                f"{patch_token_start}"
            )

        feature = layer_outputs[motion_layer_index]
        if feature is None:
            raise RuntimeError(
                f"VGGT aggregator layer {motion_layer_index} was not cached"
            )
        feature = feature[:, :, patch_token_start:, :]
        if feature.shape[-1] != FEATURE_DIM:
            raise RuntimeError(
                f"Layer {motion_layer_index} feature dim is {feature.shape[-1]}, "
                f"expected {FEATURE_DIM}"
            )
        patch_h = images.shape[-2] // VGGT_PATCH_SIZE
        patch_w = images.shape[-1] // VGGT_PATCH_SIZE
        patch_grid = (patch_h, patch_w)
        if feature.shape[2] != patch_h * patch_w:
            raise RuntimeError(
                f"Patch count {feature.shape[2]} != {patch_h}*{patch_w}"
            )
        # [time,camera,patch,channel] -> [time,camera,H,W,channel]
        feature = feature.reshape(
            len(time_indices),
            len(cameras),
            patch_h,
            patch_w,
            FEATURE_DIM,
        )
        for camera_index in range(len(cameras)):
            camera_chunks[camera_index].append(
                feature[:, camera_index].to(dtype=torch.bfloat16, device="cpu")
            )
        del images, feature, layer_outputs

    if patch_grid is None:
        raise RuntimeError(f"No frames extracted for {work.latent_name}")
    native_features = [torch.cat(chunks, dim=0) for chunks in camera_chunks]
    outputs = [compute_raw_motion(feature) for feature in native_features]
    metadata = {
        **temporal_metadata,
        "camera_order": list(cameras),
        "vggt_layer_numbers": [motion_layer_index + 1],
        "vggt_layer_indices": [motion_layer_index],
        "motion_layer_index": motion_layer_index,
        "motion_layer_number": motion_layer_index + 1,
        "patch_token_start": PATCH_TOKEN_START,
        "feature_dim": FEATURE_DIM,
        "vggt_patch_grid": list(patch_grid),
        "latent_grids": {
            camera: list(grid) for camera, grid in zip(cameras, latent_grids)
        },
        "image_resolution": image_resolution,
        "preprocess_mode": "balanced",
        "output_kind": "raw_motion",
        "motion_normalization": "none",
        "raw_motion_dtype": "float32",
        "motion_computed_on_native_patch_grid": True,
        "motion_resize_during_training": "area",
        "motion_normalize_during_training": "per-frame-q50-q90",
        "resampled_to_lingbot_grid": False,
    }
    return outputs, metadata


def save_motion_episode(
    work: EpisodeWork,
    cameras: Sequence[str],
    outputs: Sequence[torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    for camera, output, output_path in zip(cameras, outputs, work.output_paths):
        atomic_torch_save(
            {
                "raw_motion": output.contiguous(),
                "camera": camera,
                **metadata,
            },
            output_path,
        )


def write_metadata(args: argparse.Namespace, cameras: Sequence[str]) -> None:
    output_root = args.dataset_dir / args.output_dirname
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "teacher": "VGGT-Omega",
        "checkpoint": str(args.checkpoint.resolve()),
        "vggt_layer_numbers": [args.motion_layer_index + 1],
        "vggt_layer_indices": [args.motion_layer_index],
        "motion_layer_index": args.motion_layer_index,
        "motion_layer_number": args.motion_layer_index + 1,
        "feature_dim": FEATURE_DIM,
        "camera_order": list(cameras),
        "temporal_mapping": {
            "first_latent": "sampled RGB frame 0",
            "later_latents": "third frame of each following four-frame group",
            "sampled_video_indices": "[0, 3, 7, 11, ...]",
        },
        "output_kind": "raw_motion",
        "motion_normalization": "none",
        "raw_motion_dtype": "float32",
        "motion_computed_on_native_patch_grid": True,
        "motion_resize_during_training": "area",
        "motion_normalize_during_training": "per-frame-q50-q90",
        "resampled_to_lingbot_grid": False,
    }
    path = output_root / "metadata.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args.dataset_dir = args.dataset_dir.resolve()
    args.vggt_root = args.vggt_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    cameras = tuple(args.camera_keys)
    work_items = discover_work(
        args.dataset_dir,
        cameras,
        args.output_dirname,
        args.num_shards,
        args.shard_index,
    )
    if args.max_episodes is not None:
        work_items = work_items[: args.max_episodes]
    if not work_items:
        raise RuntimeError(
            f"No matching latent episodes for shard "
            f"{args.shard_index}/{args.num_shards}"
        )

    LOGGER.info(
        "Found %d episodes for shard %d/%d; layer index=%d; output=%s/%s",
        len(work_items),
        args.shard_index,
        args.num_shards,
        args.motion_layer_index,
        args.dataset_dir,
        args.output_dirname,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    LOGGER.info("Using imageio-ffmpeg %s", imageio_ffmpeg.get_ffmpeg_version())
    model = load_model(args.vggt_root, args.checkpoint, device)

    completed = skipped = 0
    for item_index, work in enumerate(work_items, start=1):
        if all(path.is_file() for path in work.output_paths) and not args.overwrite:
            skipped += 1
            LOGGER.info(
                "[%d/%d] skip episode %06d",
                item_index,
                len(work_items),
                work.episode_index,
            )
            continue
        LOGGER.info(
            "[%d/%d] extracting episode %06d",
            item_index,
            len(work_items),
            work.episode_index,
        )
        outputs, metadata = extract_motion_episode(
            model,
            work,
            cameras,
            device,
            args.motion_layer_index,
            args.image_resolution,
            args.time_batch_size,
        )
        save_motion_episode(work, cameras, outputs, metadata)
        completed += 1

    write_metadata(args, cameras)
    LOGGER.info("Done: extracted=%d, skipped=%d", completed, skipped)


if __name__ == "__main__":
    main()
