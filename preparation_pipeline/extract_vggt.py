#!/usr/bin/env python3
"""Extract multi-view VGGT-Omega features for the LIBERO LeRobot dataset.

Three synchronized camera views are passed to VGGT-Omega together at every
WAN latent time step, preserving the model's inter-view interaction. Outputs
mirror the latent tree inside the same dataset::

    processed_data/libero_10/vggt/chunk-000/<camera>/
        episode_000000_0_272.pth

Each file stores ``feature`` with shape ``[L,F,H,W,2048]``. ``F,H,W`` match
the corresponding WAN latent metadata (normally ``F,8,8`` for LIBERO), and L
is the number of selected VGGT aggregator layers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


LOGGER = logging.getLogger("extract_vggt")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "processed_data/libero_10"
DEFAULT_VGGT_ROOT = REPO_ROOT / "vggt-omega"
DEFAULT_CAMERAS = (
    "observation.images.agentview_rgb",
    "observation.images.agentview2_rgb",
    "observation.images.eye_in_hand_rgb",
)
# Zero-based aggregator indices for VGGT-Omega layers 5, 12, 18, and 24.
# Layer index 4 is required by LingBot-VA's online motion-map supervision.
DEFAULT_LAYER_INDICES = (4, 11, 17, 23)
EPISODE_RE = re.compile(r"^episode_(\d{6})_(\d+)_(\d+)\.pth$")
VGGT_PATCH_SIZE = 16
PATCH_TOKEN_START = 17
FEATURE_DIM = 2048


@dataclass(frozen=True)
class EpisodeWork:
    chunk_name: str
    latent_name: str
    episode_index: int
    latent_paths: tuple[Path, ...]
    video_paths: tuple[Path, ...]
    output_paths: tuple[Path, ...]


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
        "--layer-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYER_INDICES),
        help="Zero-based aggregator block indices",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--time-batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--output-dirname", default="vggt")
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
    if not args.layer_indices or any(i < 0 or i >= 24 for i in args.layer_indices):
        parser.error("--layer-indices must contain values between 0 and 23")
    if len(set(args.layer_indices)) != len(args.layer_indices):
        parser.error("--layer-indices must not contain duplicates")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if not args.camera_keys or len(set(args.camera_keys)) != len(args.camera_keys):
        parser.error("--camera-keys must contain unique camera names")
    return args


def load_model(vggt_root: Path, checkpoint: Path, device: torch.device):
    if not vggt_root.is_dir():
        raise FileNotFoundError(f"VGGT-Omega source not found: {vggt_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(vggt_root.resolve()))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(enable_camera=False, enable_depth=False).to(device).eval()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(("camera_head.", "dense_head.", "text_alignment_head."))
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match VGGT-Omega aggregator: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


def discover_work(
    dataset_dir: Path,
    cameras: Sequence[str],
    output_dirname: str,
    num_shards: int,
    shard_index: int,
) -> list[EpisodeWork]:
    latent_root = dataset_dir / "latents"
    video_root = dataset_dir / "videos"
    if not latent_root.is_dir():
        raise FileNotFoundError(f"Latent directory not found: {latent_root}")

    work_items: list[EpisodeWork] = []
    for chunk_dir in sorted(latent_root.glob("chunk-*")):
        reference_dir = chunk_dir / cameras[0]
        if not reference_dir.is_dir():
            continue
        for reference_path in sorted(reference_dir.glob("episode_*.pth")):
            match = EPISODE_RE.fullmatch(reference_path.name)
            if match is None:
                LOGGER.warning("Ignoring unrecognized latent file: %s", reference_path)
                continue
            episode_index = int(match.group(1))
            if episode_index % num_shards != shard_index:
                continue
            chunk_name = chunk_dir.name
            latent_paths = tuple(
                latent_root / chunk_name / camera / reference_path.name
                for camera in cameras
            )
            video_paths = tuple(
                video_root
                / chunk_name
                / camera
                / f"episode_{episode_index:06d}.mp4"
                for camera in cameras
            )
            output_paths = tuple(
                dataset_dir
                / output_dirname
                / chunk_name
                / camera
                / reference_path.name
                for camera in cameras
            )
            missing = [
                str(path) for path in (*latent_paths, *video_paths) if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Episode {episode_index:06d} has missing inputs: {missing}"
                )
            work_items.append(
                EpisodeWork(
                    chunk_name=chunk_name,
                    latent_name=reference_path.name,
                    episode_index=episode_index,
                    latent_paths=latent_paths,
                    video_paths=video_paths,
                    output_paths=output_paths,
                )
            )
    return work_items


def teacher_sample_indices(video_frames: int, latent_frames: int) -> list[int]:
    """Map WAN's 1+4k RGB samples to latent times: [0,3,7,11,...]."""
    if video_frames < 1 or (video_frames - 1) % 4:
        raise ValueError(f"video_num_frames={video_frames} is not of form 1+4k")
    expected = 1 + (video_frames - 1) // 4
    if latent_frames != expected:
        raise ValueError(
            f"latent_num_frames={latent_frames}, expected {expected} from "
            f"video_num_frames={video_frames}"
        )
    return [0] + [4 * index - 1 for index in range(1, latent_frames)]


def load_temporal_metadata(
    latent_paths: Sequence[Path], cameras: Sequence[str]
) -> tuple[list[int], dict[str, Any], list[tuple[int, int]]]:
    metadata = []
    required = (
        "latent_num_frames",
        "latent_height",
        "latent_width",
        "video_num_frames",
        "frame_ids",
    )
    for path in latent_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{path} is missing fields {missing}")
        metadata.append(
            {
                key: payload[key]
                for key in (*required, "start_frame", "end_frame")
                if key in payload
            }
        )
        del payload

    reference = metadata[0]
    reference_ids = np.asarray(reference["frame_ids"], dtype=np.int64)
    for camera, item in zip(cameras[1:], metadata[1:]):
        if int(item["latent_num_frames"]) != int(reference["latent_num_frames"]):
            raise ValueError(f"{camera}: latent_num_frames differs between cameras")
        if int(item["video_num_frames"]) != int(reference["video_num_frames"]):
            raise ValueError(f"{camera}: video_num_frames differs between cameras")
        if not np.array_equal(np.asarray(item["frame_ids"]), reference_ids):
            raise ValueError(f"{camera}: frame_ids differs between cameras")

    sample_indices = teacher_sample_indices(
        int(reference["video_num_frames"]), int(reference["latent_num_frames"])
    )
    if len(reference_ids) != int(reference["video_num_frames"]):
        raise ValueError(
            f"{latent_paths[0]} has {len(reference_ids)} frame_ids, expected "
            f"{reference['video_num_frames']}"
        )
    original_frame_ids = [int(reference_ids[index]) for index in sample_indices]
    compact = {
        "latent_num_frames": int(reference["latent_num_frames"]),
        "video_num_frames": int(reference["video_num_frames"]),
        "sampled_video_indices": sample_indices,
        "original_frame_ids": original_frame_ids,
        "start_frame": int(reference.get("start_frame", 0)),
        "end_frame": int(reference.get("end_frame", reference_ids[-1] + 1)),
    }
    latent_grids = [
        (int(item["latent_height"]), int(item["latent_width"]))
        for item in metadata
    ]
    return original_frame_ids, compact, latent_grids


def read_selected_frames(video_path: Path, frame_ids: Sequence[int]) -> list[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    video_metadata = next(reader)
    width, height = video_metadata["size"]
    requested = set(frame_ids)
    output: dict[int, np.ndarray] = {}
    try:
        for frame_index, frame_bytes in enumerate(reader):
            if frame_index in requested:
                output[frame_index] = np.frombuffer(
                    frame_bytes, dtype=np.uint8
                ).reshape(height, width, 3).copy()
                if len(output) == len(requested):
                    break
    finally:
        reader.close()
    missing = requested.difference(output)
    if missing:
        raise ValueError(f"{video_path}: missing frames {sorted(missing)[:10]}")
    return [output[index] for index in frame_ids]


def balanced_target_shape(
    height: int, width: int, image_resolution: int
) -> tuple[int, int]:
    aspect_ratio = height / max(width, 1)
    token_count = (image_resolution // VGGT_PATCH_SIZE) ** 2
    width_patches = max(1, round(np.sqrt(token_count / aspect_ratio)))
    height_patches = max(1, round(token_count / width_patches))
    return height_patches * VGGT_PATCH_SIZE, width_patches * VGGT_PATCH_SIZE


def preprocess_time_batch(
    camera_frames: Sequence[Sequence[np.ndarray]],
    time_indices: Sequence[int],
    image_resolution: int,
) -> torch.Tensor:
    batches = []
    produced_shapes: set[tuple[int, int]] = set()
    for time_index in time_indices:
        views = []
        for frames in camera_frames:
            rgb = frames[time_index]
            target_h, target_w = balanced_target_shape(
                rgb.shape[0], rgb.shape[1], image_resolution
            )
            image = Image.fromarray(rgb).resize(
                (target_w, target_h), Image.Resampling.BICUBIC
            )
            tensor = pil_to_tensor(image).float().div_(255.0)
            produced_shapes.add(tuple(tensor.shape[-2:]))
            views.append(tensor)
        batches.append(torch.stack(views))
    if len(produced_shapes) != 1:
        raise ValueError(
            f"Camera preprocessing produced unequal shapes: {sorted(produced_shapes)}"
        )
    return torch.stack(batches)


def resize_feature_grid(
    feature: torch.Tensor, output_hw: tuple[int, int]
) -> torch.Tensor:
    """Resize [L,T,H,W,C] to the corresponding WAN latent grid."""
    if tuple(feature.shape[-3:-1]) == output_hw:
        return feature
    layers, frames, height, width, channels = feature.shape
    values = feature.reshape(-1, height, width, channels)
    values = F.interpolate(
        values.permute(0, 3, 1, 2).float(),
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return values.reshape(layers, frames, *output_hw, channels)


@torch.inference_mode()
def extract_episode(
    model,
    work: EpisodeWork,
    cameras: Sequence[str],
    device: torch.device,
    layer_indices: Sequence[int],
    image_resolution: int,
    time_batch_size: int,
    output_dtype: torch.dtype,
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

        selected = []
        for layer_index in layer_indices:
            layer = layer_outputs[layer_index]
            if layer is None:
                raise RuntimeError(f"Aggregator layer {layer_index} was not cached")
            layer = layer[:, :, patch_token_start:, :]
            if layer.shape[-1] != FEATURE_DIM:
                raise RuntimeError(
                    f"Layer {layer_index} feature dim is {layer.shape[-1]}, "
                    f"expected {FEATURE_DIM}"
                )
            selected.append(layer)
        feature = torch.stack(selected)
        patch_h = images.shape[-2] // VGGT_PATCH_SIZE
        patch_w = images.shape[-1] // VGGT_PATCH_SIZE
        patch_grid = (patch_h, patch_w)
        if feature.shape[3] != patch_h * patch_w:
            raise RuntimeError(
                f"Patch count {feature.shape[3]} != {patch_h}*{patch_w}"
            )
        feature = feature.reshape(
            len(layer_indices),
            len(time_indices),
            len(cameras),
            patch_h,
            patch_w,
            FEATURE_DIM,
        )
        for camera_index, latent_grid in enumerate(latent_grids):
            camera_feature = resize_feature_grid(
                feature[:, :, camera_index], latent_grid
            )
            camera_chunks[camera_index].append(
                camera_feature.to(dtype=output_dtype, device="cpu")
            )
        del images, feature, layer_outputs

    if patch_grid is None:
        raise RuntimeError(f"No frames extracted for {work.latent_name}")
    outputs = [torch.cat(chunks, dim=1) for chunks in camera_chunks]
    metadata = {
        **temporal_metadata,
        "camera_order": list(cameras),
        "vggt_layer_numbers": [index + 1 for index in layer_indices],
        "vggt_layer_indices": list(layer_indices),
        "patch_token_start": PATCH_TOKEN_START,
        "feature_dim": FEATURE_DIM,
        "vggt_patch_grid": list(patch_grid),
        "latent_grids": {
            camera: list(grid) for camera, grid in zip(cameras, latent_grids)
        },
        "image_resolution": image_resolution,
        "preprocess_mode": "balanced",
        "resampled_to_lingbot_grid": True,
        "output_kind": "feature",
    }
    return outputs, metadata


def atomic_torch_save(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def save_episode(
    work: EpisodeWork,
    cameras: Sequence[str],
    outputs: Sequence[torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    for camera, output, output_path in zip(cameras, outputs, work.output_paths):
        atomic_torch_save(
            {"feature": output.contiguous(), "camera": camera, **metadata},
            output_path,
        )


def write_metadata(args: argparse.Namespace, cameras: Sequence[str]) -> None:
    output_root = args.dataset_dir / args.output_dirname
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "teacher": "VGGT-Omega",
        "checkpoint": str(args.checkpoint.resolve()),
        "vggt_layer_numbers": [index + 1 for index in args.layer_indices],
        "vggt_layer_indices": list(args.layer_indices),
        "feature_dim": FEATURE_DIM,
        "camera_order": list(cameras),
        "temporal_mapping": {
            "first_latent": "sampled RGB frame 0",
            "later_latents": "third frame of each following four-frame group",
            "sampled_video_indices": "[0, 3, 7, 11, ...]",
        },
        "spatial_mapping": "each camera is resized to its WAN latent grid",
        "output_kind": "feature",
    }
    path = output_root / "metadata.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
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
        "Found %d episodes for shard %d/%d; output=%s/%s",
        len(work_items),
        args.shard_index,
        args.num_shards,
        args.dataset_dir,
        args.output_dirname,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    output_dtype = getattr(torch, args.output_dtype)
    LOGGER.info("Using imageio-ffmpeg %s", imageio_ffmpeg.get_ffmpeg_version())
    model = load_model(args.vggt_root, args.checkpoint, device)

    completed = skipped = 0
    for item_index, work in enumerate(work_items, start=1):
        if all(path.is_file() for path in work.output_paths) and not args.overwrite:
            skipped += 1
            LOGGER.info(
                "[%d/%d] skip episode %06d", item_index, len(work_items),
                work.episode_index,
            )
            continue
        LOGGER.info(
            "[%d/%d] extracting episode %06d",
            item_index,
            len(work_items),
            work.episode_index,
        )
        outputs, metadata = extract_episode(
            model,
            work,
            cameras,
            device,
            args.layer_indices,
            args.image_resolution,
            args.time_batch_size,
            output_dtype,
        )
        save_episode(work, cameras, outputs, metadata)
        completed += 1

    write_metadata(args, cameras)
    LOGGER.info("Done: extracted=%d, skipped=%d", completed, skipped)


if __name__ == "__main__":
    main()
