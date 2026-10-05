#!/usr/bin/env python3
"""Extract WAN VAE video latents and UMT5 embeddings from a LeRobot dataset.

The generated layout and payload match ``robbyant/libero-long-lerobot``::

    processed_data/libero_10/
      latents/chunk-000/<video_key>/episode_000000_0_272.pth
      empty_emb.pt

LIBERO defaults are taken from LingBot-VA: 128x128, 60 FPS and temporal
stride 1. Each action segment is truncated to a frame count of ``1 + 4k`` as
required by WAN's causal VAE.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "processed_data/libero_10"
DEFAULT_WAN_PATH = Path("/gs/bs/tga-RLA/ljirong/lingbot-va/checkpoint_base")
DEFAULT_CAMERA_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.agentview2_rgb",
    "observation.images.eye_in_hand_rgb",
)
TEXT_LENGTH = 512


def largest_1_plus_4k(count: int) -> int:
    """Return the largest positive value <= count that has form 1 + 4k."""
    return 0 if count < 1 else 1 + 4 * ((count - 1) // 4)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_video_frames(video_path: Path, frame_ids: np.ndarray) -> np.ndarray:
    """Decode selected episode-local frame indices as contiguous RGB uint8."""
    import cv2

    wanted = {int(index) for index in frame_ids}
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(decoded) == len(wanted):
                break
            index += 1
    finally:
        capture.release()

    missing = wanted.difference(decoded)
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(f"{video_path}: missing requested frames {preview}")
    return np.ascontiguousarray(
        np.stack([decoded[int(index)] for index in frame_ids])
    )


def atomic_torch_save(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


class WanEncoders:
    def __init__(
        self,
        wan_path: Path,
        device: str,
        text_device: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from diffusers import AutoencoderKLWan
        from transformers import T5TokenizerFast, UMT5EncoderModel

        self.vae_class = AutoencoderKLWan
        self.wan_path = wan_path
        self.device = torch.device(device)
        self.text_device = torch.device(text_device)
        self.dtype = dtype

        print(
            f"Loading UMT5 tokenizer/text encoder from {wan_path} "
            f"on {self.text_device} ...",
            flush=True,
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(str(wan_path / "tokenizer"))
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            str(wan_path / "text_encoder"), torch_dtype=dtype
        ).to(self.text_device).eval()
        self.text_cache: dict[str, torch.Tensor] = {}

    def load_vae(self) -> None:
        print(f"Loading WAN VAE from {self.wan_path / 'vae'} ...", flush=True)
        self.vae = self.vae_class.from_pretrained(
            str(self.wan_path / "vae"), torch_dtype=self.dtype
        ).to(self.device).eval()
        self.latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.device,
        ).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=self.device,
        ).view(1, -1, 1, 1, 1)

    @torch.inference_mode()
    def encode_video(
        self, frames: np.ndarray, height: int, width: int
    ) -> tuple[torch.Tensor, int, int, int]:
        # [T,H,W,C] -> [1,C,T,H,W], matching wan_va_server preprocessing.
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if tuple(video.shape[-2:]) != (height, width):
            video = F.interpolate(
                video,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        video = (video / 255.0 * 2.0 - 1.0).unsqueeze(0)
        video = video.to(device=self.device, dtype=self.dtype)

        posterior_mean = self.vae.encode(video).latent_dist.mean
        normalized = (posterior_mean - self.latents_mean) / self.latents_std
        _, channels, latent_frames, latent_height, latent_width = normalized.shape
        latent = normalized[0].permute(1, 2, 3, 0).reshape(-1, channels)
        return (
            latent.to(dtype=torch.bfloat16, device="cpu"),
            int(latent_frames),
            int(latent_height),
            int(latent_width),
        )

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        cached = self.text_cache.get(text)
        if cached is not None:
            return cached

        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=TEXT_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.text_device)
        attention_mask = tokens.attention_mask.to(self.text_device)
        embedding = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[0]
        sequence_length = int(attention_mask[0].sum().item())
        embedding[sequence_length:] = 0
        embedding = embedding.to(dtype=torch.bfloat16, device="cpu")
        self.text_cache[text] = embedding
        return embedding

    def release_text_encoder(self) -> None:
        """Release UMT5 after all dataset prompts have been cached on CPU."""
        del self.text_encoder
        if self.text_device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--wan-path", type=Path, default=DEFAULT_WAN_PATH)
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal sampling stride; reference LIBERO latents use 1",
    )
    parser.add_argument(
        "--ori-fps",
        type=int,
        default=None,
        help="Original FPS; default reads meta/info.json (normally 60)",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text-device",
        default="cpu",
        help="UMT5 device; use cuda for fast one-time prompt encoding",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, info: dict[str, Any]) -> None:
    if args.stride <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("--stride, --height and --width must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    for subdirectory in ("vae", "tokenizer", "text_encoder"):
        if not (args.wan_path / subdirectory).is_dir():
            raise FileNotFoundError(
                f"WAN component is missing: {args.wan_path / subdirectory}"
            )
    features = info.get("features", {})
    for camera_key in args.camera_keys:
        if camera_key not in features:
            raise KeyError(f"Camera key is absent from meta/info.json: {camera_key}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    wan_path = args.wan_path.resolve()
    info_path = dataset_dir / "meta/info.json"
    episodes_path = dataset_dir / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"{dataset_dir} is not a LeRobot dataset with meta/info.json and "
            "meta/episodes.jsonl"
        )

    info = load_json(info_path)
    args.wan_path = wan_path
    validate_args(args, info)
    original_fps = args.ori_fps or int(info["fps"])
    chunk_size = args.chunk_size or int(info.get("chunks_size", 1000))
    stored_fps: int | float = original_fps / args.stride
    if float(stored_fps).is_integer():
        stored_fps = int(stored_fps)

    all_episodes = load_jsonl(episodes_path)
    episodes = [
        episode
        for episode in all_episodes
        if int(episode["episode_index"]) % args.num_shards == args.shard_index
    ]
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    if not episodes:
        print(f"Shard {args.shard_index}/{args.num_shards} has no episodes.")
        return

    encoders = WanEncoders(wan_path, args.device, args.text_device)
    # Encode every prompt once, then release the large UMT5 model before video
    # encoding. This avoids slow repeated CPU inference and leaves GPU memory
    # available to the WAN VAE.
    shard_texts = [""]
    shard_texts.extend(
        str(segment["action_text"])
        for episode in episodes
        for segment in episode["action_config"]
    )
    for text in dict.fromkeys(shard_texts):
        encoders.encode_text(text)
    empty_embedding = encoders.text_cache[""]
    empty_path = dataset_dir / "empty_emb.pt"
    if args.overwrite or not empty_path.exists():
        atomic_torch_save(empty_embedding, empty_path)
    encoders.release_text_encoder()
    encoders.load_vae()

    print(
        f"Extracting {len(episodes)}/{len(all_episodes)} episodes; "
        f"shard={args.shard_index}/{args.num_shards}, cameras={args.camera_keys}",
        flush=True,
    )
    written = skipped = 0
    for progress, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // chunk_size
        for segment in episode["action_config"]:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            text = str(segment["action_text"])
            candidate_count = (end - start + args.stride - 1) // args.stride
            video_frame_count = largest_1_plus_4k(candidate_count)
            if video_frame_count < 1:
                print(f"episode {episode_index} [{start},{end}) is empty; skipped")
                continue
            frame_ids = start + args.stride * np.arange(
                video_frame_count, dtype=np.int64
            )
            text_embedding = encoders.encode_text(text)

            for camera_key in args.camera_keys:
                filename = f"episode_{episode_index:06d}_{start}_{end}.pth"
                output_path = (
                    dataset_dir
                    / "latents"
                    / f"chunk-{episode_chunk:03d}"
                    / camera_key
                    / filename
                )
                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                video_path = (
                    dataset_dir
                    / "videos"
                    / f"chunk-{episode_chunk:03d}"
                    / camera_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video is missing: {video_path}")
                frames = read_video_frames(video_path, frame_ids)
                latent, latent_frames, latent_height, latent_width = (
                    encoders.encode_video(frames, args.height, args.width)
                )
                atomic_torch_save(
                    {
                        "latent": latent,
                        "latent_num_frames": latent_frames,
                        "latent_height": latent_height,
                        "latent_width": latent_width,
                        "video_num_frames": video_frame_count,
                        "video_height": args.height,
                        "video_width": args.width,
                        "text_emb": text_embedding,
                        "text": text,
                        "frame_ids": frame_ids,
                        "start_frame": start,
                        "end_frame": end,
                        "fps": stored_fps,
                        "ori_fps": original_fps,
                    },
                    output_path,
                )
                written += 1

        print(
            f"[{progress}/{len(episodes)}] episode {episode_index} done",
            flush=True,
        )

    print(
        f"Finished shard {args.shard_index}/{args.num_shards}: "
        f"{written} files written, {skipped} existing files skipped",
        flush=True,
    )


if __name__ == "__main__":
    main()
