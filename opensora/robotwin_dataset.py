"""Finite RoboTwin windows with obs[t] -- action[t] --> obs[t+1]."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from opensora.action_stats import ActionNormalizer
from opensora.registry import DATASETS


def validate_episode(rgb, actions, action_dim=14):
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB [observations, height, width, 3]")
    if actions.ndim != 2 or actions.shape[1] != action_dim or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite actions [steps, {action_dim}]")
    if len(actions) == 0 or len(rgb) != len(actions) + 1:
        raise ValueError("Expected N+1 observations for N nonempty actions")


@DATASETS.register_module()
class RoboTwinDataset(Dataset):
    """Map-style windows over converted NPZ episodes; no resampling during validation."""

    benchmark = "robotwin"

    def __init__(
        self, manifest_path, stats_path, split="train", Ta=8, To=4, stride=1, image_size=(256, 256), action_dim=14
    ):
        if min(Ta, To, stride, *image_size) <= 0:
            raise ValueError("Window geometry and image dimensions must be positive")
        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = Ta + To
        self.image_size = tuple(image_size)
        self.action_dim = action_dim
        self.normalizer = ActionNormalizer.from_file(stats_path, action_dim)
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("benchmark", self.benchmark) != self.benchmark:
            raise ValueError(f"Expected {self.benchmark} manifest")
        if self.normalizer.benchmark not in (None, self.benchmark):
            raise ValueError(f"Expected {self.benchmark} training statistics")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Expected RoboTwin manifest schema_version=1")
        self.episodes, self.windows = [], []
        seen_ids, seen_sources = set(), set()
        for item in self.manifest["episodes"]:
            if item["episode_id"] in seen_ids or item["source"] in seen_sources:
                raise ValueError("Duplicate episode/source across manifest splits")
            seen_ids.add(item["episode_id"])
            seen_sources.add(item["source"])
            if item["split"] != split:
                continue
            if item["action_order"] != "aloha" or item["action_representation"] != "absolute_joint":
                raise ValueError("Expected absolute_joint actions in aloha order")
            if item.get("action_source") != self.normalizer.action_source:
                raise ValueError("Episode action_source must match its training statistics")
            episode_path = path.parent / item["path"]
            with np.load(episode_path, allow_pickle=False) as data:
                validate_episode(data["rgb"], data["actions"], action_dim)
                steps = len(data["actions"])
            episode_idx = len(self.episodes)
            self.episodes.append((episode_path, item))
            self.windows.extend((episode_idx, start) for start in range(0, steps - Ta + 1, stride))
        if not self.windows:
            raise ValueError(f"No full action windows in split {split!r}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_idx, start = self.windows[index]
        path, meta = self.episodes[episode_idx]
        with np.load(path, allow_pickle=False) as data:
            rgb = data["rgb"]
            # Conditions end at obs[start]; predictions are obs[start+1:start+Ta+1].
            indices = np.maximum(np.arange(start - self.To + 1, start + self.Ta + 1), 0)
            frames = [
                np.asarray(
                    Image.fromarray(rgb[i]).resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
                )
                for i in indices
            ]
            video = torch.from_numpy(np.stack(frames).copy()).permute(3, 0, 1, 2).float() / 127.5 - 1
            actions = self.normalizer(data["actions"][start : start + self.Ta])
        return {
            "video": video,
            "action": torch.from_numpy(actions),
            "fps": float(meta["control_hz"]),
            "num_frames": self.context_length,
            "height": self.image_size[0],
            "width": self.image_size[1],
            "meta": {"episode_id": meta["episode_id"], "start": start},
        }
