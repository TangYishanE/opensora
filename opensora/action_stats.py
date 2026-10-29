"""Action normalization shared by world-model training and inference."""

import json
from pathlib import Path

import numpy as np


class ActionNormalizer:
    """Map active dimensions to quantile coordinates; constant dimensions become zero."""

    def __init__(self, stats, action_dim, stats_key=None):
        if stats_key is not None:
            stats = stats[stats_key]
        if "action" not in stats:
            raise ValueError("Expected top-level 'action' statistics or an explicit stats_key")
        self.action_source = stats.get("action_source", "command")
        self.benchmark = stats.get("benchmark")
        if self.action_source not in ("command", "state_target"):
            raise ValueError("Expected action_source 'command' or 'state_target'")
        action = stats["action"]
        self.low = np.asarray(action["q01"], dtype=np.float32)
        self.high = np.asarray(action["q99"], dtype=np.float32)
        self.mask = np.asarray(action.get("mask", [True] * action_dim), dtype=bool)
        for name, value in (("q01", self.low), ("q99", self.high), ("mask", self.mask)):
            if value.shape != (action_dim,):
                raise ValueError(f"Expected {name} shape {(action_dim,)}, got {value.shape}")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise ValueError("Action quantiles must be finite")
        if np.any(self.high < self.low):
            raise ValueError("Expected q99 >= q01")
        self.active = self.mask & (self.high > self.low)

    @classmethod
    def from_file(cls, path, action_dim, stats_key=None):
        with Path(path).open(encoding="utf-8") as stream:
            return cls(json.load(stream), action_dim, stats_key)

    def __call__(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim < 1 or actions.shape[-1] != len(self.low):
            raise ValueError(f"Expected action last dimension {len(self.low)}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Actions must be finite")
        scale = np.where(self.active, self.high - self.low, 1.0)
        normalized = 2.0 * (actions - self.low) / scale - 1.0
        # A disabled mask leaves its physical value unchanged, as in VLA statistics.
        normalized = np.where(self.mask, np.where(self.active, normalized, 0.0), actions)
        return normalized.astype(np.float32)


def fit_action_statistics(episodes):
    """Fit finite physical actions from explicitly selected training episodes."""
    arrays = [np.asarray(actions, dtype=np.float32) for actions in episodes]
    if not arrays or any(a.ndim != 2 or len(a) == 0 for a in arrays):
        raise ValueError("Expected nonempty [steps, action_dim] training episodes")
    if len({a.shape[1] for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Training actions must have one dimension and finite values")
    actions = np.concatenate(arrays)
    return {
        "action": {
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "mask": [True] * actions.shape[1],
        },
        "num_transitions": len(actions),
        "num_trajectories": len(arrays),
    }
