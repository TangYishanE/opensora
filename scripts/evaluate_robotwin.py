"""Evaluate a target-domain OpenSora checkpoint on aligned held-out RoboTwin episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from opensora.robotwin_evaluation import rollout_episode


def validate_benchmark(manifest, cfg, expected_benchmark):
    """Require the evaluation inputs to belong to the selected benchmark."""
    if manifest.get("benchmark", "robotwin") != expected_benchmark or cfg.wm_env_type != expected_benchmark:
        raise ValueError(f"Expected {expected_benchmark} manifest and wm_env_type")


def main(expected_benchmark="robotwin"):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved YAML with world_model_cfg")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cfg = OmegaConf.load(args.config)
    validate_benchmark(manifest, cfg, expected_benchmark)
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint evaluation requires a CUDA runtime image")
    from rlinf.envs.sim.world_model.backend.opensora import OpenSoraBackend

    entries = [item for item in manifest["episodes"] if item["split"] == args.split]
    if not entries:
        raise ValueError(f"No held-out episodes in split {args.split}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    backend = OpenSoraBackend(cfg, torch.device("cuda"))
    metrics = {}
    for entry in entries:
        with np.load(path.parent / entry["path"], allow_pickle=False) as data:
            predictions, result = rollout_episode(backend, data["rgb"], data["actions"])
        np.save(output / f"{entry['episode_id']}_prediction.npy", predictions)
        metrics[entry["episode_id"]] = result
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
