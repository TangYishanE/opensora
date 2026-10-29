"""Single-device RoboTwin world-model fine-tuning with a finite step budget."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opensora.action_encoder import load_action_state_dict
from opensora.robotwin_dataset import RoboTwinDataset


def train_step(model, vae, scheduler, batch, optimizer, device, dtype, condition_frames):
    """Update STDiT on future frames using the scheduler's observed-prefix mask."""
    video = batch["video"].to(device=device, dtype=dtype)
    if condition_frames <= 0 or condition_frames + batch["action"].shape[1] != video.shape[2]:
        raise ValueError("Expected condition_frames + action horizon == video frames")
    with torch.no_grad():
        latents = vae.encode(video)
    if latents.shape[2] != video.shape[2]:
        raise ValueError("RoboTwin fine-tuning currently requires a temporal-preserving VAE")
    mask = torch.ones((video.shape[0], video.shape[2]), device=device, dtype=torch.bool)
    mask[:, :condition_frames] = 0
    args = {"y": batch["action"].to(device=device, dtype=dtype), "x_mask": mask}
    for name in ("fps", "height", "width", "num_frames"):
        args[name] = batch[name].to(device=device, dtype=dtype)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=torch.device(device).type, dtype=dtype, enabled=dtype in (torch.bfloat16, torch.float16)
    ):
        loss = scheduler.training_losses(model, latents, args, mask=mask)["loss"].mean()
    if not torch.isfinite(loss):
        raise ValueError("Expected finite world-model loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def build_dataset(cfg):
    """Select the finite dataset contract for the configured benchmark."""
    benchmark = cfg.get("benchmark", "robotwin")
    if benchmark == "robodojo":
        from opensora.robodojo_dataset import RoboDojoDataset

        return RoboDojoDataset(**cfg["dataset"])
    elif benchmark == "robotwin":
        return RoboTwinDataset(**cfg["dataset"])
    else:
        raise ValueError(f"Unsupported world-model training benchmark {benchmark}")


def run(cfg):
    from opensora.registry import MODELS, SCHEDULERS, build_module

    if not torch.cuda.is_available():
        raise RuntimeError("Full STDiT fine-tuning requires a CUDA GPU")
    if cfg["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    dataset = build_dataset(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg.get("seed", 42))
    loader = DataLoader(
        dataset, batch_size=cfg.get("batch_size", 1), shuffle=True, num_workers=cfg.get("num_workers", 0)
    )
    device, dtype = torch.device("cuda"), torch.bfloat16
    vae = build_module(cfg["vae"], MODELS).to(device, dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    model = build_module(
        cfg["model"],
        MODELS,
        input_size=vae.get_latent_size((dataset.context_length, *dataset.image_size)),
        in_channels=vae.out_channels,
        enable_sequence_parallelism=False,
    )
    checkpoint_path = cfg["init_checkpoint"]
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    audit = load_action_state_dict(model, state)
    (output / "checkpoint_load.json").write_text(json.dumps(audit, indent=2))
    # Keep optimizer/master parameters in fp32; autocast controls model compute.
    model.to(device).train()
    scheduler = build_module(cfg["scheduler"], SCHEDULERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-5))
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    history = []
    while len(history) < cfg["max_steps"]:
        for batch in loader:
            history.append(train_step(model, vae, scheduler, batch, optimizer, device, dtype, dataset.To))
            if len(history) >= cfg["max_steps"]:
                break
    torch.save(model.cpu().state_dict(), output / "model.pt")
    (output / "losses.json").write_text(json.dumps(history))
    return history


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
