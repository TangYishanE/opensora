"""Autoregressive trajectory evaluation with explicit horizon and pixel baselines."""

import numpy as np
import torch
from PIL import Image

from opensora.robotwin_dataset import validate_episode


def rollout_episode(backend, rgb, actions):
    """Predict full action chunks from one initial frame and close the session on exit."""
    validate_episode(rgb, actions, backend.action_dim)
    steps = len(actions) // backend.chunk * backend.chunk
    if steps == 0:
        raise ValueError("Episode is shorter than one world-model chunk")
    size = (backend.image_size[1], backend.image_size[0])
    resized = np.stack([np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)) for frame in rgb])
    first = torch.from_numpy(resized[0].copy()).permute(2, 0, 1).float() / 127.5 - 1
    frames = [[first.unsqueeze(1).clone() for _ in range(backend.condition_frame_length)]]
    backend.open_session([0], frames, torch.zeros(1, backend.condition_frame_length, backend.action_dim), [0])
    predictions = []
    try:
        for start in range(0, steps, backend.chunk):
            predicted = backend.generate([0], torch.from_numpy(actions[start : start + backend.chunk]).unsqueeze(0))
            expected = (1, 3, backend.chunk, *backend.image_size)
            if tuple(predicted.shape) != expected or not torch.isfinite(predicted).all():
                raise ValueError(f"Expected finite generated frames {expected}, got {tuple(predicted.shape)}")
            predictions.append(predicted.detach().float().cpu().squeeze(0).permute(1, 2, 3, 0).numpy())
    finally:
        backend.close_session([0])
    predicted = np.clip((np.concatenate(predictions) + 1) / 2, 0, 1)
    target = resized[1 : steps + 1].astype(np.float32) / 255
    mse = np.square(predicted - target).mean(axis=(1, 2, 3))
    baseline = np.square(resized[0].astype(np.float32) / 255 - target).mean(axis=(1, 2, 3))
    metrics = {
        "evaluated_steps": steps,
        "discarded_tail_steps": len(actions) - steps,
        "mse_by_horizon": mse.tolist(),
        "copy_first_mse_by_horizon": baseline.tolist(),
        "psnr_by_horizon": (-10 * np.log10(np.maximum(mse, 1e-12))).tolist(),
    }
    return np.rint(predicted * 255).astype(np.uint8), metrics
