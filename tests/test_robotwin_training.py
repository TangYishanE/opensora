import numpy as np
import pytest
import torch
from torch import nn

from opensora.action_encoder import ActionEncoder, load_action_state_dict
from opensora.schedulers.rf.rectified_flow import RFlowScheduler
from scripts.train_robotwin import train_step


class SmallDynamics(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_embedder = ActionEncoder(4, action_dim, pad_action_num=4)


def test_checkpoint_migration_and_reload():
    source, target = SmallDynamics(7), SmallDynamics(14)
    original = target.action_embedder.fc[0].weight.detach().clone()
    report = load_action_state_dict(target, source.state_dict())
    assert report["reinitialized"] == ["action_embedder.fc.0.weight"]
    torch.testing.assert_close(target.action_embedder.fc[0].weight, original)
    torch.testing.assert_close(target.action_embedder.fc[2].weight, source.action_embedder.fc[2].weight)
    assert load_action_state_dict(SmallDynamics(14), target.state_dict())["reinitialized"] == []
    with pytest.raises(ValueError):
        load_action_state_dict(target, {})
    state = source.state_dict()
    state["action_embedder.fc.2.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Unexpected"):
        load_action_state_dict(target, state)


class FrameVAE(nn.Module):
    def encode(self, video):
        return video


class SmallFlowDynamics(SmallDynamics):
    def forward(self, x, timestep, y, x_mask, **kwargs):
        assert x_mask.dtype == torch.bool
        assert not x_mask[:, :4].any() and x_mask[:, 4:].all()
        prediction = self.action_embedder(y).reshape(len(x), 12, 4)
        prediction = prediction.mean(-1)[:, None, :, None, None].expand_as(x)
        return torch.cat((prediction, prediction), dim=1)


def test_training_update_and_saved_state(tmp_path):
    torch.manual_seed(12)
    model = SmallFlowDynamics(14)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.action_embedder.fc[0].weight.detach().clone()
    batch = {"video": torch.randn(2, 3, 12, 4, 4), "action": torch.randn(2, 8, 14)}
    batch.update({key: torch.ones(2) for key in ("fps", "height", "width", "num_frames")})
    loss = train_step(model, FrameVAE(), RFlowScheduler(), batch, optimizer, "cpu", torch.float32, 4)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.action_embedder.fc[0].weight)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = SmallDynamics(14)
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    torch.testing.assert_close(model.action_embedder(batch["action"]), restored.action_embedder(batch["action"]))


@pytest.mark.parametrize("script", ["convert_robotwin.py", "train_robotwin.py", "evaluate_robotwin.py"])
def test_cli_help(script):
    import subprocess
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
