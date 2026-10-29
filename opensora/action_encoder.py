"""Continuous action tokens for STDiT video time steps."""

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, hidden_size, action_dim, cfg=False, training=False, pad_action_num=1):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.cfg = cfg
        self.pad_action_num = pad_action_num
        self.fc = nn.Sequential(
            nn.Linear(action_dim, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        self.none_action = nn.Embedding(1, hidden_size)
        self.train(training)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.action_dim:
            raise ValueError(f"Expected [batch, horizon, {self.action_dim}], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        output = self.fc(x)
        prefix = self.none_action.weight.expand(batch_size, self.pad_action_num, -1)
        output = torch.cat((prefix, output), dim=1)
        if self.cfg and self.training:
            drop = torch.rand(batch_size, 1, 1, device=x.device) < 0.1
            output = torch.where(drop, self.none_action.weight.view(1, 1, -1), output)
        return output.reshape(-1, self.hidden_size)


def load_action_state_dict(model, state_dict):
    """Load a full STDiT state, rebuilding only a changed action input projection."""
    target = model.state_dict()
    if target.keys() != state_dict.keys():
        raise ValueError("Expected a full STDiT checkpoint with matching parameter names")
    changed = [key for key in target if target[key].shape != state_dict[key].shape]
    if changed and changed != ["action_embedder.fc.0.weight"]:
        raise ValueError(f"Unexpected checkpoint shape mismatches: {changed}")
    retained = {key: value for key, value in state_dict.items() if key not in changed}
    result = model.load_state_dict(retained, strict=not changed)
    return {
        "reinitialized": changed,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
