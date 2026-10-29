import pytest
import torch

from opensora.action_encoder import ActionEncoder


@pytest.mark.parametrize("action_dim", [7, 14])
@pytest.mark.parametrize("cfg", [False, True])
def test_action_tokens_shape_gradients_and_cfg(action_dim, cfg):
    torch.manual_seed(123)
    model = ActionEncoder(16, action_dim, cfg=cfg, training=True, pad_action_num=4)
    action = torch.randn(32, 8, action_dim, requires_grad=True)
    output = model(action)
    assert output.shape == (32 * 12, 16)
    output.square().mean().backward()
    assert torch.isfinite(action.grad).all()
    assert model.fc[0].weight.grad.abs().sum() > 0
    model.eval()
    assert torch.equal(model(action), model(action))


def test_wrong_dimension_rejected():
    with pytest.raises(ValueError, match="14"):
        ActionEncoder(8, 14)(torch.zeros(1, 8, 7))
