import numpy as np
import pytest

from opensora.action_stats import ActionNormalizer, fit_action_statistics


def test_normalization_and_constant_dimensions():
    normalizer = ActionNormalizer({"action": {"q01": [0, 2, 0], "q99": [2, 2, 1], "mask": [True, True, False]}}, 3)
    np.testing.assert_array_equal(
        normalizer([[0, 2, 0.7], [2, 2, 0.3]]), np.array([[-1, 0, 0.7], [1, 0, 0.3]], np.float32)
    )


@pytest.mark.parametrize(
    "stats,dim",
    [
        ({"action": {"q01": [0], "q99": [1]}}, 14),
        ({"action": {"q01": [2], "q99": [1]}}, 1),
        ({"action": {"q01": [0], "q99": [np.nan]}}, 1),
    ],
)
def test_invalid_statistics(stats, dim):
    with pytest.raises(ValueError):
        ActionNormalizer(stats, dim)


def test_explicit_statistics_key():
    stats = {"train": {"action": {"q01": [0], "q99": [2]}}}
    with pytest.raises(ValueError):
        ActionNormalizer(stats, 1)
    np.testing.assert_array_equal(ActionNormalizer(stats, 1, "train")([[1]]), [[0]])


def test_fit_and_input_validation():
    stats = fit_action_statistics([np.arange(28).reshape(2, 14)])
    assert len(stats["action"]["q01"]) == 14
    assert stats["num_transitions"] == 2
    norm = ActionNormalizer(stats, 14)
    with pytest.raises(ValueError):
        norm(np.zeros((2, 7)))
    with pytest.raises(ValueError):
        norm(np.full((2, 14), np.inf))
    with pytest.raises(ValueError):
        fit_action_statistics([])
