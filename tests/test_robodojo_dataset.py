import io
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from opensora.robodojo_dataset import RoboDojoDataset
from scripts.convert_robodojo import convert, read_episode


def source_manifest(tmp_path, encoded=False):
    episodes = []
    for i, split in enumerate(("train", "val", "test")):
        path = tmp_path / f"episode{i}.hdf5"
        frames = np.zeros((11, 4, 6, 3), np.uint8)
        frames[..., 0] = np.arange(11)[:, None, None] * 10
        actions = np.tile(np.arange(14, dtype=np.float32), (11, 1)) + np.arange(11)[:, None] + i * 100
        with h5py.File(path, "w") as f:
            if encoded:
                d = f.create_dataset("vision/cam_head/color", (11,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for t, frame in enumerate(frames):
                    stream = io.BytesIO()
                    Image.fromarray(frame).save(stream, format="PNG")
                    d[t] = np.frombuffer(stream.getvalue(), np.uint8)
            else:
                f["vision/cam_head/color"] = frames
            for key, part in zip(
                ("left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state"),
                np.split(actions, [6, 12, 13], axis=-1),
            ):
                f["action/" + key] = part
            f["flat"] = actions
        episodes.append(
            dict(
                episode_id=f"e{i}",
                path=path.name,
                split=split,
                image_key="vision/cam_head/color",
                action_key=dict(
                    keys=[
                        "action/left_arm_joint_state",
                        "action/right_arm_joint_state",
                        "action/left_ee_joint_state",
                        "action/right_ee_joint_state",
                    ],
                    order="native",
                ),
                action_order="aloha",
                action_representation="absolute_joint",
                action_source="command",
                alignment="incoming",
                color_order="rgb",
                control_hz=30,
                instruction="put bottles into dustbin",
                source_revision="fixture-v1",
                embodiment="arx_x5",
            )
        )
    path = tmp_path / "source.json"
    path.write_text(json.dumps(dict(episodes=episodes)))
    return path


@pytest.mark.parametrize("encoded", [False, True])
def test_conversion_order_windows_and_train_only_stats(tmp_path, encoded):
    manifest = source_manifest(tmp_path, encoded)
    convert(manifest, tmp_path / "out")
    output = tmp_path / "out"
    dataset = RoboDojoDataset(output / "manifest.json", output / "dataset_statistics.json", image_size=(2, 3))
    assert len(dataset) == 3 and dataset[0]["video"].shape == (3, 12, 2, 3)
    with np.load(output / "episodes/e0.npz") as data:
        np.testing.assert_array_equal(data["actions"][0], [1, 2, 3, 4, 5, 6, 13, 7, 8, 9, 10, 11, 12, 14])
    stats = json.loads((output / "dataset_statistics.json").read_text())
    assert stats["benchmark"] == "robodojo" and max(stats["action"]["q99"]) < 24
    assert len(list((output / "init").glob("*.npy"))) == 1
    assert len(RoboDojoDataset(output / "manifest.json", output / "dataset_statistics.json", split="test")) == 3


def test_flat_order_and_invalid_group_width(tmp_path):
    source_manifest(tmp_path)
    path = tmp_path / "episode0.hdf5"
    _, native = read_episode(path, "vision/cam_head/color", dict(keys="flat", order="native"), "incoming", "rgb")
    _, aloha = read_episode(path, "vision/cam_head/color", dict(keys="flat", order="aloha"), "incoming", "rgb")
    assert native[0, 6] == 13 and aloha[0, 6] == 7
    with pytest.raises(ValueError):
        read_episode(path, "vision/cam_head/color", dict(keys=["flat"] * 4, order="native"), "incoming", "rgb")
    with pytest.raises(ValueError):
        read_episode(path, "vision/cam_head/color", "flat", "incoming", "rgb")


def test_benchmark_statistics_cannot_be_crossed(tmp_path):
    convert(source_manifest(tmp_path), tmp_path / "out")
    path = tmp_path / "out/dataset_statistics.json"
    stats = json.loads(path.read_text())
    stats["benchmark"] = "robotwin"
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="statistics"):
        RoboDojoDataset(tmp_path / "out/manifest.json", path)


@pytest.mark.parametrize("script", ["convert_robodojo.py", "train_robodojo.py", "evaluate_robodojo.py"])
def test_cli_help(script):
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_evaluation_rejects_cross_benchmark_inputs():
    from omegaconf import OmegaConf

    from scripts.evaluate_robotwin import validate_benchmark

    validate_benchmark({"benchmark": "robodojo"}, OmegaConf.create({"wm_env_type": "robodojo"}), "robodojo")
    for manifest_domain, cfg_domain in [("robotwin", "robodojo"), ("robodojo", "libero")]:
        with pytest.raises(ValueError, match="Expected robodojo"):
            validate_benchmark(
                {"benchmark": manifest_domain}, OmegaConf.create({"wm_env_type": cfg_domain}), "robodojo"
            )


def test_training_selects_robodojo_dataset_and_rejects_wrong_cli_config(tmp_path):
    import yaml

    from scripts.train_robotwin import build_dataset

    convert(source_manifest(tmp_path), tmp_path / "out")
    cfg = {
        "benchmark": "robodojo",
        "dataset": {
            "manifest_path": tmp_path / "out/manifest.json",
            "stats_path": tmp_path / "out/dataset_statistics.json",
        },
    }
    assert isinstance(build_dataset(cfg), RoboDojoDataset)
    cfg["benchmark"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported"):
        build_dataset(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"benchmark": "robotwin"}))
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_robodojo", "--config", str(path)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "benchmark: robodojo" in result.stderr
