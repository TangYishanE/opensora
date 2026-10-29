import io
import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from opensora.registry import DATASETS
from opensora.robotwin_dataset import RoboTwinDataset
from scripts.convert_robotwin import convert, read_episode


def write_source(tmp_path, encoded=False):
    entries = []
    for index, split in enumerate(("train", "val", "test")):
        rgb = np.stack([np.full((6, 8, 3), t * 10, np.uint8) for t in range(11)])
        # Incoming commands encode their destination observation index.
        actions = np.repeat(np.arange(11, dtype=np.float32)[:, None], 14, axis=1)
        actions += 1000 * index
        path = tmp_path / f"episode_{index}.hdf5"
        with h5py.File(path, "w") as data:
            if encoded:
                images = data.create_dataset("rgb", (11,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for t, frame in enumerate(rgb):
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format="PNG")
                    images[t] = np.frombuffer(buffer.getvalue(), np.uint8)
            else:
                data["rgb"] = rgb
            data["actions"] = actions
        entries.append(
            dict(
                episode_id=f"ep_{index}",
                path=path.name,
                split=split,
                image_key="rgb",
                action_key="actions",
                alignment="incoming",
                color_order="rgb",
                control_hz=30,
                instruction="place cup",
                source_revision="fixture-v1",
                task="place_empty_cup",
                embodiment="piper",
                action_order="aloha",
                action_representation="absolute_joint",
                action_source="command",
            )
        )
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"episodes": entries}))
    return manifest


@pytest.mark.parametrize("encoded", [False, True])
def test_conversion_windows_statistics_and_registry(tmp_path, encoded):
    source = write_source(tmp_path, encoded)
    output = tmp_path / "converted"
    convert(source, output)
    stats = json.loads((output / "dataset_statistics.json").read_text())
    assert stats["num_transitions"] == 10
    assert max(stats["action"]["q99"]) < 11  # val/test actions are above 1000
    assert len(list((output / "init").glob("*.npy"))) == 1
    cfg = dict(
        type="RoboTwinDataset",
        manifest_path=str(output / "manifest.json"),
        stats_path=str(output / "dataset_statistics.json"),
        image_size=(3, 4),
    )
    cls = DATASETS.module_dict[cfg.pop("type")]
    dataset = cls(**cfg)
    assert isinstance(dataset, RoboTwinDataset)
    assert len(dataset) == 3
    first = dataset[0]
    assert first["video"].shape == (3, 12, 3, 4)
    # Four conditions are obs[0], then obs[1:9] are future targets.
    expected = torch.tensor([0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70, 80]) / 127.5 - 1
    torch.testing.assert_close(first["video"][0, :, 0, 0], expected)
    restored = (first["action"].numpy() + 1) / 2 * (
        np.array(stats["action"]["q99"]) - np.array(stats["action"]["q01"])
    ) + stats["action"]["q01"]
    np.testing.assert_allclose(restored[:, 0], np.arange(1, 9), atol=1e-6)
    batches = list(DataLoader(dataset, batch_size=1, num_workers=2))
    assert sorted(int(batch["meta"]["start"]) for batch in batches) == [0, 1, 2]
    with pytest.raises(FileExistsError):
        convert(source, output)


def test_duplicate_sources_fail_before_output_creation(tmp_path):
    source = write_source(tmp_path)
    entries = json.loads(source.read_text())
    entries["episodes"][1]["path"] = entries["episodes"][0]["path"]
    source.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="Duplicate"):
        convert(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_alignment_is_explicit(tmp_path):
    write_source(tmp_path)
    path = tmp_path / "episode_0.hdf5"
    _, incoming = read_episode(path, "rgb", "actions", "incoming", "rgb")
    _, outgoing = read_episode(path, "rgb", "actions", "outgoing", "rgb")
    assert incoming[0, 0] == 1 and outgoing[0, 0] == 0
    with pytest.raises(ValueError):
        read_episode(path, "rgb", "actions", "transition", "rgb")


def test_official_fixed_width_jpeg_and_raw_color_order(tmp_path):
    path = tmp_path / "images.hdf5"
    raw = np.zeros((3, 6, 8, 3), np.uint8)
    raw[..., 0] = 240
    encoded = []
    for frame in raw:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG")
        encoded.append(buffer.getvalue())
    with h5py.File(path, "w") as data:
        data.create_dataset("rgb", data=encoded, dtype=f"S{max(map(len, encoded))}")
        data["raw_bgr"] = raw[..., ::-1]
        data["actions"] = np.zeros((2, 14), np.float32)
    rgb, _ = read_episode(path, "rgb", "actions", "transition", "rgb")
    assert rgb[0, 0, 0, 0] > 230 and rgb[0, 0, 0, 2] < 10
    restored, _ = read_episode(path, "raw_bgr", "actions", "transition", "bgr")
    np.testing.assert_array_equal(restored, raw)
    corrected, _ = read_episode(path, "rgb", "actions", "transition", "bgr")
    np.testing.assert_array_equal(corrected, rgb[..., ::-1])
