"""Convert an explicit RoboTwin HDF5 episode manifest to aligned NPZ and RLinf resets."""

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from opensora.action_stats import fit_action_statistics
from opensora.robotwin_dataset import validate_episode


def read_episode(path, image_key, action_key, alignment, color_order):
    """Decode a recorded episode; alignment must come from the source collector contract."""
    if color_order not in ("rgb", "bgr"):
        raise ValueError("Expected color_order 'rgb' or 'bgr'")
    with h5py.File(path, "r") as data:
        raw = data[image_key][:]
        actions = np.asarray(data[action_key][:], dtype=np.float32)
    if raw.ndim == 4 and raw.shape[-1] == 3:
        rgb = raw
    else:
        # Legacy collectors passed RGB arrays to OpenCV's BGR encoder without conversion.
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(bytes(frame))).convert("RGB")) for frame in raw])
    if color_order == "bgr":
        rgb = rgb[..., ::-1].copy()
    if alignment == "transition":
        pass  # The source already stores N+1 observations and N outgoing commands.
    elif alignment == "outgoing" and len(rgb) == len(actions):
        actions = actions[:-1]
    elif alignment == "incoming" and len(rgb) == len(actions):
        actions = actions[1:]
    else:
        raise ValueError("Expected explicit transition or equally sized outgoing/incoming source arrays")
    validate_episode(rgb, actions)
    return rgb, actions


def convert(manifest_path, output_dir, *, reader=read_episode, benchmark="robotwin"):
    """Convert without overwriting an existing dataset, fitting statistics on train only."""
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = source["episodes"]
    if not entries or not any(e["split"] == "train" for e in entries):
        raise ValueError("Expected a nonempty manifest containing train episodes")
    seen_ids, seen_paths = set(), set()
    action_sources = {item.get("action_source") for item in entries}
    if len(action_sources) != 1:
        raise ValueError("Do not mix recorded commands and state targets in one dataset")
    for item in entries:
        episode_id = item["episode_id"]
        path = (manifest_path.parent / item["path"]).resolve()
        if (
            not episode_id
            or Path(episode_id).name != episode_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in episode_id)
        ):
            raise ValueError("episode_id must contain only letters, digits, underscore or hyphen")
        if episode_id in seen_ids or str(path) in seen_paths:
            raise ValueError("Duplicate episode/source across splits")
        seen_ids.add(episode_id)
        seen_paths.add(str(path))
        if item["split"] not in ("train", "val", "test") or item["control_hz"] <= 0:
            raise ValueError("Expected train/val/test and positive control_hz")
        if item["action_order"] != "aloha" or item["action_representation"] != "absolute_joint":
            raise ValueError("Only verified aloha absolute_joint source actions are supported")
        if item.get("action_source") not in ("command", "state_target"):
            raise ValueError("Declare action_source as command or state_target; recorded joint states are not commands")
        if not item["instruction"].strip() or not item["source_revision"].strip():
            raise ValueError("instruction and source_revision must be recorded")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "episodes").mkdir()
    (output_dir / "init").mkdir()
    train_actions, output_entries = [], []
    for item in entries:
        path = (manifest_path.parent / item["path"]).resolve()
        rgb, actions = reader(path, item["image_key"], item["action_key"], item["alignment"], item["color_order"])
        relative_path = f"episodes/{item['episode_id']}.npz"
        np.savez_compressed(output_dir / relative_path, rgb=rgb, actions=actions)
        # Trusted local object NPY is the existing RLinf reset format, not training data.
        reset = {"image": rgb[0], "abs_action": actions[0], "instruction": item["instruction"]}
        if item["split"] == "train":
            train_actions.append(actions)
            np.save(output_dir / "init" / f"{item['episode_id']}.npy", np.asarray([reset], dtype=object))
        output_entries.append({**item, "path": relative_path, "source": str(path), "num_actions": len(actions)})
    stats = fit_action_statistics(train_actions)
    stats.update(
        schema_version=1,
        action_representation="absolute_joint",
        action_order="aloha",
        action_source=action_sources.pop(),
        benchmark=benchmark,
    )
    (output_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    result = {"schema_version": 1, "benchmark": benchmark, "episodes": output_entries}
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.manifest, args.output)


if __name__ == "__main__":
    main()
