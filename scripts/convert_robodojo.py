"""Convert explicit RoboDojo HDF5 fields into the shared OpenSora trajectory contract."""

import argparse
import io

import h5py
import numpy as np
from PIL import Image

from opensora.robotwin_dataset import validate_episode
from scripts.convert_robotwin import convert as convert_trajectories


def read_episode(path, image_key, action_key, alignment, color_order):
    """Decode head RGB and reorder four joint groups from native to ALOHA order.

    action_key is {keys: [four HDF5 paths] or one flat path, order: native|aloha}.
    Native group widths are 6,6,1,1; ALOHA widths are 6,1,6,1.
    """
    if not isinstance(action_key, dict) or action_key.get("order") not in ("native", "aloha"):
        raise ValueError("Declare action_key.order as native or aloha")
    keys, order = action_key["keys"], action_key["order"]
    with h5py.File(path, "r") as data:
        raw = data[image_key][:]
        if isinstance(keys, str):
            actions = np.asarray(data[keys][:], dtype=np.float32)
        else:
            widths = (6, 6, 1, 1) if order == "native" else (6, 1, 6, 1)
            if len(keys) != 4:
                raise ValueError("Expected four explicit action group paths")
            parts = []
            for key, width in zip(keys, widths):
                value = np.asarray(data[key][:], dtype=np.float32)
                if value.ndim == 1 and width == 1:
                    value = value[:, None]
                if value.ndim != 2 or value.shape[1] != width:
                    raise ValueError(f"Expected action group {key} width {width}, got {value.shape}")
                parts.append(value)
            actions = np.concatenate(parts, axis=1)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError("Expected 14 joint action coordinates")
    if order == "native":
        actions = actions[:, [*range(6), 12, *range(6, 12), 13]]
    if raw.ndim == 4 and raw.shape[-1] in (3, 4):
        rgb = raw[..., :3]
    else:
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(bytes(frame))).convert("RGB")) for frame in raw])
    if color_order not in ("rgb", "bgr"):
        raise ValueError("Expected decoded color_order rgb or bgr")
    if color_order == "bgr":
        rgb = rgb[..., ::-1].copy()
    if alignment == "transition":
        pass
    elif alignment == "incoming" and len(actions) == len(rgb):
        actions = actions[1:]
    elif alignment == "outgoing" and len(actions) == len(rgb):
        actions = actions[:-1]
    else:
        raise ValueError("Expected explicit transition or equally sized incoming/outgoing arrays")
    validate_episode(rgb, actions)
    return rgb, actions


def convert(manifest_path, output_dir):
    return convert_trajectories(manifest_path, output_dir, reader=read_episode, benchmark="robodojo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.manifest, args.output)


if __name__ == "__main__":
    main()
