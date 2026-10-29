"""RoboDojo windows using canonical ALOHA-order joint targets."""

import json
from pathlib import Path

from opensora.registry import DATASETS
from opensora.robotwin_dataset import RoboTwinDataset


@DATASETS.register_module()
class RoboDojoDataset(RoboTwinDataset):
    """Reuse the finite trajectory contract while rejecting other benchmark manifests."""

    benchmark = "robodojo"

    def __init__(self, manifest_path, stats_path, **kwargs):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("benchmark") != self.benchmark:
            raise ValueError("RoboDojoDataset requires benchmark='robodojo'")
        super().__init__(manifest_path, stats_path, **kwargs)
