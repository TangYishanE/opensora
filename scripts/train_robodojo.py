"""Train the action-conditioned OpenSora model on a RoboDojo manifest."""

import argparse
from pathlib import Path

import yaml

from scripts.train_robotwin import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("benchmark") != "robodojo":
        raise ValueError("Expected benchmark: robodojo in the training configuration")
    run(cfg)


if __name__ == "__main__":
    main()
