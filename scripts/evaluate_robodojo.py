"""Evaluate RoboDojo trajectories with the shared autoregressive rollout evaluator."""

from scripts.evaluate_robotwin import main

if __name__ == "__main__":
    main(expected_benchmark="robodojo")
