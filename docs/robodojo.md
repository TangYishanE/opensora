# RoboDojo adaptation

Use both updated checkouts together. `RoboDojoDataset` and the RLinf backend share the OpenSora `ActionNormalizer`. The physical runtime is optional and loaded only by `RoboDojoEnv`.

## Docker module tests

From the parent project workspace, build the existing base test image and its RoboDojo extension:

```bash
docker compose -f experiments/environment/compose.robotwin.yaml build unit
docker compose -f experiments/environment/compose.robodojo.yaml build unit
docker compose -f experiments/environment/compose.robodojo.yaml run --rm unit
```

These CPU tests exercise converters, datasets, training helpers, evaluation guards, RLinf sessions and the physical adapter with an external vendor fake. The image is not a full CUDA training or Isaac Sim runtime. In a standalone clone, supply the paired RLinf checkout and mount paths yourself.

## Convert explicit source fields

Copy `config/robodojo_source_manifest.example.json` and set raw HDF5 paths. The example references the two released ARX-X5 bowl-stacking demos at dataset revision `269457a3b42d8d0ae69898a601f2c1487c2f826e`. It reads `vision/cam_head/colors` and the four `state/*_joint_states` groups, uses decoded RGB and the recorded 25 Hz frequency, and assigns separate train/test episodes.

```bash
python -m scripts.convert_robodojo --manifest /data/source.json --output /data/converted
```

The output directory must be new. Conversion writes aligned NPZ episodes, a manifest, train-only q01/q99 statistics and train-only RLinf reset files. Keep the statistics with the resulting checkpoint.

Declare `action_key` as an object with `keys` (four explicit group paths or one flat 14D path) and `order` (`native` or `aloha`). Native order is `[left arm:6, right arm:6, left gripper:1, right gripper:1]`; the canonical model order is `[left arm:6, left gripper:1, right arm:6, right gripper:1]`. No gripper inversion or joint-sign changes are applied.

`alignment: transition` requires N+1 observations and N outgoing commands. For equal-length arrays, `outgoing` drops the last command; `incoming` drops the first action to treat item t+1 as the target from image t to t+1. Select alignment from the collector contract. Do not repeat the original observation-shift correction on already corrected demos.

The example explicitly uses `action_source: state_target`: measured state t+1 supervises the image transition. This establishes a demonstration-target data path, not equivalence to a commanded joint target. For actual command recordings, select the verified `action/*` fields, correct alignment and `action_source: command`. Store the collector revision and instruction in every episode. Do not mix the two action sources.

## Train and evaluate

Set the dataset, VAE, initialization checkpoint and output paths in `config/train_robodojo_put_bottles_into_dustbin.yaml`, then run:

```bash
python -m scripts.train_robodojo --config config/train_robodojo_put_bottles_into_dustbin.yaml
```

This configuration is a template for the bottle task; the two public bowl demos validate the data path only. Use matching task data for task-specific fine-tuning. Four context frames, eight future frames/actions and `pad_action_num=4` must agree. The shared finite training loop freezes the VAE, migrates the input action projection from 7D to 14D and uses the production rectified-flow loss. Full STDiT XL training requires a separately prepared CUDA runtime and sufficient memory.

For held-out autoregressive evaluation:

```bash
python -m scripts.evaluate_robodojo --config /data/robodojo/backend-resolved.yaml --manifest /data/converted/manifest.json --split test --output /results/robodojo-eval
```

The config must be a resolved flat RLinf environment YAML containing `wm_env_type: robodojo` and `world_model_cfg`. Start from `env/opensora_robodojo_put_bottles_into_dustbin.yaml` in RLinf, resolve Hydra defaults, and set actual paths. Statistics and manifest must belong to RoboDojo. Enable `world_model_cfg.allow_state_target_actions: True` only for offline evaluation of a checkpoint trained with state-target data. The evaluator reports per-horizon MSE/PSNR against ground truth and a copy-first-frame baseline. It does not measure physical success.

The world-model preset is image-only, has empty proprioception, no wrist cameras and `reward_mode: none`; pair it with a policy trained for those inputs. The physical Pi05 preset uses three images and 14D proprioception and therefore cannot be used unchanged with this world model. Policy RL needs an appropriate target-domain reward model and validated command-conditioned dynamics.

## Physical RoboDojo runtime

RLinf registers `env_type: robodojo` separately from `env_type: world_model` / `wm_env_type: robodojo`. Its installer pins runtime commit `6d76a0c4a49a7a47bf5ee3924de2ccd7ecb8a8b7`, Isaac Sim 5.1, Torch 2.7 and the compatible IsaacLab/cuRobo forks.

On Linux x86_64 with NVIDIA CUDA and assets under `/data/robodojo/Assets`, use RLinf's `requirements/install.sh embodied --env robodojo`, or build `docker/Dockerfile` with `BUILD_TARGET=embodied-robodojo`. The installer sets Isaac Sim's `OMNI_KIT_ACCEPT_EULA=YES`. The Docker build checks installed sources; GPU availability is checked at runtime, not during image construction.

Run `python tests/e2e_tests/embodied/robodojo_env_smoke.py --output result.json` in that runtime with `ROBODOJO_ASSETS_ROOT` and `REPO_PATH` set. It resets layout 0, holds the measured joint targets for two chunks and closes the simulator. For physical Pi05 evaluation, set the checkpoint and normalization paths in `evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml`, then use `bash evaluations/run_eval.sh robodojo robodojo_put_bottles_into_dustbin_pi05_eval`.

The adapter supports native/ALOHA order, grouped seeds, partial resets, executed-action masks, success metrics, timeouts and bounded unstable-layout retries. `elapsed_steps` counts submitted steps; `valid_action_mask` records actually executed steps. Inspect reset attempts/seeds when comparing benchmark results. Keep `enable_offload: False` because runtime close is terminal. The legacy `success_seeds` list contains evaluation layout IDs, not policy outcomes.

The GPU smoke CI job is enabled only when the repository variable `ROBODOJO_ASSETS_ROOT` is configured on its GPU runner. CPU tests and real-demo data conversion have been run locally; full Isaac Sim, pretrained OpenSora generation and policy success evaluation remain unverified.
