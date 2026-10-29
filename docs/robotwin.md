# RoboTwin world-model adaptation

Use the explicit episode manifest to convert RoboTwin demonstrations, train an action-conditioned world model, and evaluate held-out video prediction. The first task is `place_empty_cup`, with a single head camera and 14 joint/hand values ordered as left arm, left hand, right arm, right hand.

## Data contract

`scripts/convert_robotwin.py` writes finite NPZ episodes containing uint8 RGB `[N+1,H,W,3]` and physical actions `[N,14]`. `obs[t]` precedes `action[t]`, which targets `obs[t+1]`. The converter also writes train-only quantile statistics and trusted local NPY reset files under `init/`. The reset action key is `abs_action`.

Start from `config/robotwin_source_manifest.example.json`. Supply an independently assigned `train`, `val` or `test` split for each episode, the HDF5 field paths, instruction, embodiment, source revision, action source and recording frequency. Verify these values against your collector; the example frequency and alignment are templates, not universal RoboTwin settings.

Choose `alignment: transition` for N+1 frames with N outgoing commands, `outgoing` for equal-length arrays with the last unused command removed, or `incoming` for equal-length arrays with the first previous command removed. JPEG/PNG byte arrays and raw HWC images are supported. `color_order` describes the array after image decoding: `bgr` explicitly swaps red and blue, including for legacy files written by passing RGB into OpenCV's BGR JPEG encoder. Inspect a frame and replay a trajectory before claiming control equivalence.

The inspected `RLinf_support` source records measured joint positions under `joint_action/vector`. Declare these as `action_source: state_target`, not `command`. They support structural conversion and state-conditioned demonstration experiments. Converting them does not establish equivalence to executing commands in the simulator. The RLinf backend rejects state-target statistics unless `world_model_cfg.allow_state_target_actions: true` is explicitly set for offline evaluation. For command-conditioned closed-loop experiments, collect the actual executed commands and their matching observations.

Run from the OpenSora checkout with that checkout on `PYTHONPATH`:

```bash
python scripts/convert_robotwin.py --manifest /data/robotwin/source_manifest.json --output /data/robotwin/converted
```

The output directory must be new. Conversion fails on missing fields, invalid dimensions, non-finite actions, mixed action sources, and duplicate source episodes. Quantiles are fitted on train episodes only. Constant quantile dimensions map to zero; masked dimensions keep their physical value. Training and RLinf inference use the same `ActionNormalizer` without clipping.

## Training

```bash
python scripts/train_robotwin.py --config config/train_robotwin_place_empty_cup.yaml
```

This separate, finite single-device entry uses `RoboTwinDataset`, the existing STDiT and rectified-flow objective, a frozen temporal-preserving SDXL VAE, fp32 optimizer parameters and bf16 autocast. It keeps the existing `train_web.py` workflow available. It writes the resolved configuration, checkpoint-loading audit, losses and `model.pt`. It does not resume optimizer state or replace the distributed ColossalAI training entry.

The 7D-to-14D checkpoint migration reinitializes only `action_embedder.fc.0.weight`. Other missing or differently shaped parameters fail loading. A full target-domain model must then be trained; the freshly initialized projection is not a valid trained robot controller. The ActionEncoder pads the configured number of condition tokens, including during classifier-free dropout, independently of the number of action coordinates.

Set actual dataset, checkpoint, VAE and output paths before running. `To=4`, `Ta=8` and `pad_action_num=4` must agree. The configuration defaults to five optimizer steps for a smoke test. Full XL training has not been validated on the project's 12GB GPU; reducing the step count does not reduce optimizer memory. Use measured memory requirements before planning full training.

## Offline evaluation

```bash
python scripts/evaluate_robotwin.py --config /configs/resolved_world_model.yaml --manifest /data/robotwin/converted/manifest.json --split test --output /results/prediction-run
```

The YAML must contain a fully resolved `world_model_cfg` with the target checkpoint, VAE and training statistics. Both the OpenSora and RLinf checkouts must be on `PYTHONPATH`. Evaluation starts from a repeated first frame and autoregressively executes complete eight-action chunks. It saves predicted frames, per-horizon MSE/PSNR and a copy-first-frame baseline, and records discarded tail actions. These are video-prediction metrics, not simulator task success.

## RLinf integration

`examples/embodiment/config/env/opensora_robotwin_place_empty_cup.yaml` in the paired RLinf checkout is an **inference-only** preset. It uses `action_key: abs_action`, `state_dim: 0`, `is_eval: true` and `reward_mode: none`. It returns empty proprioception and no wrist image; select a policy trained for those inputs. Rewards and predicted success remain zero in this mode. Using a policy checkpoint that needs real proprioception requires additional work.

Policy RL requires executed-command data, a trained target world model, a matching image-only policy and a validated target reward model. Verify the execution horizon and policy log-probability mask together. The existing 25-step RoboTwin policy configuration is not interchangeable with the eight-step world-model preset.

## Docker unit tests

Place the `opensora` and `Rlinf` checkouts next to each other. From their parent directory:

```bash
docker build -f opensora/docker/robotwin-unit.Dockerfile -t opensora-robotwin-unit:local .
docker run --rm -v "$PWD/opensora:/workspace/opensora:ro" -v "$PWD/Rlinf:/workspace/Rlinf:ro" opensora-robotwin-unit:local
```

The image pins CPU PyTorch and test dependencies. ColossalAI is installed without optional training dependencies solely to import the existing rectified-flow module; this image is a unit-test runtime, not a complete model-training image. Tests use small generation components while running production action conversion, session, reset and episode methods. They do not validate pretrained video quality or SAPIEN rendering. Test dependencies are in `requirements/requirements-robotwin-test.txt`.
