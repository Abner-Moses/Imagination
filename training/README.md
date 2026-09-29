# Imagination offline training

This directory contains the learned-model research layer. It does not replace the
C++ analytical extractor and is not needed on the flight computer.

```text
raw sequence -> C++ pre_extract -> cached features + validity -> learned model
RGB frame ---------------------------------------------> learned baseline
                                                        shared HTransformer
                                                        hazard + waypoint heads
```

The controlled experiment changes the low-level front end. Both models use the
same samples, labels, state representation, HTransformer, heads, losses, training
schedule, and metrics.

## Environment

Create a separate Python environment and install:

```sh
python -m pip install -r training/requirements.txt
```

ONNX export additionally needs `onnx`; numerical export checking needs
`onnxruntime`. None of these packages are linked into the C++ runtime.

## Analytical contract

The C++ exporter is authoritative. Dataset preparation reads its OpenCV
`frame_N.yml.gz` files and verifies this versioned order:

```text
Y, Cb, Cr, Gx, Gy, GradientMagnitude,
HOG_0..HOG_8, HarrisResponse, CannyEdge, ContourMap,
ChromaGradientCb, ChromaGradientCr, OpticalFlowU, OpticalFlowV,
Depth, DepthGradientX, DepthGradientY, Slope, Roughness, GeometryConfidence
```

Each cached NPZ contains `features` as float32 `[28,32,32]`, `validity` as
`[28,32,32]`, channel names, registry version, preprocessing hash, and timestamp.
The model computes `concat(features * validity, validity)`, producing
`[56,32,32]`. A measured zero therefore remains distinct from an unavailable value.

Vehicle state is separate. The initial registry contains acceleration, angular
velocity, attitude, and ultrasonic distance. Each field has an independent
validity bit. Missing measurements are zero-filled only as storage; the validity
bit remains false. State is encoded by a small MLP and used by the waypoint head.

## Raw episode format

Dataset units are complete episodes. Adjacent frames are never independently
randomized across splits.

```text
dataset/raw/episodes/episode_000001/
  metadata.json
  frames.jsonl
  calibration.yaml
  rgb/000000.jpg
  labels/hazard/000000.npy
  labels/hazard_validity/000000.npy   # optional
```

`metadata.json` contains:

```json
{
  "episode_id": "episode_000001",
  "environment_id": "room_A_layout_2",
  "trajectory_id": "trajectory_07",
  "split": "train",
  "source": "real",
  "label_provenance": "manual obstacle annotation protocol v1"
}
```

Every `frames.jsonl` row contains `frame_index`, strictly increasing `timestamp`,
`rgb_path`, `hazard_path`, optional `hazard_validity_path`, and `waypoint`.
Waypoint may be `[dx,dy,dz,yaw_rad]` or the stored
`[dx,dy,dz,sin_yaw,cos_yaw]`. Optional `sensors` fields use the state names from
`training/common.py` and matching `<name>_valid` flags. For C++ motion input,
`roll_rad`, `pitch_rad`, `yaw_rad`, `ultrasonic_m`, their validity flags, and a
sensor timestamp are recognized.

Hazard labels must come from simulator scene truth, manual annotation, or another
independent verified source. Do not derive targets from the analytical slope,
roughness, depth, or other model inputs. Waypoints must come from a demonstrated
trajectory, verified planner, simulator reference path, or manual validation.

## Prepare and validate data

Build C++ first, then edit `training/configs/dataset.yaml`:

```sh
python -m training.data.prepare_dataset --config training/configs/dataset.yaml
```

Preparation constructs an ordered sequence manifest, calls the existing
`imagination pre_extract --sequence --data` command once per episode, checks channel
order and shape, and writes a compressed cache plus `manifest.jsonl`. The cache key
includes frames, episode metadata, extractor configuration hash, and channel
registry. Dataset validation checks images, tensors, masks, labels, timestamps,
identities, waypoints, and train/validation/test leakage.

The generated manifest contains one row per frame and records sample, episode,
environment, trajectory, split, source, label provenance, paths, state/masks,
waypoint, and flow/geometry availability. Entire environment, episode, and
trajectory identities are restricted to one split.

For a code-only smoke fixture:

```sh
python -m training.data.prepare_dataset \
  --synthetic-smoke /tmp/imagination-smoke --frames-per-split 4
```

This fixture is explicitly not research data.

## Models

The analytical path is:

```text
28 features + 28 validity -> 1x1 56-to-192 adapter -> shared HTransformer
```

The baseline path is:

```text
RGB 256x256 -> learned Conv/MBConv front end -> 192x32x32 -> shared HTransformer
```

The HTransformer preserves the CoHAtNet-style division:

- separate linear projections produce Q and K from spatial tokens;
- V is produced by MBConv over the 2-D feature map;
- MBConv contains pointwise expansion, depthwise spatial convolution,
  squeeze-and-excitation, pointwise projection, and a conditional residual;
- relative positional bias is added to `QK^T / sqrt(d)` before softmax.

Thus analytical extraction supplies explicit low-level quantities, MBConv learns
local combinations, and attention learns global relationships. V is not produced
by a conventional linear value projection.

Reference: Hasan et al., *CoHAtNet: An integrated convolutional-transformer
architecture with hybrid self-attention for end-to-end camera localization*,
Image and Vision Computing 162 (2025), DOI
[`10.1016/j.imavis.2025.105674`](https://doi.org/10.1016/j.imavis.2025.105674),
and the authors' [reference implementation](https://github.com/Husseinhhameed/CoHAtNet).

The shared hazard decoder produces one `32x32` logit map. The shared navigation
head predicts `[dx,dy,dz,sin_yaw,cos_yaw]` in the local UAV frame.

## Train, resume, and evaluate

Edit the manifest path in both configs, then run:

```sh
python -m training.train --config training/configs/analytical.yaml
python -m training.train --config training/configs/baseline.yaml
python -m training.train --config training/configs/analytical.yaml \
  --resume training/runs/analytical/last.pt

python -m training.evaluate \
  --checkpoint training/runs/analytical/best.pt \
  --manifest dataset/prepared/manifest.jsonl --split test \
  --output training/results/analytical
```

Checkpoints record model, optimizer, scheduler, epoch, global step, resolved config,
metrics, manifest hash, channel registry/version, and Git commit. Runs also record
the seed, package versions, device, platform, and CSV metrics.

Hazard training uses masked class-weighted BCE plus configurable Dice or Tversky
overlap. Tversky defaults weight false negatives more strongly. Navigation uses
SmoothL1 for translation and normalized sine/cosine yaw. Evaluation reports TP,
TN, FP, FN, precision, recall, FNR, FPR, F1, IoU, translation MAE/RMSE, and heading
error overall and by environment and episode.

Compare two checkpoints using explicit, researcher-selected margins:

```sh
python -m training.evaluate \
  --compare training/runs/analytical/best.pt training/runs/baseline/best.pt \
  --manifest dataset/prepared/manifest.jsonl \
  --fnr-epsilon 0.01 --navigation-delta 0.10 \
  --output training/results/comparison
```

The tool reports whether those configured numerical criteria are met. It does not
claim statistical significance. Model latency and parameters are reported
separately from placeholders for C++ preprocessing, total latency, memory, and
energy, so model-only cost is never presented as end-to-end system cost.

## Ablations

Representation ablations mask channels and validity only inside the learned model:

```sh
python -m training.ablate --config training/configs/analytical.yaml \
  --without hog,flow --mode representation \
  --output training/runs/ablation_hog_flow
```

End-to-end compute ablations also generate a matching C++ extractor configuration.
Regenerate the cache before training:

```sh
python -m training.ablate --config training/configs/analytical.yaml \
  --without hog,flow --mode end_to_end \
  --extractor-config configs/pre_extraction.yaml \
  --dataset-config training/configs/dataset.yaml \
  --output training/runs/ablation_hog_flow_e2e
```

Run `training.data.prepare_dataset` with the generated `dataset.yaml`, then point
the generated experiment at that manifest. The experiment metadata explicitly
distinguishes representation and end-to-end claims.

## ONNX and tests

```sh
python -m training.export_onnx \
  --checkpoint training/runs/analytical/best.pt \
  --output training/models/imagination_analytical.onnx

python -m unittest training.tests.test_training -v
```

The ONNX graph begins at analytical features, validity, and state. It does not
contain the C++ extractor. If ONNX Runtime is installed, export is numerically
checked against PyTorch.

No spatial augmentation is applied to cached features in v1. Flips and rotations
would require exact changes to derivative signs, HOG bins, flow, geometry, waypoint,
and yaw. Spatial augmentation should instead be applied to complete raw sequences
before running the C++ extractor.
