# Primary experiment protocol

This protocol defines the frozen pre-LBA comparison. Architecture selection and
hyperparameter development use training and validation data only. The final held-out
test split remains sealed until the primary runs and model-selection rules are
complete.

## Research question

Can analytical assistance reduce learned resource requirements while preserving
useful UAV perception? End-to-end cost includes analytical extraction, mapping,
relation construction, model inference, and post-processing.

## Dataset and frozen splits

The dataset contract contains 250 rendered episodes and 20,000 frames. Split
generation uses seed 24051991 and groups complete episodes, trajectories, and
environment layouts. The frozen lock records:

| Split | Episodes | Frames |
|---|---:|---:|
| Train | 175 | 14,000 |
| Validation | 38 | 3,040 |
| Test | 37 | 2,960 |

data/manifests/manifest_lock.json stores manifest hashes. Normal readiness verifies
the hashes and split isolation; it does not regenerate matching manifests.

Readiness may verify that final-test source files exist and do not overlap other
splits. It must not calculate final-test task metrics, select thresholds, or compare
architectures.

## Compared systems

The primary sequence contains four registered systems:

1. CNN-HTransformer (CoHAtNet-inspired)
2. CNN-ViT
3. CNN
4. Imagination IMF-HTransformer

The first three receive their defined RGB observation. IMF-HTransformer receives
the versioned analytical tensor and validity. All models receive the shared
state/relation/candidate contract where specified. This is a system comparison; the
low-level observations are intentionally different. Existing front-end and
map-context ablations provide narrower comparisons.

The CNN-HTransformer baseline uses the project's controlled RGB front end and
MBConv-value attention. It is inspired by CoHAtNet and is not presented as an exact
reproduction of a published CoHAtNet configuration.

## Primary schedule

Each family completes every seed before the next family starts:

| Order | Model | Seeds |
|---:|---|---|
| 1 | cnn_htransformer | 7, 17, 29 |
| 2 | cnn_vit | 7, 17, 29 |
| 3 | cnn | 7, 17, 29 |
| 4 | imf_htransformer | 7, 17, 29 |

The maximum is 80 epochs per run. Early stopping uses patience 12, minimum epoch 8,
and minimum delta 0.001. The shared optimizer uses the frozen research learning
rate, weight decay, gradient clipping, loss definitions, and validation rule.
Automatic micro-batch probing preserves target effective batch size 8 with gradient
accumulation.

All primary artifacts are tagged:

    analytical_boundary: human_designed_pre_lba

The IMF Learning-Boundary Auditor is outside this experiment and is not invoked by
training readiness or the sequence.

## Targets

The source landing mask has classes unsafe 0, caution 1, and suitable 2. A 32 by 32
hazard cell is positive when at least 5 percent of its source pixels are unsafe or
caution. Landing is positive when at least 98 percent are suitable.

POI heatmaps use the configured simulator object classes. Semantic output groups
background, grass, track, cone, rock, and other obstacle. Scene labels are derived
from the documented target rules in data/preprocessing/targets.py.

Candidate verdict supervision is valid only when a predicted-map candidate has
resolved map XYZ, projects into the current frame, and agrees with the separately
cached target-depth cell within max(0.25 m, 15 percent of projected range).
Target-only depth is a supervision artifact. Invalid candidate targets are masked
from loss and metrics, and coverage is reported for train and validation.

## Causal map contract

At frame t, the model receives only prior predicted-map state, permitted analytical
geometry, and current vehicle state. Frame-t predictions update the map after
inference. Simulator semantic, landing, and depth targets never seed or update the
predicted map.

Episode ordering is preserved during map rollout. Training minibatch shuffling does
not alter the separately constructed causal contexts.

## Shared training contract

The four models use the same:

- train and validation manifests;
- target definitions and class registries;
- state, relation, and candidate contracts;
- shared heads where specified by the architecture;
- multitask loss and positive weights derived from train only;
- early-stopping rule, seed set, epoch budget, and effective batch;
- checkpoint, resume, logging, and validation infrastructure.

Architecture-specific observation and backbone differences remain explicit. A
baseline is not weakened to favor IMF.

## Validation and reporting

After each run, the orchestrator evaluates validation only and saves the best and
last checkpoints, history, metrics, effective configuration, resource metrics, and
training-plan hash. After all three seeds of one family, it reports mean and sample
standard deviation without selecting a winner.

Principal validation metrics include hazard false-negative rate, recall, precision,
and IoU; landing IoU and candidate detection rate; POI macro F1; semantic mean IoU;
scene accuracy; and candidate verdict metrics where supervision is valid.

Resource reporting separates preprocessing, map work, neural inference, and total
latency where measured. Attention score pairs, MACs, and wall-clock latency are
different quantities. Pair reduction alone is not reported as a speedup.

Threshold selection rules and any non-inferiority margins must be frozen before
final-test evaluation. Statistical claims require episode-aware uncertainty
analysis; three seeds alone do not establish broad generalization.

## Reproducibility

Training readiness writes a versioned plan containing manifest and configuration
hashes, contract versions, model order, seeds, epochs, early stopping, effective
batch plan, optimizer and losses, profile, device, and repository revision. Each
primary run records the plan hash. Matching completed runs are skipped, interrupted
runs resume, and incompatible checkpoints fail.

The canonical commands are:

    python train_research_sequence.py --prepare-only
    python train_research_sequence.py

The second command starts training only after readiness reports READY.

## Current claim boundary

The repository implements the analytical and learned division described in
[architecture.md](architecture.md). Primary task quality, end-to-end resource
savings, robustness outside the synthetic domain, and target-hardware energy remain
unmeasured. No result should be described as superior until the frozen experiment
has been run and analyzed.
