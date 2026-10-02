# IMF Learning-Boundary Auditor

IMF-LBA is a design-time research tool that operationalizes the rule “learn only what cannot be specified reliably in advance.” It measures whether each registered analytical responsibility satisfies a declared error, uncertainty, validity, and deployment-cost envelope. It does not prove that mathematics cannot solve a responsibility, generate neural modules, alter source code, change the 28-channel cache, or start training.

## Decision statistic

For responsibility `i` over calibration window `W`, IMF-LBA computes:

```text
e = measured error / accepted error ε
u = measured uncertainty / accepted uncertainty υ
f = failure rate / accepted failure rate φ
c = measured cost / cost budget κ

R = max(e, u, f, c)
```

The maximum is the primary safety-oriented statistic. A weighted average cannot allow excellent cost or uncertainty to cancel unacceptable error. The emitted arithmetic mean of known normalized components is diagnostic only.

Bootstrap resampling estimates `[R_lower, R_upper]` at the configured confidence level. The decision is:

- `ANALYTICAL` when `R_upper <= 1`;
- `LEARNING_CANDIDATE` when `R_lower > 1`;
- `UNDECIDED` when the interval crosses 1, evidence/coverage is insufficient, a diagnostic is unavailable, a metric is unmeasured, or a threshold is unfrozen;
- `DEPENDENCY_BLOCKED` when a required upstream responsibility is analytically insufficient or itself blocked.

`LEARNING_CANDIDATE` means the registered implementation failed its declared envelope. An optimized analytical method, learned assistance, or a hybrid alternative may be investigated. It does not mean a neural method is automatically better.

## Evidence and diagnostics

| Responsibility | Initial diagnostic | Evidence | Current limitation |
|---|---|---|---|
| Optical flow | Current luminance vs flow-warped previous luminance | Self-consistency | Not external motion truth; requires texture and motion |
| Depth reconstruction | Absolute metric error against declared calibration camera-Z depth | Supervised reference | Synthetic calibration evidence does not establish real-world accuracy |
| Repeated 3-D consistency | Local-map residual of flow-linked reconstructed points | Self-consistency | Depends on flow/depth/pose and is not independent ground truth |
| Map association | Support-covariance-normalized innovation of repeated points | Self-consistency | Proxy correspondences; duplicate/incorrect merge labels are unavailable |
| Semantic evidence fusion | Probability normalization, finiteness, evidence bound, contradiction response | Analytical identity | Requires real model-prediction sequences for deployment sufficiency |
| Visibility | Agreement at calibration-depth-projected points | Supervised reference | Current calibration probe lacks explicit occlusion/out-of-FOV cases |
| Drift prediction | Constant-velocity prediction vs subsequent trajectory displacement | Cross-sensor/reference trajectory | Simple physical model, not aerodynamic prediction |

The registry also contains explicit diagnostic placeholders for color conversion, gradients, HOG, corners, edges/contours, chroma transitions, depth derivatives, slope, roughness, geometry support, position uncertainty, density, restricted-zone distance, and landing clearance. No diagnostic is invented merely to force a decision.

Self-consistency evidence is not reported as equivalent to external ground truth. Every result records evidence type, strength, applicable frames, diagnostic samples, warnings, and metadata. A five-frame window can be useful for debugging but remains low confidence when minimum frames, samples, or environmental coverage are not met.

## Threshold policy

Limits live in `common/configs/imf_audit.yaml`. Every limit records value, units, source, and an optional note. Repository defaults are `null`, deliberately producing `UNDECIDED`. A research team must source and freeze limits from a mission requirement, sensor specification, physical requirement, literature, or explicit engineering budget before using IMF-LBA for architecture selection. A provisional development value must label itself as provisional.

Cached audits do not contain deployment-stage latency. Auditor runtime is recorded separately and never substituted for analytical-operator cost. Optional measured cost entries must state hardware, OS, backend, implementation, value, and units. Mac/PC timing is not Arduino Uno Q timing.

## Dependencies and MSK

The dependency graph prevents downstream blame. If depth is a learning candidate, depth gradients become `DEPENDENCY_BLOCKED(depth)`, followed by slope and roughness as appropriate. Their analytical implementations were not fairly tested.

`MSK` means analytically unavailable or rejected. At tensor level it remains a placeholder value plus validity:

```text
measured zero: value=0, validity=1
MSK:           value=0, validity=0
```

The suggested MSK contract names remaining upstream evidence and blocked downstream responsibilities. It is advisory and never mutates caches or model inputs automatically.

## Calibration and stability

Run a fixed window:

```sh
python run.py audit-imf --manifest data/manifests/validation_episode.jsonl --split val --frames 25
```

Or progressive windows:

```sh
python run.py audit-imf --manifest data/manifests/validation_episode.jsonl --split val --windows 5 10 25 50
```

Architecture-selection mode rejects final-test records. Each report stores manifest/calibration/config hashes, exact frame IDs, bootstrap seed, Git revision, software, hardware, thresholds, threshold sources, and operator versions.

Stability between windows is the fraction of shared resolved decisions that agree. Unresolved and dependency-blocked operators are excluded. If no decisions are resolved, stability is `N/A`, not zero or one.

## Outputs and workflow

Each unique run under `artifacts/results/imf_audit/` contains:

- `learning_boundary.json`, `.csv`, and `.md`;
- operator metrics and versioned registry;
- dependency graph JSON/DOT and rendered figure;
- progressive-window mask stability;
- audited 28-feature contract and advisory MSK contract;
- reproducibility metadata;
- ratio, normalized-component, status, stability, dependency, and cost/error figures.

The workflow stops after measurement and reporting:

```text
allowed calibration data -> IMF-LBA -> researcher review
                         -> human architecture decision -> later implementation/training
```

Operator correctness and task utility are separate. Correctly computing Sobel does not prove Sobel helps hazard detection; that remains a controlled training/ablation question. Likewise, marking depth `MSK` does not give a learner evidence from which depth can be inferred. The report therefore lists available upstream observations rather than proposing a model.

Automatic module synthesis, source modification, architecture search, online flight reconfiguration, continual analytical/neural switching, and differentiable boundary optimization are future research and are out of scope.
