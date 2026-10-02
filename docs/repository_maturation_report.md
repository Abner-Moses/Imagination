# Repository maturation report

This pass kept the four pre-LBA model architectures, losses, target definitions,
splits, and causal map order unchanged. The recorded workloads use CPU, batch size
2, 64 data samples, five model repetitions, and 16 causal-map frames on the same
development Mac. Raw JSON is stored in:

- artifacts/benchmarks/repository_maturation_before.json
- artifacts/benchmarks/repository_maturation_after.json

## Refactor

- Added one authoritative model registry for IDs, display names, factories,
  observation requirements, profiles, and architecture version.
- Centralized artifact paths and serialized contract versions.
- Defined shared input/output field contracts without changing tensor names.
- Rewrote the README, architecture overview, and primary protocol around the frozen
  research path and qualified CoHAtNet-inspired terminology.
- Split required, development, and optional Python dependencies.
- Added a fast dataset-independent project health command and minimal Ruff/Pytest
  configuration.
- Removed an unregistered, unused external CoHAtNet model module. The controlled
  CNN-HTransformer baseline and scientific ablations remain.
- Corrected analytical-cache progress accounting for already complete episodes.

## Measured data-path change

The dataset previously opened every analytical NPZ once for validation and again
for loading. It now loads and validates each sample in one pass. Manifest and camera
calibration parsing are cached with modification-time and file-size invalidation.
Numerical-equivalence tests compare loaded tensors with the underlying NPZ arrays.

| Measurement | Before | After | Delta |
|---|---:|---:|---:|
| RGB dataset initialization | 349.82 ms | 234.90 ms | -32.8% |
| Analytical dataset initialization | 487.75 ms | 13.61 ms | -97.2% |
| RGB sample loading | 80.95 samples/s | 119.30 samples/s | +47.4% |
| Analytical sample loading | 93.27 samples/s | 140.64 samples/s | +50.8% |

## Representative pipeline measurements

| Model | Forward p50, before → after | Training step p50, before → after | Map ms/frame, before → after | Peak RSS, before → after |
|---|---:|---:|---:|---:|
| CNN-HTransformer | 87.80 → 67.65 ms | 407.38 → 317.10 ms | 77.34 → 57.21 | 1,117.0 → 930.6 MB |
| CNN-ViT | 80.34 → 61.16 ms | 208.40 → 208.34 ms | 75.11 → 59.92 | 1,117.0 → 963.5 MB |
| CNN | 80.08 → 61.65 ms | 387.31 → 322.72 ms | 73.48 → 53.44 | 1,117.0 → 963.5 MB |
| IMF-HTransformer | 68.21 → 51.09 ms | 383.99 → 293.78 ms | 73.32 → 50.03 | 1,117.0 → 963.5 MB |

The data-path improvement follows directly from fewer file opens and parses. The
broader model, map, and RSS differences include normal process, filesystem-cache,
and host-load variation; they are recorded as observed measurements rather than
attributed wholly to the refactor. No attempted hot-path optimization with a
measured regression was retained.

## Behavior and compatibility

The numerical model architecture and primary configuration did not change. One
backend compatibility fix casts camera intrinsics and depth scale to float32 while
moving them into an MPS model; MPS does not support float64 tensors. CPU values and
serialized dataset fields remain unchanged.

Checkpoint, candidate, map, analytical, and model architecture versions were not
bumped because their serialized contracts did not change. Checkpoint schema,
training-plan, readiness, manifest, and training-statistics identifiers are now
declared in the central registry using their existing values.

Final validation passed: 80 Python tests plus four subtests, six C++ tests, the
four-model smoke suite, real-data forward/backward, four tiny overfit probes,
checkpoint resume, causal-map leakage checks, figure generation, and ONNX structural
checks for all four models. Ruff, compilation, and the dataset-independent health
command also pass. ONNX Runtime was unavailable for cross-runtime numerical checks.

The complete preparation command reports READY with training-plan hash
8e640f63b319436c035ebcf2c371231d3a305a23e34b464fd0d2fea7e1915748. Primary
training and final-test evaluation were not run.
