# Current architecture and readiness

This is the current repository snapshot. Superseded restructure notes and pre-v7 measurements were removed to keep one authoritative status record.

## Repository consolidation

Generated builds, checkpoints, plots, results, caches, an obsolete ZIP, and the local virtual environment were removed from the working tree without touching `data/`. The environment is reproducible from `common/requirements.txt`. Superseded audit/config snapshots were removed from `docs/archive`; this file now holds the current readiness record. The three RGB families share one `RGBPerceptionModel`, configuration merging has one implementation, and the ablation tool uses the active model/config paths. Active files outside `data/` and `.git` occupy about 2 MiB.

## Architecture

The repository contains four model families with shared data, heads, losses, metrics, checkpointing, evaluation, and benchmarking:

| Family | Visual path | Research role |
|---|---|---|
| CNN | RGB convolution | Local learned baseline |
| CNN–ViT | RGB convolution + conventional attention | Standard learned global-context baseline |
| CNN–HTransformer | RGB convolution + MBConv-value attention | Isolates convolution-assisted attention |
| IMF–HTransformer | Analytical families + IMF map/context + MBConv-value attention | Proposed mathematically assisted model |

IMF provides the fixed 28×32×32 feature/validity contract, separately masked 13-value state, a causal uncertainty-aware semantic map, 30 global relations, and up to 32 candidates with 40 masked features each. Candidate geometry is egocentric for learning while projection retains absolute map XYZ. The learned outputs are hazard, landing, semantic and POI maps plus scene and candidate verdicts.

Spatial HTransformer attention keeps distinct learned Q/K and `V = MBConv(F)` with no conventional linear spatial V projection. Unordered map tokens use independent pointwise K/V encoders. Attention mode, context use, Q/K width, heads, neighbors, active queries, and FFN width are configurable per stage.

## Current corrections

| Previous issue | Current behavior |
|---|---|
| Flat heterogeneous IMF projection | Appearance, motion, and geometry use separate mask-aware lightweight adapters; flat fusion remains an ablation |
| Point-only semantic map | Entities store XYZ, covariance/provenance, bounded semantic evidence, posterior uncertainty, support, age, and extent |
| Euclidean-only association | Local pre-gating followed by semantic compatibility and Mahalanobis gating; Euclidean fallback is recorded |
| Arithmetic temporal averaging | Correlation-discounted bounded Dirichlet-style evidence fusion |
| Map-relative coordinates used as absolute XYZ | Merged entities retain absolute map XYZ; egocentric and map-relative values remain separate |
| UAV-relative candidate clearance | Candidate-centered point-to-disc obstacle/restricted clearances, free radius, drifted clearance, and landing margin |
| Mixed drift frames | Deterministic drift remains map-frame; learned drift is yaw-aligned egocentric |
| Missing candidate values represented as zero | Every candidate value has explicit feature validity; visibility is one-hot |
| Candidate verdicts lacked global context | Candidate encodings receive shared masked state and relation context |
| Convolution over ordered map-token lists | Pointwise context K/V makes context attention permutation-invariant |
| One attention policy for both stages | Stage 3 and 4 policies, context, Q/K, heads, neighbors, and FFN ratios are independent |

Map schema is `imagination-semantic-map-v3`; model architecture is `imagination-imf-math-assisted-v7`; candidate features are `imagination-candidate-mask-contract-v2`; metric attention is `imf-stagewise-activequery-relational-v5`. Incompatible checkpoints fail explicitly.

## Development resource probe

Synthetic CPU measurements compare implementation cost only. They contain no trained-quality or held-out-test evidence.

| Configuration | Params (M) | MAC/sample (G) | Stage 3/4 pairs | p50 / p95 ms |
|---|---:|---:|---:|---:|
| Sparse/sparse, full QK, context both | 11.73 | 1.104 | 98,304 / 49,152 | 41.53 / 43.15 |
| Sparse/dense, context stage 4 | 11.67 | 1.099 | 49,152 / 73,728 | 34.92 / 35.98 |
| Previous + medium QK | 10.90 | 1.023 | 49,152 / 73,728 | 34.18 / 35.52 |
| Previous + MLP ratio 1 | 9.42 | 0.872 | 49,152 / 73,728 | 33.45 / 37.41 |
| Small QK and reduced heads | 10.64 | 0.997 | 32,768 / 49,152 | 33.08 / 34.17 |
| Active queries, full QK | 11.67 | 1.024 | 24,576 / 43,776 | 33.25 / 34.49 |
| Dense/dense, context both | 11.73 | 1.151 | 442,368 / 73,728 | 31.50 / 32.83 |

Dense/dense was fastest on this development CPU despite more score pairs. Sparse pair reduction is therefore not a latency claim. Preserve the reference and compare dense/dense and stage-3-sparse/stage-4-dense medium-QK under validation quality before selecting an architecture.

## Validation

- Frozen source: 250 episodes and 20,000 frames.
- Grouped manifests: 14,000 train, 3,040 validation, 2,960 locked test.
- Python: 55 tests and 4 subtests pass.
- C++: 6/6 CTest tests pass.
- All four models forward/backward and complete the synthetic smoke path.
- ONNX checker passes for the fixed-shape IMF smoke export; ONNX Runtime parity remains unavailable locally.
- No final test metrics, full-cache generation, primary suite, or 80-epoch training was run during the architecture revision.

## IMF Learning-Boundary Auditor

`common/imf_audit/` now provides a design-time, human-reviewed sufficiency audit. It registers analytical responsibilities and dependencies, applies `R=max(error/ε, uncertainty/υ, failure/φ, cost/κ)`, supports bootstrap three-way decisions and progressive-window stability, rejects final-test architecture-selection input, and emits JSON/CSV/Markdown/MSK/figure artifacts. Default limits are deliberately unfrozen, so the 5/10/25/50-frame development audit remains `UNDECIDED` rather than inventing safety criteria. It neither mutates the model/cache nor starts training.

## Remaining scientific limits

Pose covariance is a configurable approximation, region boundaries are point-to-disc approximations, and extended-region visibility is center-point approximate. Richer IMF dissimilarity and active-query pruning are experimental and disabled by default. Target-device latency, memory, power, joules/frame, task-quality effects, broad-domain robustness, and novelty relative to all prior work remain unmeasured.

The repository is technically validated for development, but primary training should wait until the existing real-data cache, supervision-coverage, and overfit readiness checks pass.
