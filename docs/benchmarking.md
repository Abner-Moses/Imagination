# Benchmarking

Keep the three cost boundaries separate.

1. **IMF extraction:** C++ per-stage and total latency, output tensor size and measurable process memory.
2. **Learned model:** parameters, serialized state size, estimated MACs, batch-one latency percentiles and allocator/process memory.
3. **Complete pipeline:** extraction or RGB preparation, streaming map update, network and post-processing.

The primary resource question is evaluated on the third boundary. A smaller learned model is not a system-level saving unless `IMF extraction + map + relations + learned model + post-processing` compares favorably with the equivalent complete baseline. Pair counts and neural MACs are explanatory measurements, not substitutes for end-to-end latency, memory, and measured target energy.

For the IMF revision, further split the map cost into covariance propagation, spatial pre-gating, Mahalanobis solve/fallback, semantic-evidence update, visibility, relation calculation, token merging/selection, and candidate projection. The HTransformer report includes spatial token count, configured K, bounded map-context count, actual attention score-pair count, dense-reference pair count, and measured block/model latency. Pair reduction is not itself a runtime saving. Metric-neighbor selection also has measurable cost and is included in attention timing.

Compare `flat_1x1_fusion`, family fusion, standard relative attention, metric attention, and gathered sparse metric attention with the same split/model budget. Mapping ablations compare Bayesian versus arithmetic semantic evidence, Mahalanobis versus Euclidean association, uncertainty on/off, visibility on/off, and egocentric versus map-aligned candidate context. Report duplicate/incorrect association and unresolved projection rates where ground truth supports them. Quality/latency effects remain `NOT_MEASURED` until the respective runs complete.

The gather implementation creates only `B×heads×Q×K` spatial scores and bounded `B×heads×Q×M` map-context scores (`M≤32`, Q=N unless active queries are enabled). Metric neighbor selection compares each active query with a bounded candidate pool P (configured from K and capped at N), which costs `O(BQP)` distance work. It does not materialize dense attention scores. At small grids P may equal N; report both the theoretical pair count and measured latency. The stage attention timer includes score gathering and the joint softmax/output path; `sparse_gather_ms` and `context_attention_ms` are nested component diagnostics, not additive partitions of that timer.

`python run.py --benchmark` benchmarks one instance of each family for the selected profile. `python -m tools.benchmark --preextraction-csv <file> --output <json>` summarizes C++ timing output. An optional target power log has columns `timestamp_s,power_w`; trapezoidal integration reports average watts and joules/frame. Missing hardware readings stay null/`NOT_MEASURED`; machine TDP is not an energy measurement.

Use a warmup and fixed image/profile/device, report mean/median/p95/std/sample count, and note whether data loading and file I/O are included. MAC estimates count Conv/Linear layers and explicit attention products; opaque OpenCV work is runtime-only and must not be represented as exact FLOPs.

Training logs epoch loss components, validation FNR/IoU/recall, batch throughput, epoch time and moving ETA. `artifacts/progress.json` is the resumable progress record. Development Mac results do not establish Uno Q latency or energy. The edge target needs a measured end-to-end batch-one run, process RSS, model/runtime memory and externally measured power.

## Current status

The Python evaluator now exercises causal, per-episode predicted-map context and mapping post-processing. A complete raw-frame C++ extraction + neural inference + map harness has not been timed across the full dataset, so end-to-end latency remains `NOT_MEASURED`. CPU peak memory is process high-water RSS where the platform exposes it; CUDA/MPS memory comes from the backend allocator and is not directly comparable to C++ process RSS. The current recorded model profiles are CPU-only development-machine measurements, not target-hardware evidence.

## IMF mathematical-assistance revision smoke benchmark

The 2026-10-01 CPU smoke benchmark is in `artifacts/benchmarks/models_research.json`. It ran batch-one inference on the development Mac. The IMF_HTransformer case includes 32 valid synthetic map-context tokens; these numbers are implementation diagnostics, not a controlled accuracy or end-to-end comparison. RGB models do not use the same map-token cross-attention path in this probe.

| Model | Parameters | MAC/sample | p50 / p95 latency (ms) | Attention score pairs |
|---|---:|---:|---:|---:|
| CNN | 6,447,214 | 796,805,376 | 35.87 / 37.07 | n/a |
| CNN_ViT | 7,899,310 | 1,151,289,600 | 37.04 / 38.58 | 442,368 |
| CNN_HTransformer | 11,628,160 | 1,381,910,784 | 38.64 / 39.37 | 442,368 |
| IMF_HTransformer | 11,560,950 | 1,218,821,888 | 81.58 / 84.17 | 147,456 vs 516,096 dense reference |

The IMF configuration used K=32 gathered image neighbors and 32 map-context tokens at each HTransformer stage. Across both stages, the reported image/map attention score count is 71.43% below its dense reference. Metric neighbor selection made 36,864 bounded distance comparisons in this 16×16 and 8×8 stage configuration. On this CPU, the complete IMF model probe is about twice as slow as CNN_HTransformer despite fewer score pairs; gather/indexing, metric selection, and map context dominate enough that pair reduction does not imply lower latency. Quality effects are `NOT_MEASURED` because no training or test evaluation was run. Development energy is `NOT_MEASURED`.

These updated counts supersede the earlier pre-revision model table for the IMF architecture. The C++ pre-extraction sample remains the separate existing 60-frame synthetic timing record (7.89 ms mean); this revision did not alter the C++ extractor. Synthetic microbench timings for covariance propagation, visibility, map update and relation/token construction appear below; they are not full-sequence stage timings. End-to-end frame latency remains unmeasured. Do not sum the C++ timing and model probe as an end-to-end result.

`python -m tools.benchmark_imf_revision` records a deterministic synthetic microbenchmark at `artifacts/benchmarks/imf_revision_microbench.json`. With 80 known objects, eight observations/object and one warmup repetition, Euclidean association produced 40 mixed map entities (all 80 truth objects involved in a merge); Mahalanobis association produced 80 entities and no mixed entities. This specific close-object fixture is not a real-scene quality estimate. After adding the posterior-overlap compatibility gate, development-Mac Mahalanobis+Bayesian update p50/p95 was about 0.366/0.380 ms per observation versus 0.210/0.220 ms for Euclidean+arithmetic. Covariance propagation was about 0.126/0.131 ms per point; current visibility classification 0.0106/0.0120 ms per entity; relation plus token construction 3.254/3.362 ms per call for this 80-entity map. The relation benchmark uses vectorized density calculation and a trace-derived cached uncertainty summary. These are CPU microbenchmarks of synthetic state, with no accuracy inference beyond the constructed fixture; no full sequence or target-device map cost is established.

## Stage-specific attention development matrix

`python -m tools.benchmark_attention_matrix --device cpu --warmup 5 --repetitions 20` writes `artifacts/benchmarks/imf_attention_matrix.json`. It uses synthetic tensors and no dataset, validation split or test split. Each configuration runs in its own process; process high-water RSS includes Python/PyTorch and model allocations, while an incremental RSS delta was unavailable. Component times below are from a separately instrumented forward and therefore should not be added as if they partitioned the timed total. This matrix measures resource behavior only; quality is `NOT_MEASURED`.

| Configuration | Params (M) | MAC/sample (G) | Active Q3/Q4 | Stage 3/4 score elements | Stage 3/4 attention ms | Query selection ms | Neighbor selection ms | Sparse gather ms (3/4) | Context projection + attention ms | Total p50 / p95 ms | Process peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse/sparse, full QK, context both | 11.73 | 1.104 | 256 / 64 | 98,304 / 49,152 | 4.978 / 2.559 | 0.000 | 1.779 | 1.282 / 0.616 | 1.914 | 41.53 / 43.15 | 414.9 |
| A: sparse/dense, context both | 11.73 | 1.107 | 256 / 64 | 98,304 / 73,728 | 4.839 / 0.437 | 0.000 | 1.457 | 1.058 / 0.000 | 1.477 | 37.43 / 39.19 | 396.6 |
| B: sparse/dense, context stage 4 | 11.67 | 1.099 | 256 / 64 | 49,152 / 73,728 | 2.537 / 0.478 | 0.000 | 1.455 | 0.973 / 0.000 | 0.247 | 34.92 / 35.98 | 347.9 |
| C: B + medium QK | 10.90 | 1.023 | 256 / 64 | 49,152 / 73,728 | 2.152 / 0.483 | 0.000 | 1.292 | 0.713 / 0.000 | 0.249 | 34.18 / 35.52 | 342.7 |
| D: C + MLP ratio 1 | 9.42 | 0.872 | 256 / 64 | 49,152 / 73,728 | 2.108 / 0.495 | 0.000 | 1.484 | 0.820 / 0.000 | 0.255 | 33.45 / 37.41 | 329.6 |
| E: smaller QK + reduced heads | 10.64 | 0.997 | 256 / 64 | 32,768 / 49,152 | 1.697 / 0.394 | 0.000 | 1.463 | 0.733 / 0.000 | 0.206 | 33.08 / 34.17 | 332.2 |
| F: active queries, full QK | 11.67 | 1.024 | 128 / 38 | 24,576 / 43,776 | 1.371 / 0.437 | 0.360 | 0.601 | 0.584 / 0.000 | 0.228 | 33.25 / 34.49 | 330.3 |
| Dense/dense, context both | 11.73 | 1.151 | 256 / 64 | 442,368 / 73,728 | 1.026 / 0.446 | 0.000 | 0.001 | 0.000 / 0.000 | 0.579 | 31.50 / 32.83 | 315.1 |
| Dense/sparse, context both | 11.73 | 1.148 | 256 / 64 | 442,368 / 49,152 | 1.033 / 2.225 | 0.000 | 0.340 | 0.000 / 0.518 | 0.982 | 34.17 / 35.51 | 357.5 |

Score elements include spatial self-attention and enabled map-context cross-attention. The JSON reports Q and K projection MAC separately per stage; with active queries, Q projection uses selected Q while K projection still covers all stage tokens. `attention_ms`, `sparse_gather_ms`, and `context_attention_ms` are nested measurements, not additive time partitions. This CPU run shows dense/dense is fastest despite the most score elements, while active-query mode is fastest among the tested sparse variants. Neither has trained quality evidence. Preserve sparse/sparse as the reference and treat dense/dense, medium QK and active queries as development candidates for quality-controlled ablations, not as a selected scientific winner or an edge-device claim. Pair reduction is not measured latency savings.
