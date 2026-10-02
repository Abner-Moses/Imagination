# Learned perception model contract

## IMF input

`IMF_HTransformer` receives a channel-major float32 tensor `B×28×32×32`, a same-shaped binary validity tensor, the 13 separately masked sensor values, 30 separately masked relational values, and up to 32 candidate-map tokens. Candidate tokens have a whole-token validity mask and a separate per-feature validity tensor `[B,32,40]`. The fixed analytical channel registry is documented in [imf.md](imf.md). Sensor state is not appended to the spatial tensor.

The IMF front end divides the 28 channels into three fusion families, without changing the finer channel-family registry used for ablation:

| Fusion family | Channels | Research width |
|---|---|---:|
| Appearance / visual structure | Y, Cb, Cr, Gx, Gy, magnitude, HOG 0–8, Harris, Canny, contours, Cb/Cr gradients (1–20) | 96 |
| Motion | OpticalFlowU/V (21–22) | 32 |
| Geometry | Depth, depth gradients, slope, roughness, GeometryConfidence (23–28) | 64 |

For family `f`, the adapter input is `concat(F_f × V_f, V_f)`. Each adapter applies a 1×1 channel projection, normalization/SiLU, depthwise 3×3 convolution, and normalization/SiLU. Family widths are profile-configurable and must sum to the backbone embedding width. The old flat 56→embedding adapter remains selectable as the `flat` fusion reference. Neither path computes per-frame min/max normalization; the C++ channel values already follow fixed physical or bounded normalization rules.

Missing observations have a zero feature and a zero validity value. Measured zeros, such as no Canny edge or zero flow with valid temporal support, retain validity one. Geometry validity is independent from geometry confidence; `GeometryConfidence` is a bounded support score, not a calibrated probability.

## State and relational inputs

The state order is acceleration `(x,y,z)` m/s², angular velocity `(x,y,z)` rad/s, attitude `(roll,pitch,yaw)` rad, magnetometer `(x,y,z)` µT, and ultrasonic sensor-axis range m. This is 13 values with a separate validity vector. The simulator's vertical ground clearance is a distinct geometry-scale oracle; it is not renamed as ultrasonic altitude.

The 30-value relational vector and 40-value candidate schema are versioned in `common/registry.py`. Relations are produced from the prior predicted semantic map. Map XYZ is kept separately in a typed field for projection; the candidate token's first three values are yaw-aligned egocentric XYZ (+x forward, +y left, +z up). Each candidate gets its own track/restricted clearance, nearest-obstacle clearance, free radius, named obstacle/cone density, and map-frame drifted clearance/landing margin. A separate 40-value validity vector marks which candidate measurements exist. Visibility is represented by six one-hot states rather than an ordinal scalar. Candidate free radius uses candidate-centered point-to-disc clearances to both obstacles and restricted zones; no-land-only objects remain in the obstacle category. The current relational frame is 2.5-D: yaw canonicalizes the horizontal plane while roll/pitch do not rotate the terrain frame. If yaw is unavailable, orientation validity is zero and map-aligned values are used.

## Outputs

- `hazard_logits`: `[B,1,32,32]`; sigmoid is the probability of unsafe traversal.
- `landing_logits`: `[B,1,32,32]`; sigmoid is the probability of a suitable landing cell.
- `semantic_logits`: `[B,6,32,32]`; background, grass, track, cone, rock, other obstacle.
- `poi.class_logits`: `[B,K,32,32]`; configured detectable-class heatmaps, default K=7.
- `scene_risk_logits`: `[B,3]`; safe, caution, dangerous scene class.
- `candidate.risk_logits`, `candidate.landing_safe_logits`, and validity for each map token.

All outputs are logits during training. Region extraction and local-map registration are deterministic operations outside the neural model. The model does not predict waypoints, motor commands, absolute map coordinates, or scientific value.

## Backbone and attention

The IMF backbone uses a family-aware embedding and a CoHAtNet-inspired MBConv-Value HTransformer. Spatial Q and K are independent learned projections. Spatial V is tokenized from `MBConv(F)`; no conventional `Linear(V)` follows it. Map tokens instead use independent pointwise context K/V encoders because they are an unordered set. Stage-3/stage-4 attention mode, map-context use, neighbors, Q/K width, head count, and MLP ratio are independently configurable. Gathered sparse attention creates `B×heads×Q×K` scores rather than `B×heads×N×N`; theoretical pair reduction is not evidence of lower wall-clock latency. Optional active-query attention keeps the dense feature grid but skips global attention for deterministically inactive queries. Optional normalized IMF relational dissimilarity and active queries are off by default. Map context contributes at most 32 additional keys/values and is admitted to current-view attention only when projection and visibility are valid.

Attention ablations select relative-position bias, IMF metric bias, both, or neither, and dense or gathered sparse attention. Dense relative attention is the reference. The metric term is a normalized geometric position distance over supported points, not a claim that semantic dissimilarities form a metric. Q/K normalization and topology smoothness loss are experimental and disabled by default.

## Causal map use

Training and validation roll each episode forward in frame order using predictions from earlier frames. At frame `t`, context comes from predictions through `t−1`; only after inference are frame-`t` predictions added. Simulator labels/depth are separate supervision or diagnostics and never enter the predicted-map inference state. Candidate projection always uses the separate map-frame XYZ, calibration, camera pose, and current IMF geometry. If geometry is unsupported, an observation remains unresolved rather than receiving invented XYZ.

The map's covariance is a first-order approximation from pixel/depth noise and a configurable support-derived pose model. The pose covariance is explicitly heuristic because the current C++ pose API does not expose a calibrated covariance. Association uses spatial pre-gating, shared semantic-posterior mass, and Mahalanobis distance when both covariances are valid; otherwise the map records a Euclidean fallback. The configurable posterior-overlap threshold allows soft contradictory observations to revise an entity belief while keeping nearly disjoint class hypotheses separate. Semantic class evidence uses bounded, correlation-discounted Dirichlet accumulation; this is an evidential fusion approximation, not a calibrated posterior guarantee.
