# Imagination Math Framework (IMF)

IMF is deterministic image processing, motion, calibrated geometry, sensor interpretation, persistent local mapping, uncertainty handling and relational calculation. It is not a neural network. The C++ image/reconstruction implementation is under `IMF_HTransformer/imf/cpp/`; Python's `common/mapping/` handles local-map evidence and spatial relations. IMF now means more than the 28-channel image bank: it also supplies known geometry, visibility, map evidence, topology and token relevance before learned contextual reasoning.

Its design rule is operational: if a relationship can be calculated reliably from permitted observations, calibration, geometry, physics, or probability, IMF supplies it with explicit validity and provenance. It does not encode a hand-written final hazard or landing classifier. Learned components remain responsible for semantic interpretation and context-dependent composition of the calculated facts.

The primary experiment freezes this boundary as a human-designed pre-LBA baseline.
A later Learning-Boundary Auditor experiment may measure the boundary, but it is
not invoked by readiness, training, or frame-by-frame inference.

## Fixed spatial channel registry

The channel-major float32 tensor and binary validity bank are each `28×32×32`. The registry order is unchanged. The canonical machine-readable spec is `data/configs/feature_spec.json`; its source is `common/feature_spec.py`, and the C++ scaling/clipping is in `vision.cpp::normalizeChannel`.

| Index | Channel(s) | Output domain / interpretation |
|---:|---|---|
| 1–3 | Y, Cb, Cr | Fixed color-range planes in `[0,1]`; no per-frame scaling |
| 4–5 | Gx, Gy | Signed 3×3 Sobel response, fixed factor 2 and clip to `[-1,1]` |
| 6 | GradientMagnitude | Reused Gx/Gy magnitude, fixed √2 scale and clip to `[0,1]` |
| 7–15 | HOG_0…HOG_8 | Unsigned 9-bin orientation energy, locally normalized per cell |
| 16 | HarrisResponse | Continuous fixed-scale signed square-root structure-tensor response, clipped to `[-1,1]` |
| 17–18 | CannyEdge, ContourMap | Binary `[0,1]` maps; a valid zero means measured absence |
| 19–20 | ChromaGradientCb/Cr | Fixed √2-scaled chroma derivative magnitudes in `[0,1]` |
| 21–22 | OpticalFlowU/V | Previous-to-current image displacement divided by configured pixel scale and clipped to `[-1,1]` |
| 23 | Depth | Current-view geometry in metres divided by configured depth scale and clipped to `[0,1]` |
| 24–25 | DepthGradientX/Y | Sobel depth change × feature-grid focal length ÷ depth, then configured fixed scale and clip to `[-1,1]`; dimensionless relative depth gradient |
| 26 | Slope | Surface angle in radians multiplied by `2/π` and clipped to `[0,1]` |
| 27 | Roughness | Surface roughness in metres divided by configured scale and clipped to `[0,1]` |
| 28 | GeometryConfidence | Bounded support score `[0,1]`; not a calibrated probability |

The complete numerical contract is verified in `feature_spec.py::validate_feature_batch`. Per-frame min/max normalization is prohibited. Geometry masks indicate supported reconstructed measurements; a confidence value of zero and a validity value of one can mean extraction completed but support quality was zero. Missing flow or geometry uses validity zero, so it is distinct from a measured zero.

## Family-aware neural input

The separate fusion-family registry in `feature_spec.py` groups channels 1–20 as appearance/visual structure, 21–22 as motion, and 23–28 as geometry. It does not replace the finer ablation families in `IMF_HTransformer/model.py`. For each family the adapter sees `[F×V, V]`, then applies a 1×1 projection and depthwise 3×3 local refinement with normalization/activation. Widths are configuration/profile controlled and sum to the embedding. The legacy flat 56-channel adapter remains available for direct comparison. State remains a separate 13-value masked vector.

Normalization is intentionally feature-specific at extraction time: bounded colors/edges/confidence remain bounded; signed derivatives and flow remain signed; depth and roughness retain configured physical scales; HOG is locally normalized. The neural adapter does not standardize heterogeneous values together or inspect validation/test statistics.

## Shared computation and temporal systems

Luminance conversion, Gaussian smoothing and Sobel Gx/Gy are shared by magnitude, HOG, Harris and Canny-related processing where applicable; Canny boundary extraction reuses the existing gradients and contour rasterization reuses the edge result. Chroma derivatives are computed on Cb/Cr because their derivatives are not luminance derivatives. Dense Farnebäck produces current-grid U/V features from consecutive camera frames; sparse pyramidal Lucas–Kanade tracks maintain correspondences for relative geometry. There is no optical-flow sensor.

Some channels intentionally overlap: derivatives feed magnitude/HOG/Harris, and Canny/contours both represent boundaries. This is a theoretical redundancy concern, not evidence of waste or harm. Family and fine-grained representation ablations retain the channels; true compute ablation must also disable the corresponding C++ stage and report its measured cost.

The persistent semantic map is the default temporal evidence mechanism. It stores bounded geometry, covariance, fused semantic evidence, support, age, and extent rather than retaining a long sequence of raw image tokens. This implements the hypothesis that probabilistic map memory can replace learned temporal accumulation for these facts. It does not claim that every possible dynamic phenomenon can be represented without a temporal model.

## Metric positions and attention topology

When valid calibrated current-view depth exists, `common/models/metric_attention.py::positions_from_analytical` back-projects feature cells into the OpenCV camera frame (`+x` right, `+y` down, `+z` forward). Intrinsics are scaled to the 32×32 feature grid. Depth is the IMF channel multiplied by its configured depth range; target-only simulator depth is never used. Unsupported positions contribute neutral metric bias and sparse selection falls back to a deterministic grid neighborhood.

The primary IMF configuration uses camera-frame Euclidean position distance normalized by `attention.scale_m` as a physical topology prior. It is a true Euclidean metric only for supported XYZ points; the full relation with unsupported points is a partial metric/dissimilarity. Learned Q/K remain responsible for semantic relationships. The configuration can select standard relative bias, metric bias, both, or neither. Metric and relative priors are not stacked implicitly in the primary setting.

The sparse implementation bounds a grid candidate pool, selects metric-near supported neighbors, preserves active query/global anchors, then gathers only K keys/values. The score tensor is `B×heads×Q×K` (Q=N when active queries are disabled); candidate-map cross-attention is separately bounded by 32 tokens. Pair-count reduction is theoretical until wall-clock timing confirms the gather path is faster on the target. The dense path remains an explicit reference. No out-of-distribution guarantee follows from metric bias.

Stage-3 and stage-4 attention mode, neighbor count/selection, and map-context enablement are independent. The optional `imf_dissimilarity_experimental.yaml` adds normalized optical-flow, slope, and roughness differences for selected query/key pairs only. Channel scales are dimensionless in the already-normalized feature bank: 0.25 flow corresponds to 4 px/frame under the default 16 px flow scale; 0.25 slope corresponds to 22.5°; 0.25 roughness corresponds to 0.025 m under the default 0.1 m scale. Nonnegative component weights are configurable. GeometryConfidence and valid flow support attenuate prior strength; missing descriptors add a neutral term. Because validity-dependent component sets vary by pair, this is called an IMF relational dissimilarity, not a guaranteed metric. It is disabled by default.

Optional active-query mode derives a deterministic importance map from gradient/edge strength, low geometry support, motion discontinuity, and visible projected safety/mission map tokens. It selects global-attention queries while retaining every spatial feature for local MBConv, residual processing, and the decoder. Mandatory evidence and one maximum-importance query per coverage bin can exceed the configured fraction. It does not use labels. The experimental configuration is disabled by default and must be measured for actual score and latency savings as well as trained quality.

## Numerical and scientific limits

The channel bank contains fixed normalized quantities but their values are not all directly comparable as physical units; each channel's units and validity semantics are listed in the JSON spec. `GeometryConfidence` is a support measure, not calibrated uncertainty. Metric attention uses measured geometry validity rather than treating that score as a probability. The semantic map's position covariance is a first-order approximation from pixel/depth noise plus a configurable support-derived pose covariance; the pose part is heuristic because current C++ pose diagnostics do not expose a calibrated covariance.

The C++ tests under `tests/cpp/` cover synthetic image, motion and geometry behavior. Python tests cover channel ranges/masks, map projection, covariance, visibility, attention and causality. `python run.py --validate` runs the supported project checks. Reproducible figures can be generated through `python run.py --figures`.

Analytical correctness and task utility remain distinct. Operator diagnostics can
test whether a calculation meets its measurement envelope; only model ablations can
establish whether its output benefits the UAV perception tasks.
