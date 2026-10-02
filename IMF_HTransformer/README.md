# IMF–HTransformer

The C++ IMF extractor supplies a fixed 28×32×32 channel-major tensor and validity mask. The learned input is split into appearance, motion, and geometry families; each family has a small mask-aware 1×1 plus depthwise 3×3 adapter. The old flat 56-channel adapter remains an ablation. Sensor state and predicted local-map relations remain separate masked inputs.

The governing rule is to calculate reliable visual, geometric, probabilistic, and physical relationships in IMF and leave residual semantic compatibility and contextual verdict formation to the learned path. The map is therefore causal metric memory rather than a raw-history neural memory, and deterministic candidate facts such as clearance, density, uncertainty, and drift are supplied rather than reconstructed by the verdict heads.

Spatial HTransformer attention uses learned Q/K and spatial MBConv-derived V. Persistent map tokens use independent pointwise context K/V projections because the entity list is unordered; no convolution is applied over candidate order. Per-candidate geometry is masked feature-by-feature. Stage 3/4 attention mode, context use, neighbors, Q/K width, heads, and FFN ratio are configurable. Active-query selection and richer IMF relational dissimilarity are experimental and disabled by default.

The C++ library source is in `imf/cpp/`. Root `imagination.hpp` forwards to its public API. Generate the cache with `python run.py --preextract`; train explicitly with `python run.py --train imf_htransformer`.

Candidate geometry separates no-fly/restricted regions from no-land-only obstacles. Clearance uses candidate-centered point-to-disc approximations, with explicit feature validity. Development attention-policy timings and limits are recorded in [`docs/benchmarking.md`](../docs/benchmarking.md); pair reduction is not treated as a latency claim.
