# Architecture

This document is the canonical overview of the frozen pre-LBA system. Detailed
numerical definitions live in [model_contract.md](model_contract.md),
[imf.md](imf.md), and [semantic_map.md](semantic_map.md).

## Research boundary

Imagination assigns operations according to one question: why is this learned?
Reliable image processing, geometry, probability, physics, coordinate transforms,
and deterministic map operations belong in IMF. Convolution handles local learned
structure. The HTransformer handles residual semantic compatibility and contextual
composition.

| Responsibility | Owner |
|---|---|
| Color, gradients, orientation, corners, edges, contours | IMF |
| Optical motion and metric surface support | IMF |
| Measurement validity and approximate position uncertainty | IMF |
| Causal persistent memory and semantic evidence fusion | IMF |
| Map association, visibility, coordinates, distance, density, clearance, drift | IMF |
| Local feature combinations | Convolution and MBConv |
| Residual long-range semantic compatibility | HTransformer Q/K |
| Scene and candidate interpretation | Shared learned heads |

This allocation reduces what the network must rediscover. Training must still
determine whether it improves the accuracy/resource trade-off.

## End-to-end flow

    frame t RGB + vehicle sensors
                    |
          +---------+----------+
          |                    |
       RGB path          IMF extraction
          |          appearance / motion / geometry
          |                    |
          +----------+---------+
                     |
              local learned encoder
                     |
       predicted map through frame t-1
          |          |
          |     deterministic relations,
          |     candidates, visibility,
          |     egocentric coordinates
          |          |
          +----------+
                     |
          convolution-assisted HTransformer
                     |
       hazard / landing / semantic / POI maps
             scene and candidate verdicts
                     |
        frame t predictions update the map

The update order is a scientific invariant. Current or future simulator labels
never enter the predicted map or current-frame context.

## Observations

The RGB models receive a 256 by 256 RGB tensor. IMF-HTransformer receives a fixed
28 by 32 by 32 analytical tensor and a same-shaped validity tensor. The channel
cache remains stable and is divided only inside the model:

- appearance and structure: channels 1–20;
- motion: channels 21–22;
- geometry: channels 23–28.

Each IMF family concatenates masked values with validity, applies a one-by-one
projection, then a depthwise three-by-three refinement. Research widths are
96, 32, and 64, summing to the 192-channel backbone embedding. Small and tiny
profiles use smaller configured widths. Flat fusion remains an ablation.

The 13 vehicle-state values never enter the spatial IMF tensor. State, 30
deterministic relations, and up to 32 map candidates retain separate masks.
Candidate values also have per-feature validity, so a measured zero differs from an
unavailable value.

## Four model families

The model registry in common/models/registry.py is authoritative.

| Model | Front end | Context mechanism |
|---|---|---|
| CNN | learned RGB convolution | local convolution |
| CNN-ViT | learned RGB convolution | conventional transformer |
| CNN-HTransformer (CoHAtNet-inspired) | learned RGB convolution | MBConv-value HTransformer |
| Imagination IMF-HTransformer | family-aware analytical fusion | IMF-assisted MBConv-value HTransformer |

Shared components include output heads, state/relation/candidate conditioning,
losses, metrics, evaluation, checkpoints, and training. The model differences are
intentional experimental variables, not separate training frameworks.

## HTransformer

For a spatial feature map F, Q and K are independent learned projections. The
spatial value tensor is:

[
V = \operatorname{Tokenize}(\operatorname{MBConv}(F)).
]

There is no conventional linear projection of spatial V before attention. This
MBConv-value rule applies to the regular image grid. Map candidates form an
unordered set, so their keys and values use independent pointwise encoders. No
convolution is applied across candidate array order.

Stage 3 operates on 16 by 16 tokens and stage 4 on 8 by 8 tokens. Their attention
mode, neighbor count, map-context use, Q/K dimension, heads, and FFN ratios are
configured separately. The primary architecture uses one block in each stage.
Alternative widths and attention policies remain explicit ablations.

Gathered sparse attention creates scores for selected query-neighbor pairs rather
than building the dense spatial score matrix. The default prior uses supported
camera-frame geometry distance. Optional flow, slope, and roughness dissimilarity
and active-query selection are experimental and disabled in the primary path.
Reduced pair count is reported separately from MACs and measured latency.

## Persistent semantic map

The map stores resolved entities in a stable local metric frame. An entity carries:

- absolute map-frame XYZ and optional 3 by 3 position covariance;
- semantic evidence, posterior probabilities, entropy, and support;
- extent, first/last observation times, source observations, and attributes.

Pixel/depth uncertainty is propagated with a first-order projection Jacobian.
Camera-pose covariance is a documented support-derived approximation because the
current estimator does not expose calibrated covariance.

Association first uses a voxel pre-gate and semantic compatibility. When both
covariances are valid, the final gate uses a stabilized Mahalanobis solve.
Euclidean fallback is recorded only when covariance is unavailable. Repeated
semantics use bounded, correlation-discounted Dirichlet-style evidence rather than
raw arithmetic averaging.

The map remains useful when an entity is outside the current camera. Visibility
gates current-view token attention; it does not erase persistent obstacles or
remove them from deterministic safety geometry.

## Relations and candidates

The map remains in its stable frame. Learned positions and horizontal vectors are
converted into a yaw-aligned ego frame with positive x forward and positive y left.
Point transforms include translation; vector transforms apply rotation only.

IMF computes candidate-centered obstacle and restricted-region clearance, free
radius, cone and obstacle density, map-frame drifted clearances, and landing margin.
Extended regions currently use point-to-disc distance from stored centers and
extents. Merged regions preserve absolute map coordinates, while the candidate
tensor carries egocentric coordinates.

Visibility is represented by independent categorical indicators for unknown,
visible, behind-camera, outside-FOV, occluded, and depth-inconsistent states. It is
not encoded as an ordinal scalar.

## Outputs and loss

All models return the same output names:

- hazard_logits and landing_logits;
- semantic_logits;
- poi.class_logits;
- scene_risk_logits;
- candidate.risk_logits and candidate.landing_safe_logits.

Spatial outputs are 32 by 32. Candidate verdicts combine mask-aware candidate
features with shared state and relation embeddings. The heads interpret supplied
geometry; they do not reconstruct clearance or drift arithmetic.

Targets and masking are defined in [model_contract.md](model_contract.md).
Training uses the shared multitask loss and validation metrics.

## Causal training path

For each episode:

1. Build frame t context from the model's predictions through frame t-1.
2. Calculate current deterministic relations and projected candidates.
3. Run frame t inference.
4. Apply current prediction thresholds and geometric support.
5. Update the predicted map.

Candidate target depth is read only when constructing supervision validity after
context creation. It never becomes a model or map input.

## Serialized contracts

common/registry.py owns the analytical channel, candidate, map, model, checkpoint,
manifest, statistics, readiness, and training-plan versions. Incompatible
checkpoints fail before weights are loaded. Style-only refactors do not bump these
versions.

## Current limitations

Pose covariance is approximate. Region distance uses disc approximations. Extended
entity visibility is center-point based. Deployment depth is reconstructed rather
than sensed. The dataset is synthetic. Resource measurements are development-host
results, and target-device energy has not been measured. These limits constrain the
claims that can be made from the primary experiment.
