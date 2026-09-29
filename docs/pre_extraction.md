# Analytical pre-extraction

The primary research output is now `VisualFeatureFrame::feature_tensor`: a
contiguous float32 **C×32×32** image-aligned tensor, with identically shaped uint8
validity masks and ordered channel names. The full bank has **28 channels**.
`VisualFeatureExtractor` is the single stateful entry point. Its older visual
planes, LBP, texture orientation, sparse corners, FAST points and APIs remain
available as comparison features.

This stage stops at explicit analytical data. It does not run a learned adapter,
CoHAtNet, HTransformer, semantic classification, navigation or landing decisions.
Existing independent reconstruction/landing commands remain unchanged.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure

# Supplied still: meaningful spatial maps; flow and geometry explicitly unavailable.
./build/imagination pre_extract 00001.png output/analytical \
  --config configs/pre_extraction.yaml --data --debug

# Known synthetic camera sequence, synthetic altitude/attitude, NOT a hardware trial.
./build/imagination pre_extract ignored output/analytical_synthetic \
  --synthetic --frames 80 --data --debug

# Ablation: channels omitted and their unnecessary dependencies skipped.
./build/imagination pre_extract 00001.png output/ablated \
  --without hog,roughness,contours --no-geometry --debug
```

Diagnostic JPEG: `output/analytical/analytical_features.jpg`. Magenta means invalid,
not zero. Each panel uses a fixed range matching the tensor normalization. The
synthetic sequence has a separate JPEG under `output/analytical_synthetic/`.
Root images have no verified temporal timing/calibration: they are not silently
presented as real sequential camera motion.

`--data` writes `frame_N.yml.gz` containing the tensor, masks, names, calibration,
normalization scales, diagnostics, relative pose and stage times. `--debug` writes
montages every ten frames and a final montage; a supported local map also produces
`local_map.ply`, `dem.csv` and terrain previews. No GUI is required. `timings.csv`
and `run.txt` are always written. Export/render/capture time is outside extraction
timing. The summary includes initialization and all supplied frames; use CSV rows
to select a declared warmup interval for comparisons.

### Real sequences and cameras

```sh
./build/imagination pre_extract sequence.yaml output/sequence \
  --sequence --calibration camera.yaml --frames 500 --data --debug

cmake -S . -B build -DBUILD_MOTION_CAMERA=ON
cmake --build build -j2
./build/imagination pre_extract 0 output/live --camera --frames 300 \
  --calibration camera.yaml --imu /tmp/imu.txt --ultrasonic /tmp/range.txt
./build/imagination pre_extract recording.mp4 output/video --video --frames 300
```

A sequence manifest uses explicit timestamps and optional per-frame sensor samples:

```yaml
%YAML:1.0
frames:
  - { image: "frame0.png", timestamp_s: 0.0, altitude_m: 2.0,
      range_timestamp_s: 0.0, imu: [0.0, 0.0, 0.0], imu_timestamp_s: 0.0 }
  - { image: "frame1.png", timestamp_s: 0.1, altitude_m: 2.0,
      range_timestamp_s: 0.1, imu: [0.0, 0.0, 0.01], imu_timestamp_s: 0.1 }
```

These numeric values illustrate the format; replace them with actual measurements.
Image paths are relative to the manifest. Omit unavailable readings. Timestamps
must increase and share a clock. Video timestamps use frame index / reported FPS
(or `--fps`, if the video has no valid FPS); camera timestamps are frame-delivery
monotonic seconds, not guaranteed exposure timestamps. Snapshot inputs are accepted
only in camera mode. The sensor acquisition driver/attitude filter is external;
see [sensor inputs](sensor_inputs.md).

Calibration YAML contains the existing `camera: {width,height,fx,fy,cx,cy}` block
and optionally a `distortion` sequence in OpenCV order (4, 5, 8, 12 or 14 values).
An omitted distortion sequence declares already rectified/negligibly distorted
input. A provided calibration must match the source image size. No calibration is
required for 2-D channels; malformed calibration is an input error, not silently
substituted guessed intrinsics.

## API and storage

```cpp
#include "imagination.hpp"
using namespace metric_mapping;
auto settings = analyticalFeatureSettings(); // analytical on; legacy planes off
settings.analytical.features.reset(std::size_t(AnalyticFeature::Hog));
VisualFeatureExtractor extractor(settings, camera); // camera is optional
const auto& frame = extractor.extract(bgr, timestamp_s, altitude, imu);
const auto& tensor = frame.feature_tensor;
cv::Mat channel = tensor.plane(0); // borrowed float32 H×W view
cv::Mat valid = tensor.mask(0);    // 255 = supported; 0 = unknown
```

The same extractor/result is used by `visual_features` and `pre_extract`. Legacy
construction defaults are preserved; `analyticalFeatureSettings()` selects the
new research profile. `settings.features` independently selects legacy comparison
features. Result storage is borrowed and may be overwritten on the next call.
Clone a tensor if it must outlive that call. Indexing is
`((channel * height) + row) * width + column`; optional `batch_dimension=true`
adds a leading size-one dimension without changing element order. Link
`Imagination::core`; no tensor framework is required.

Default working size is exactly 256×256, configurable, with an exact 32×32 output
grid. Non-square source frames are resized anisotropically; focal lengths and
principal points follow that same transformation. Small sources may be upscaled
in this research mode (this adds no information). Legacy sparse-only sizing still
preserves aspect ratio and never upscales. With scale `sx,sy`:
`fx'=sx*fx`, `fy'=sy*fy`, `cx'=sx*(cx+0.5)-0.5`, `cy'=sy*(cy+0.5)-0.5`.
Distortion remap tables are cached; rectification happens before both image and
geometry processing. HOG bins refer to the **working image**, including its resize.

## Channel contract and normalization

No channel uses framewise min-max normalization. All clipping is deterministic.
Masks distinguish a physical zero from an unavailable value. GeometryConfidence
is defined everywhere: unsupported cells deliberately contain a valid zero.

| Indices (one-based) | Channels | Stored normalization |
|---|---|---|
| 1–3 | Y, Cb, Cr | OpenCV float BGR→YCrCb, reordered; clipped [0,1] |
| 4–5 | Gx, Gy | Sobel(Y8)/(8×255), then ×2; signed [-1,1] |
| 6 | GradientMagnitude | sqrt(Gx_raw²+Gy_raw²) ×sqrt(2), [0,1] |
| 7–15 | HOG_0 … HOG_8 | Nine unsigned bins at 0,20,…160 degrees; per-cell L2 norm |
| 16 | HarrisResponse | `sign(R)*sqrt(abs(R))/harris_scale`, R=det(M)-.04 trace(M)², clipped [-1,1]; default scale .01 |
| 17–18 | CannyEdge, ContourMap | Binary occupancy [0,1] |
| 19–20 | ChromaGradientCb, ChromaGradientCr | Sobel/8 magnitude ×sqrt(2), [0,1] |
| 21–22 | OpticalFlowU, OpticalFlowV | Working-pixel displacement / `flow_scale_px` (16), clipped [-1,1] |
| 23 | Depth | Camera optical-axis depth in metres / `depth_scale_m` (20), [0,1] |
| 24–25 | DepthGradientX, DepthGradientY | Local image-plane m/m derivative / `depth_gradient_scale` (5), [-1,1] |
| 26 | Slope | World-surface slope radians / (π/2), [0,1] |
| 27 | Roughness | Residual standard deviation in metres / `roughness_scale_m` (.1), [0,1] |
| 28 | GeometryConfidence | Support score [0,1], not a calibrated probability |

Sobel raw derivatives use luminance smoothed with a 3×3 Gaussian, sigma 1.
HOG shares those derivatives and magnitude, weights votes by magnitude, interpolates
between neighboring orientation bins modulo π, and sums directly into output-grid
cells. Each histogram divides by `sqrt(sum(bin²)+1e-12)`. No flattened descriptor
or overlapping block-layout ambiguity is introduced. Harris shares box-averaged
`Gx²,GxGy,Gy²` with existing Shi–Tomasi/texture features when both are selected.
Canny consumes the same derivatives converted to signed 16-bit raw Sobel units;
thresholds default to 20/60, L2 magnitude. Contours rasterize Canny traces.
Continuous spatial planes use area pooling; binary edges use occupancy pooling.

## Temporal and 3-D branches

```text
BGR camera frame + timestamp + optional calibration / IMU attitude / altitude
  → calibrated resize / optional rectification
  ├→ luminance, shared Gaussian/Sobel → magnitude, HOG, Harris, Canny/contours
  ├→ Y/Cb/Cr → chroma gradients
  ├→ previous luminance → dense Farnebäck → current-aligned U/V
  └→ existing sparse LK → filtered correspondences
       ├→ essential RANSAC / recoverPose → relative pose + unit-baseline points
       └→ validated nadir + range scale → local metric poses → triangulateTrack
            → bounded colored map → TerrainGrid → bounded IDW
            → metric derivatives/slope + Perona–Malik/residual roughness
            → current-camera projection, supported triangles and z-buffer
  → selected channels + masks → C×32×32 float32 → STOP
```

Farnebäck defaults: pyramid scale .5, 2 levels, 15-pixel window, 2 iterations,
polynomial size 5, sigma 1.2. It provides scene displacement, **not UAV metric
velocity**. OpenCV forward flow is defined at previous-frame pixels. We scatter
it to nearest current-frame endpoints, average collisions, and leave holes invalid
before mask-aware pooling (at least half-cell support). This ensures tensor
registration but does not resolve occlusions or certify every correspondence.
First frame, reset and excessive frame gaps yield invalid zero-filled flow.

Sparse LK remains the existing persistent Shi–Tomasi tracker (100-point budget,
15×15 window, one extra pyramid level, 12 iterations). The analytical profile
turns forward-backward checking and metric triangulation on. Raw LK survivors are
exposed separately before the constrained similarity gate, for essential geometry.
Essential RANSAC/recoverPose runs every five frames with sufficient tracks, bounded
to 100 hypotheses. Its points undergo cheirality, parallax and reprojection checks.
Pure rotation, very small baseline and planar degeneracy may make this branch
unavailable. A recovered translation is a **unit direction**, never metres.
`RelativePose::scale_valid` remains false in this branch: no unvalidated scale
fusion or unscaled points are inserted into the metric map.

The metric map uses the existing constrained nadir estimator, not essential
translation length. It requires fresh heights at both image endpoints, consistent
image/height scale, accepted motion consensus, and IMU validity when supplied or
required. It reuses `triangulateTrack()` with known local metric poses. This
conservative path preserves the existing camera-right/world+X,
camera-down/world−Y, optical-axis/world−Z conventions. See
[optical-flow equations](optical_flow.md) and [coordinates](coordinates.md).
The local origin is the first camera center of the current continuity segment;
it is neither GPS/global position nor a globally optimized map. No general VIO,
loop closure or SLAM is implemented. Without valid scale, appearance/dense-flow and
possible relative-pose diagnostics continue, while metric channels stay invalid.

Metric triangulation is limited to four attempts every five frames, with track
age, minimum .05 m baseline, minimum 1° parallax, maximum 1 px reprojection error
and positive depth ≤100 m. New landmarks are colored from the current image.
The local map is capped at 256 points, 10 s age and 5 m three-dimensional distance
from the current camera; 5 cm voxels merge observations using the existing voxel
accumulator. The sparse tracker's 64-landmark pool supplies new entries. Segment
loss/reset clears all retained metric points. Metadata records track ID, age/time,
observation count, residual and support score. This bounded map can remain sparse.

## Terrain and current-image alignment

`createTerrainGrid()` retains the highest measured +Z in each 10 cm XY cell.
Maximum size is 4096 cells; an oversized extent suppresses geometry rather than
allocating unbounded memory. Existing IDW uses **measured cells only**, radius .3 m,
maximum interpolation distance .3 m, minimum three and maximum twelve neighbors,
weights `1/d²`. Unsupported gaps stay invalid. Interpolated cells are distinguished
from observations.

`measureSurface()` is nonsemantic. Shared Sobel derivatives respect metric spacing
and world-Y sign; slope is `atan(hypot(dz/dX,dz/dY))`. A complete 3×3 neighborhood
is required. The existing conservative landing slope remains unchanged in its
separate decision workflow. Roughness reuses the legacy residual-statistics and
Perona–Malik implementation: five four-neighbor iterations, step .2, conductivity
.05 m, then sample standard deviation of original minus smoothed elevations in a
5×5 window. Complete neighborhoods are required for analytical roughness.

Valid adjacent grid samples form triangles. `K*T_world_to_camera` projects them
into the current 32×32 camera grid; perspective-correct interpolation and a
nearest-depth z-buffer handle overlapping surfaces. Isolated points produce only
single supported samples. No arbitrary hole filling or resized bird's-eye DEM is
concatenated with RGB features. Confidence is reduced by reprojection error, pose
health, age, distance to measured support, and interpolated provenance.

Depth gradients are computed **after** projection with support-aware Sobel. A
pixel-space derivative is multiplied by `fx_grid/depth` or `fy_grid/depth` to
express a local image-plane metres-per-metre derivative. This perspective-local
quantity is not a world elevation gradient; slope is measured separately on the
metric DEM and projected alongside it. Occlusion boundaries/holes invalidate
unsupported derivative neighborhoods.

## Ablation, costs and validity

`features` / `--features` selects appearance, gradients, hog, harris, canny,
contours, chroma, flow, depth, depth_gradients, slope, roughness,
geometry_confidence. `--without` removes families. `enable_geometry: 0` /
`--no-geometry` omits all six geometry channels and skips reconstruction. With
geometry enabled but unavailable, its selected slots instead contain invalid
placeholders (confidence zero). Removing every geometric family also skips that
branch entirely. Removing roughness skips diffusion/statistics; removing
contours skips tracing if no legacy contour feature needs it. Dependency stages
may remain required by another selected family.

Every result exposes named timing entries with `ran` flags, including preparation,
color, Sobel, magnitude, HOG, Harris, Canny, contours, chroma, dense flow, sparse
tracking, pose, triangulation, map update, voxel work, grid, IDW, projection,
depth derivatives, terrain derivatives, slope, diffusion, roughness, resizing and
packing. Disabled/unavailable stages have `ran=false`. Geometry statistics include
tracks/inliers, new metric triangulations, map points, reprojection residual,
coverage and separate calibration/scale/geometry/flow validity. `payload_bytes`
counts tensor values/masks and retained point records, **not scratch, peak RSS or
OpenCV allocations**. The full tensor alone is 114,688 bytes plus 28,672 mask bytes.

The full candidate bank is intentionally more expensive than sparse-only motion.
Dense Farnebäck, remapping and dense descriptors should be ablated and benchmarked
on the actual UNO Q. No watts or end-to-end safety claims are inferred from CPU
latency. The research hypothesis concerns reducing learned extraction under known
constraints; it does not assert that classical vision is always preferable.

### Finished implementation benchmark

Release/OpenCV 5.0.0, one OpenCV thread, Apple M3 Pro development machine,
120 synthetic 256×256 frames, including initialization and without capture/export:

| Bank | Channels | Mean ms | Median ms | p95 ms | Worst ms |
|---|---:|---:|---:|---:|---:|
| Full 2-D + motion + metric geometry | 28 | 11.511 | 11.300 | 11.958 | 20.304 |
| Dense flow only | 2 | 5.626 | 5.663 | 5.707 | 6.150 |
| Twenty spatial channels, no flow/geometry | 20 | 4.972 | 4.965 | 5.045 | 5.754 |

The full run ended with 67 bounded local-map points and 41.50% current-view
geometry coverage. Stage means were: color 0.116 ms, Sobel 0.130,
magnitude 0.027, HOG 0.329, Harris 0.416, Canny 1.475, contours 1.930,
chroma 0.188, dense flow 5.526, sparse tracking 0.439, pose 0.134,
terrain/IDW 0.038, projection 0.021, resizing 0.328 and packing 0.226 ms.
Conditional geometry stages run only when support exists, so their
simple per-frame means include skipped frames. The command writes raw per-frame
data to `output/analytical_synthetic/timings.csv`.

These are synthetic development-machine latency measurements, not UNO Q latency,
energy, flight accuracy or evidence of model-level resource savings. Dense flow
is the largest measured stage and should be a primary ablation on the target.

## Limits and later integration

Metric reconstruction still assumes a near-nadir camera, mostly static locally
horizontal terrain, reliable synchronized calibration/range, and adequate parallax.
IMU input is estimated camera attitude, not raw MPU-6050 counts. Sparse support,
texture loss, rolling shutter, lighting changes and moving-majority scenes can
reduce validity or bias motion. Terrain is a single-valued elevation surface; it
cannot represent overhangs. Confidence is diagnostic evidence, not uncertainty
calibration. Sparse samples are not proof of a fully observed safe surface.

The tensor/mask/channel-name contract is the connection for a future learned
channel adapter and HTransformer. Future sensor fusion can replace/provide a
validated metric pose and registered geometry behind this contract. Add normals,
curvature and other sensor features with explicit units and masks. Training,
learned local/global attention, task heads, navigation, hazard outputs and
non-inferiority evaluation remain outside this stage.

Algorithm references: [OpenCV optical flow](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html),
[OpenCV calibrated pose and triangulation](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).

---

# Legacy comparison feature API

The following describes the preserved `visual_features` workflow. Its defaults,
normalizations and historical benchmark differ from the primary 28-channel profile.

# Camera analytic pre-extraction

The research question is: **To what extent can an analytically assisted,
convolution-attention perception architecture reduce onboard resource consumption
for GPS-denied autonomous UAV exploration while maintaining non-inferior
safety-critical perception and navigation performance relative to CNN-ViT systems?**

This module supplies camera-derived inputs for testing that hypothesis. It is
neither the complete mathematical/sensor framework nor a replacement for learned
perception. Future experiments can reduce learned convolutional feature extraction
while retaining learned local relationships and Transformer global/contextual
reasoning. No efficiency or safety equivalence to a learned baseline is established
by these algorithms or their synthetic tests.

## Pipeline and ownership

```text
RGB camera input (OpenCV BGR8 channel order)
       ↓
aspect-preserving reduction to at most 320 × 240; never upscale
       ↓
shared grayscale / Sobel derivatives / local structure tensor
       ├─ selected dense maps + validity masks
       ├─ bounded corners and descriptor-free FAST keypoints
       ├─ bounded contours and simplified shape boundaries
       └─ calibrated sequential input → existing sparse LK motion tracker
       ↓
VisualFeatureFrame: timestamp, coordinates, calibration, values, sparse records
       ↓
future synchronized mathematical/sensor framework
       ↓
future structured feature/token encoder
       ↓
future convolution-assisted local processing + global attention
       ↓
future navigation / exploration decisions
```

`metric_visual` is an alias of the common implementation library; temporal behavior
is included when `BUILD_MOTION` is enabled. It does not call stereo reconstruction or alter landing behavior.
`BUILD_VISUAL_FEATURES=OFF` removes this module. `BUILD_MOTION=OFF` leaves spatial
extraction available without temporal tracking. There are no new external dependencies.

## Programmatic use

Link your CMake target to `metric_visual` and include
`imagination.hpp`:

```cpp
using namespace metric_mapping;
VisualFeatureSettings settings;
settings.features = selectVisualFeatures(
    "gradients,edge_magnitude,corners,texture_orientation,motion");
settings.profiling = true; // false by default: no per-stage clocks
VisualFeatureExtractor extractor(settings, camera); // source-size calibration

// Call repeatedly with genuinely sequential, rectified BGR8 frames.
const auto& frame = extractor.extract(bgr, timestamp_s,
                                     AltitudeSample{height_m, range_timestamp_s});
for (const auto& plane : visualFeaturePlanes(frame)) {
    // plane.name, plane.units, plane.values, plane.valid
}
// frame.corners/keypoints/contours/tracks and frame.motion are typed records.
```

Omit calibration and altitude for spatial extraction. Motion then reports unavailable,
not a valid zero velocity. Calibration is not evidence of temporal provenance:
the caller must supply sequential frames and strictly increasing timestamps.
The motion subsystem checks fresh range samples and its geometric assumptions;
see [optical_flow.md](architecture.md) for scale, signs, yaw, and triangulation.
Call `resetMotion()` when starting a new sequence.

The extractor owns reusable buffers. Its result and shallow `cv::Mat` views are
borrowed until the next call; clone required maps before retaining or asynchronously
consuming them. Use one extractor per camera stream; concurrent calls are unsupported.
`visualFeaturePlanes()` creates an adapter vector only when requested. `requested`
and `computed` distinguish disabled features from requested-but-unavailable motion.
An empty point set can be a successfully computed result on an untextured image.

Coordinates have +u right, +v down in the working image. The affine map
`pixel_to_source` uses `u_src = sx*(u+0.5)-0.5` (likewise v). Intrinsics are scaled
with that same pixel-center convention. Outputs retain source/working dimensions,
frame ID and timestamp. Source-size calibration must match the input dimensions.

## Feature definitions

Intensity is grayscale divided by 255. Maps use float32 except binary masks,
LBP codes, and boundary rasters (uint8). Image processing does not require linear-light
RGB; gradients are in encoded camera intensity, so exposure/gamma changes affect them.

| Selection name | Reusable output and meaning |
| --- | --- |
| `gradients` | `gradient_x/y`: Sobel3 divided by 8×255, intensity per working pixel |
| `edge_magnitude` | `sqrt(gx²+gy²)` |
| `edge_orientation` | `atan2(gy,gx)` in [0,2π), gradient **normal**; mask excludes low magnitude and border |
| `corners` | Minimum structure-tensor eigenvalue map and grid-distributed Shi-Tomasi local maxima |
| `keypoints` | Grid-distributed FAST points; no descriptors or per-frame matching |
| `color_transitions` | Combined gradient norm of `(R-G)/255` and `(B-(R+G)/2)/255`; achromatic transitions cancel |
| `texture_orientation` | Dominant local **tangent** in [0,π), tensor coherence, validity mask |
| `local_binary_texture` | LBP8 radius 1; clockwise from NW, neighbor ≥ center sets bit; invalid one-pixel border |
| `contours` | Retained Canny edge traces in working pixels |
| `shape_boundaries` | Binary raster of closed, Douglas-Peucker-simplified retained traces |
| `motion` | Existing `MotionEstimate` and persistent `TrackedFeature` records |

For the 5×5 averaged tensor
`J = [[mean(gx²), mean(gx*gy)], [mean(gx*gy), mean(gy²)]]`, define
`d = sqrt((Jxx-Jyy)²+4Jxy²)` and `t = Jxx+Jyy`.
Corner response is `(t-d)/2`. Texture tangent is
`(0.5*atan2(2Jxy,Jxx-Jyy)+π/2) mod π`; coherence is `d/t` when energy is sufficient.
The orientation mask also rejects low coherence and the filtering border.
Undefined angles are zero-filled but **must be interpreted with their masks**.
LBP is categorical; numerical distance between codes is not a texture distance.

Corners and FAST points each default to at most 100, with a 5×4 grid and 6-pixel
spacing. These are alternative feature families for ablation, not necessarily
independent information. Motion keeps its own persistent corners and does not
depend on redetecting the spatial corner output each frame.

Canny reuses signed Sobel derivatives (thresholds 20/60 in raw Sobel units,
L2 magnitude). Retained traces default to at least 5 points, at most 64 contours
and 2,048 total points. Larger point-count traces are considered first; traces
that exceed the remaining budget are skipped whole and `contours_truncated` is set.
Simplification uses epsilon 1 working pixel. Closed edge traces and their simplified
polygons are **not semantic objects, filled masks, or guaranteed physical boundaries**.
OpenCV may trace both sides of a thin edge; duplicate-looking contours are possible.
No depth, obstacle class, or landing safety should be inferred from these alone.

See the official [Sobel/filter documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
and [Canny/corner documentation](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html)
for the underlying classical operations.

## Running diagnostics and ablations

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
./build/visual_features 00001.png output/pre_extraction --benchmark 100 --data
./build/visual_features 00001.png output/edges \
  --features gradients,edge_magnitude,edge_orientation --benchmark 100
ctest --test-dir build --output-on-failure
```

Outputs: `visual_features.jpg`, `run.txt`, optional `visual_features.yml.gz`, and
optional `benchmark.csv`. The structured export stores named raw planes, masks,
units, points, contours, coordinate metadata and spatial settings. Its motion
section is a summary; the in-process API exposes the full motion result/settings.
JPEG contrast/color mapping is for inspection only and does not alter exported data.

The supplied `00001.png` is 512×512 and becomes 240×240 without aspect distortion.
Neither root sample has verified temporal timing or calibration, so the JPEG
marks motion unavailable. It does not claim their difference is real motion.
Temporal integration is tested with known synthetic sequential warps. Actual
camera sequences still need calibration and independent motion ground truth.

The CLI uses one OpenCV thread, excludes 10 warmup frames and measures a repeated
still image. It reports preparation alone, each selected spatial feature in
isolation, and their shared combination: mean, median, p95, worst time, logical
payload bytes, and stage times. Isolated costs include dependencies; summing them
overcounts shared work. Motion is excluded from this still-image benchmark; use
`motion_benchmark` for its synthetic sequence. Capture, disk I/O, rendering and
serialization are excluded from extraction timing. Times are wall-clock latency,
not wattage or process CPU utilization.

Measured on 2026-09-27, Apple M3 Pro, OpenCV 5.0.0, Release build, one thread,
100 measured repetitions of `00001.png` after 10 warmups:

| Selection | Mean ms | Median ms | p95 ms | Worst ms |
| --- | ---: | ---: | ---: | ---: |
| Preparation only | 3.358 | 3.352 | 3.536 | 3.576 |
| Gradients (including preparation) | 3.380 | 3.387 | 3.578 | 3.656 |
| FAST (including preparation) | 3.278 | 3.263 | 3.410 | 3.473 |
| Texture orientation (including preparation) | 3.928 | 3.886 | 4.075 | 4.091 |
| All ten spatial families | 5.226 | 5.200 | 5.358 | 5.376 |

The 512×512 source becomes 240×240. Combined-stage means: preparation 3.219 ms,
derivatives 0.083, edges 0.097, tensor 0.807, corners 0.137, FAST 0.062,
color transitions 0.238, LBP 0.108, boundaries 0.475. Separate runs have normal
timing variability (FAST's total happened to fall below the baseline run).
Do not extrapolate these measurements to a small SBC. Native low-resolution
capture avoids this particular resize cost. Full per-family results are generated
in `output/pre_extraction/benchmark.csv`; rerun on the target board.

The legacy benchmark above predates the primary tensor profile. Current validation
has 50 regression cases (including nine analytical groups and the preserved visual,
motion, terrain, reconstruction, configuration and output groups),
synthetic motion demonstration and stereo-demo JSON round-trip pass in Release.
The suite also passes AddressSanitizer/UndefinedBehaviorSanitizer with float-cast
overflow checks. A spatial-only build with motion, stereo and RGB-D executables
disabled passes its applicable regression suite. New tests cover numeric gradients,
orientation validity, LBP bit order, chromatic transitions, point/contour budgets,
ablation scheduling, coordinate mapping, structured export and temporal integration.

## Compute and memory choices

All families are enabled by default for research inspection; this is not a
recommended flight feature set. Select only the features useful to an experiment.
Disabled families skip their stages and output maps. Derivatives, tensor products,
grayscale and Canny are shared where needed. Persistent Mats reuse storage,
sparse output vectors reserve their bounded capacities, and scalar image loops
use float. Orientation trigonometry runs only when selected. Diagnostic rendering,
export, and representation-adapter allocation are explicit off-path calls.

OpenCV contour/FAST output and temporary polygons can still allocate; this is not
a zero-allocation or hard-real-time implementation. Resolution bounds their input
cost, while budgets bound retained output. Contour processing and candidate sorting
depend on texture. Dense analytic maps can use **more memory than RGB alone**:
the all-feature sample payload is about 2.38 MB, excluding scratch/capacities/RSS.
The next encoder must select, pool or sparsify useful data rather than assume that
extracting every map automatically saves resources.
