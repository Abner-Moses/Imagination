# Sparse temporal motion

`metric_motion` estimates local camera motion from sequential frames. It shares
Imagination's calibration, ultrasonic measurement, and z-up conventions. It
does not invoke ORB, stereo reconstruction, terrain analysis, or a neural model.
The existing two-view pipeline remains responsible for dense reconstruction.

```text
CAMERA FRAME + monotonic capture timestamp
      |
      v
at most 320x240 grayscale (never upscale; adjust intrinsics)
      |
      v
up to 100 spatially distributed, persistent Shi-Tomasi corners
      |
      v
sparse pyramidal Lucas-Kanade; reuse previous pyramid and gradients
      |
      v
status / border / patch error / optional backward check
      |
      v
median displacement gate -> calibrated similarity fit -> inlier consensus
      |                          |
      |                          +--> yaw, scale, coverage, residual, quality
      v
rotation-compensated normalized displacement
      + fresh ultrasonic camera heights at both frames
      |
      v
metric horizontal displacement / timestamp difference = velocity
      |
      +--> local dead-reckoned pose, explicitly numbered continuity segment
      |
      v  [optional; only with age, baseline, parallax and reprojection gates]
up to four triangulation attempts every five frames
      |
      v
bounded sparse landmarks -> Imagination perception/navigation caller
```

## Three different operations

**Optical flow** estimates correspondence `p1 -> p2`. LK minimizes local patch
brightness differences, using image gradients to solve a small linear system.
It operates on selected neighborhoods, not one flow vector per image pixel.
No descriptors or all-to-all matching are needed for adjacent frames. Image
pyramid/gradient construction still processes the image: sparse flow is not
literally only 100 pixel reads.

**Ego-motion** interprets correspondences under a physical model. This estimator
assumes rectified pinhole images, a downward camera, negligible roll/pitch
change, predominantly static, approximately horizontal planar ground, and
small inter-frame motion. Camera motion is opposite to scene image motion.

**Triangulation** intersects calibrated rays from different camera positions.
Correspondence alone has neither a known baseline nor metric depth. The
optional map uses metric poses integrated from accepted range-scaled motion;
it is not an independent validation of that scale or of the plane assumption.

## Equations and coordinates

Normalize a pixel with the intrinsics at the working resolution:

```text
q = [(u-cx)/fx, (v-cy)/fy]^T        camera ray = [qx, qy, 1]^T
q2 = s R(theta) q1 + t             fitted similarity, four parameters
R(theta) = [cos(theta), -sin(theta); sin(theta), cos(theta)]
s = h1/h2                         ideal horizontal-plane height ratio
```

Fit in normalized coordinates so unequal `fx` and `fy` do not turn yaw into an
incorrect pixel-space rotation. Centered least squares obtains `a=s cos(theta)`
and `b=s sin(theta)` from dot/cross products; translation is the centroid
difference after applying this rotation/scale. No SVD or generic homography
solver is required.

At zero yaw, the camera-to-world rotation is `D3 = diag(1,-1,-1)`:
world X points image-right, Y image-up, Z up. For local world yaw `psi`,
`R_WC = Rz(psi) D3`. The fitted image angle `theta` equals the change in this
world yaw. Let `D2 = diag(1,-1)`. Horizontal camera displacement, expressed in
the **previous camera's local z-up axes**, is:

```text
normalized_displacement = -D2 R(theta)^T t / s
planar_displacement_m   = -h2 D2 R(theta)^T t
planar_velocity_mps     = planar_displacement_m / (timestamp2-timestamp1)
```

With no yaw and constant height this reduces to:

```text
delta_X = -h * delta_u / fx
delta_Y = +h * delta_v / fy
```

`median_flow_px` is the raw median accepted scene displacement, not the fitted
camera translation. `normalized_displacement` is dimensionless and remains
available without altitude. Its unit direction is absent for motion below
0.1 equivalent pixel, including stationary/pure-yaw frames. Check `valid`
before consuming these fields. Zero-initialized fields on invalid results do
not mean a measured zero velocity.

An ultrasonic sensor pointing world-down measures `range = h_camera + offset_z`.
`altitudeFromUltrasonic()` therefore returns `h_camera = range - offset_z`.
For this adapter the existing measurement's offset must be from the **camera
center**, not an unrelated UAV reference. A horizontal offset is valid only
over the same horizontal surface. Timestamps must use the frame clock.

Both heights must be positive, finite, nonfuture, and at most 0.15 s old at
their corresponding frame. The fitted scale must agree with `h1/h2` within
5% relative error. Missing/invalid/stale readings or a failed scale check
suppress metric displacement/velocity; image motion may still be valid.
This check cannot detect a constant multiplicative range error. Sensor operating
range, echo validity, and synchronization must be checked by the acquisition
layer. `maximum_error_m` in `UltrasonicMeasurement` belongs to the existing
terrain check; it is not a probabilistic uncertainty model for this estimator.

## Tracking, rejection, and quality

| Default | Value |
| --- | --- |
| Maximum working image | 320x240, aspect preserved, no upscaling |
| Corner budget / grid | 100 / 5 columns by 4 rows, per-cell quotas |
| Replenishment | Below 60 tracks, at least 5 frames between detections |
| Corner selection | Shi-Tomasi, quality 0.01, spacing 7 px, block 3 |
| LK | 15x15 window, maxLevel=1 (base + half resolution), 12 iterations, epsilon 0.03 |
| LK error | Mean absolute patch difference <=20 on 8-bit images; min eigenvalue 1e-4 |
| Backward check | Off; optional round-trip threshold 0.75 px |
| Geometry | At least 12 inliers, residual gate 1.5 px, >=60% of attempted tracks |
| Coverage / quality | >=30% of grid cells / >=0.2 |
| Per-frame model limits | Absolute yaw <=0.15 rad; scale in [0.8,1.2] |
| Maximum frame gap | 0.5 seconds; larger gaps reseed and report invalid motion |
| Triangulation | Off by default; >=5 observations, >=0.05 m baseline, >=1 degree parallax |
| Triangulation budget | Every 5 frames, <=4 attempts, <=64 stored landmarks |
| Triangulation checks | Positive depth <=100 m, <=1 px reprojection error in each view |

First discard unsuccessful, nonfinite, border-crossing, and high-error LK
tracks. If enabled, run backward LK only on survivors. A loose robust gate
keeps displacements within `max(3 px, 4*median radial deviation)` of the
componentwise median; this leaves room for spatially varying yaw flow.
Fit a similarity. If fewer than 80% of LK survivors support that model, or
it violates the motion limits, evaluate at most 32 deterministic two-point
RANSAC-style hypotheses. Refit twice on pixel-residual inliers. The final
support/coverage gates apply against the original attempted track count.

```text
quality = (inliers / attempted_tracks) * occupied_grid_fraction
          / (1 + RMS_model_residual_px / model_error_px)
```

This is a measurable health score, not a calibrated probability. A majority
moving object, repetitive texture, or a coherent wrong match can still defeat
it. Parallax gates triangulation, not flow quality: hovering can give a valid
zero-motion estimate even though no depth can be triangulated.

Cheap yaw/scale fitting runs every frame; reusing an old yaw estimate while the
vehicle turns would bias velocity. It uses short linear passes over the tracks
and costs far less than LK. Stronger consensus is conditional, corner detection is
count-triggered and rate-limited, and triangulation is both rate- and geometry-gated.
Low-texture frames retry detection at the same bounded interval.

## Sparse geometry and continuity

Each retained track has a stable ID/age and, when mapping is enabled, one anchor
pixel/pose. For unit world rays `u,v` and baseline `B=C2-C1`, use:

```text
c = dot(u,v); denominator = 1-c*c = sin(parallax)^2
lambda = (dot(u,B)-c*dot(v,B)) / denominator
mu     = (c*dot(u,B)-dot(v,B)) / denominator
P = (C1 + lambda*u + C2 + mu*v) / 2
```

Small baseline and nearly parallel rays make depth unstable. Check baseline
before attempting a point, then parallax, positive depth and reprojection in
both cameras. Pure rotation supplies no baseline. Attempt only a few tracks,
rotate the starting index for fairness, and triangulate each track once.
The map replaces old entries when its capacity is reached; it never grows
without bound and does not run bundle adjustment or loop closure.

`pose` uses the start of its `segment_id` as `(0,0,0)` with initial yaw zero.
It is local dead reckoning, not GPS/global localization. Missing metric scale,
bad motion, reset, or a frame gap breaks the chain, clears landmarks/anchors,
and starts a new segment once fresh height is available. Never merge different
segments without an external registration. Vertical pose changes assume the
same ground elevation; rising terrain cannot be distinguished from descent.

## Calling from Imagination

```cmake
target_link_libraries(your_navigation_target PRIVATE metric_motion)
```

```cpp
#include "imagination.hpp"

metric_mapping::OpticalFlowSettings settings;
settings.triangulation = true;  // Optional small map.
metric_mapping::SparseFlowTracker tracker(calibrated_camera, settings);

// For each frame: timestamp and range timestamp are on the same monotonic clock.
auto altitude = metric_mapping::altitudeFromUltrasonic(reading, range_timestamp_s);
auto motion = tracker.processFrame(rectified_frame, frame_timestamp_s, altitude);
if (motion.valid && motion.planar_velocity_mps) {
    // Consume local horizontal velocity. Transform using your navigation attitude
    // before combining it with an externally registered world map.
    const auto velocity = *motion.planar_velocity_mps;
}
// If no valid range is available:
// auto motion = tracker.processFrame(rectified_frame, frame_timestamp_s);
const auto& landmarks = tracker.landmarks(); // Borrowed; changes on next process/reset.
```

`tracks()` and `landmarks()` expose const references without copying. To use
landmarks in existing terrain tools, explicitly transform `landmark.point.world_m`
from its local segment into your map frame, then construct `ColoredPoint` values
for `VoxelGridAccumulator` or `createTerrainGrid`. They carry no sampled RGB;
assign a display color yourself. Sparse landmarks do not establish a complete
landing footprint, so this library does not automatically mark terrain safe.
The tracker is single-stream, noncopyable, and not internally thread-safe.
Calibration/settings are fixed for its lifetime; construct another tracker
after changing calibration. `reset()` clears temporal continuity.

## Live uncalibrated optical flow

For calibrated optical flow with **IMU and ultrasonic inputs**, see
[sensor_inputs.md](sensor_inputs.md). Both live camera commands support that mode.

For image-space motion directly from a USB/built-in camera:

```sh
cmake -S . -B build -DBUILD_MOTION_CAMERA=ON -DBUILD_TESTING=ON
cmake --build build -j2
mkdir -p output
./build/imagination optical_flow 0 300 output/camera_flow > output/camera_flow.csv
# Same command through its compatibility executable:
./build/optical_flow 0 300
```

Arguments are camera device index (default 0), frame count (default 300), and an
optional directory for arrow images every ten frames. The camera is opened through
OpenCV `VideoCapture`. It requests 320×240 at 10 FPS and uses the actual delivered
dimensions if the camera ignores the size request. Processing still downsamples
to at most 320×240. The backend may ignore FPS/buffer requests; timestamps describe
frame delivery, not guaranteed exposure time. Grant camera access to the terminal
application if your operating system requires it. Capture failures stop with an error.

Each captured frame goes through grayscale preparation, persistent sparse LK
tracking, residual/boundary rejection and robust image-space similarity consensus.
Defaults are 100 points, a 15×15 window and one extra pyramid level. This shares
the existing tracker implementation; it does not run dense flow or descriptors.

The flushed CSV stream reports validity/status, working dimensions, active/tracked/
accepted point counts, dominant horizontal/vertical displacement in **working
pixels**, image velocity in **pixels/second**, quality and processing milliseconds.
Positive flow is right/down. The first frame initializes tracks; missing texture or
poor consensus yields an invalid estimate. Blank velocity fields mean unavailable.
These are image measurements, not UAV displacement or metric velocity.

`PixelFlowTracker` provides the same camera-independent processing to C++ callers:

```cpp
// Construct after obtaining the first frame's actual size. No calibration needed.
metric_mapping::PixelFlowTracker flow(first_frame.size());
auto initial = flow.processFrame(first_frame, first_timestamp_s);
auto result = flow.processFrame(next_frame, next_timestamp_s);
if (result.valid) {
    // result.median_flow_px and optional result.velocity_px_s
    // flow.tracks(): persistent IDs, previous/current working pixel coordinates
}
// Explicit optional drawing: cv::Mat arrows = flow.debugImage();
```

The wrapper uses equal coordinate scales solely to fit an image-space similarity;
these constants are not measured intrinsics. Its return type exposes no metric
motion or camera pose, and it never accepts altitude or triangulates. Raw distorted
images can be tracked, but significant perspective changes, large distortion,
blur or a majority moving foreground can defeat the simple consensus model.
Use calibrated `motion_camera` below for physical motion estimation and fresh
range-sensor scale. No raw-frame optical flow should be interpreted as proof of
safe navigation.

## Synthetic profiling and calibrated capture

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_MOTION_CAMERA=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/motion_benchmark --frames 1000 --debug /tmp/motion-demo
./build/motion_benchmark --frames 1000 --levels 0
./build/motion_benchmark --frames 1000 --forward-backward --triangulate
./build/motion_benchmark --frames 500 --speed 6
```

The benchmark generates reproducible warped texture, uses one CPU thread,
excludes image generation/debug export from tracking measurements, and reports
20 warmup frames separately from measured frames. It reports average/median/
p95/worst elapsed time, process CPU time, stage averages, feature/replenishment/
inlier counts, and error against synthetic motion. Detection events and cold
initialization are reported separately so warmup does not hide their cost.
The exit code checks accuracy/validity, never a machine-specific timing limit.
Processing-capacity FPS is not camera throughput. No watts are inferred.

For a webcam, fill `configs/motion_camera.yaml` with measured calibration:

```sh
./build/motion_camera configs/motion_camera.yaml 0 300 > motion.csv
./build/motion_camera configs/motion_camera.yaml 0 300 /tmp/range.txt /tmp/motion-debug > motion.csv
```

The optional range snapshot file contains exactly three numbers:
`timestamp_s range_m sensor_offset_z_m`. An external acquisition process should
atomically replace it with each new valid sample. Timestamp uses the runner's
`steady_clock` seconds-since-epoch clock (normally CLOCK_MONOTONIC on Linux);
use the same clock in your sensor process. Invalid/stale files suppress metric
output. This is an adapter, not a GPIO/serial sensor driver. File I/O, capture,
CSV reporting, and optional PNG export are outside the library's timed hot path.
The runner requests 10 FPS and calibrated size, rejects a returned size mismatch,
and timestamps frame delivery; a capture backend can still buffer frames.
For flight integration prefer actual capture/exposure timestamps in the C++ API.
The runner assumes already rectified/negligibly distorted frames; it does not
calibrate or undistort a raw webcam stream.

`BUILD_MOTION=OFF` removes the temporal library/benchmark dependency. Webcam
support is independently opt-in (`BUILD_MOTION_CAMERA=OFF` by default), with
no GUI. OpenCV `video` provides LK; some packaged OpenCV builds (including the
tested Homebrew build) link `video` transitively to `dnn`. This code calls no
DNN functions and loads no models; use a suitably minimal OpenCV build if its
packaged transitive libraries matter to deployment size.

## Compute choices and limits

Two reusable image/derivative pyramid buffers avoid rebuilding the previous
frame or recomputing gradients for backward tracking. Mats and track/scratch
vectors retain capacity. Geometry uses double precision for approximately 100
points, while LK pixels/coordinates use OpenCV's 8-bit/float representation.
Corner detection visits only underfilled cells, preserving surviving IDs.
There is no per-frame image warp, descriptor, dense flow, dense depth, essential
matrix, or neural inference. Debug images are produced only by `debugImage()`;
stage clocks are enabled only with `profiling=true`. The application, not the
library, chooses process-wide OpenCV thread count.

The base plus half-resolution pyramid is justified by the larger-motion test;
single-level LK is cheaper for consistently subpixel motion but loses tracks
on faster motion and then pays for repeated detection. These defaults are a
starting point, not a proven optimum on a particular SBC. See the measured
[benchmark report](architecture.md). Frame storage scales with working
resolution; tracking/model state scales with the feature budget. At defaults,
the two unpadded gray/derivative pyramids contain about 0.96 MB, plus padding,
gray/mask/resizing buffers, small track state, and OpenCV internal scratch.
This is an estimate of owned buffers, not measured process RSS.

Limitations include pitch/roll, sloped/nonplanar terrain, substantial depth
variation, motion blur, exposure changes, rolling shutter, large inter-frame
motion, repeated patterns, majority moving objects, inaccurate calibration,
range errors, and accumulated drift. A low model residual cannot prove nadir
attitude. No relocalization, global map registration, sensor fusion filter,
obstacle semantics, or flight controller is included. Hardware capture, SBC
latency, and actual power consumption require testing on the target device.

The experiment tests a narrow hypothesis: known physical constraints and
temporal continuity can yield useful, compact motion information with little
computation. It does not establish that learned perception is inefficient in
general or replace higher-level scene understanding.

API reference: [OpenCV sparse LK and pyramid documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

## Analytical tensor integration

The `pre_extract` profile adds **dense Farneback** for two spatial U/V channels.
It is independent of sparse LK and does not replace sparse ego-motion or
triangulation. `retain_correspondences=true` exposes `correspondences()` after
LK/error/optional backward rejection but before the nadir similarity gate, enabling
bounded essential-pose diagnostics without a second sparse tracker.
The analytical mode reuses the metric tracker's local segment/landmarks, samples
color, and connects them to bounded terrain/projection processing. Sparse-only
commands keep their original defaults and computational path. See
[the full analytical pipeline](pre_extraction.md#temporal-and-3-d-branches).
