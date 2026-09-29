# A beginner's guide to Imagination

Think of the program as a set of tools that turn measurements into useful facts.
The files follow those jobs; the public header describes what each tool accepts
and returns. The same data can be used by later research stages without drawing it.

```text
main.cpp: read arguments and choose a workflow
    |
    +-- config.cpp: validate settings and inputs
    +-- vision.cpp: camera frame -> gradients, points, textures and boundaries
    +-- motion.cpp: sequential frames -> optical flow -> camera motion
    +-- geometry.cpp: calibrated images/depth -> metric points
    |       |
    |       +-- terrain.cpp: points -> ground grid -> landing checks
    |
    +-- output.cpp: write files or draw diagnostics when explicitly requested

tests.cpp: give each stage known inputs and check the answers
```

`imagination.hpp` is the shared public contract. `src/internal.hpp` contains only
small helpers needed between modules. Implementation classes and scratch buffers
stay inside the module that uses them. The project uses ordinary functions,
records and OpenCV buffers; there is no plugin framework or dynamic module loader.

## Words you will see in the code

- **Pixel:** one cell in an image. `cv::Mat` stores an image or numeric grid.
- **Gradient:** how quickly brightness changes across neighboring pixels.
- **Corner:** an image point whose neighborhood changes in two directions.
- **Optical flow:** the movement of an image point between frames, in pixels.
- **Ego-motion:** the camera's motion estimated from image correspondences.
- **Calibration:** numbers that relate image pixels to rays leaving the camera.
- **Triangulation:** intersecting rays from different camera positions to estimate a 3-D point.
- **Terrain grid:** ground elevation sampled in square cells.
- **Validity mask:** which values can be used; an unknown value is not a safe value.
- **`std::optional`:** a result that may be unavailable; check it before using it.
- **Regression test:** a known example that catches accidental behavior changes.

Optical flow alone does not measure metres. The motion module needs calibrated
camera geometry and a valid height reading to estimate metric planar motion.
Contours are image boundaries, not semantic object detections. These distinctions
are essential when interpreting the output or making navigation decisions.

## How to read the main processing functions

Start with `VisualFeatureExtractor::extract()` in `src/vision.cpp`. Its flow is:

1. `validateFrame()` checks type, calibration and timestamps.
2. `prepare()` reduces resolution and converts to grayscale.
3. `extractSpatialFeatures()` computes selected maps, sharing derivatives and the tensor.
4. `motion()` calls the existing temporal tracker when calibration is available.
5. Return the reusable result with coordinates, validity, timing and feature records.

Next read `SparseFlowTracker::processFrame()` in `src/motion.cpp`:

1. Check the input and decide whether the height measurement is fresh enough.
2. Prepare grayscale, then build/reuse the small image pyramid.
3. Track points with Lucas-Kanade.
4. `estimateTrackedMotion()` rejects geometric outliers and checks whether metric scale is available.
5. Update optional sparse landmarks, replenish missing points, and remember this frame.

The explicit checks are part of the algorithm, not unnecessary complexity.
Changing pixel-center math, coordinate signs, height freshness, or the treatment
of unknown terrain can produce plausible but wrong answers.

## What was simplified inside the code

- One camera-resize rule now serves both motion and visual extraction. It preserves
  camera rays and OpenCV's pixel-center convention, including odd image dimensions.
- Visual feature dependencies are named booleans instead of repeated long conditions.
  Disabled features still avoid unnecessary work.
- The motion loop delegates pyramid construction and geometric/metric estimation
  to named steps. Track IDs and their triangulation anchors are filtered together.
- Commands share strict integer parsing and benchmark statistics. One command table
  drives both help and execution, so their lists cannot accidentally disagree.
- Public declarations no longer include test-only and workflow helper structures.
- Sample configuration is normal YAML again; documentation is split by topic.

## Making a safe first change

Change one setting or one feature at a time. Locate its test and add a small known
example when changing behavior. Run the full tests, inspect the output, and explain
the physical assumptions in your review. Do not remove a calibration or hazard
check merely to obtain more valid-looking output.

Return values from the visual extractor borrow reusable buffers; clone maps before
keeping them past the next frame. One tracker/extractor belongs to one stream.
Output drawing and file writing are explicit operations outside the flight hot path.

## Build and compatibility

CMake compiles modules directly and omits optional motion/vision sources when disabled.
The common library is available as `Imagination::core` or `metric_mapping`; existing
`metric_motion`, `metric_stereo` and `metric_visual` names remain aliases when enabled.
Tiny forwarding headers under `build/include/metric_mapping/` support older includes.
They contain no implementation and are not files contributors need to maintain.

`configs/` contains the canonical examples. For existing scripts, CMake also creates
missing copies under `build/configs/` without overwriting edited copies. Relative
paths are always resolved from the YAML file's own directory; prefer the canonical
examples when starting a new experiment.

The existing geometric pipelines remain independent of the future learned model.
The next research boundary is synchronized, calibrated sensor data joined to the
feature records, then an explicit representation/token encoder. See
[the pre-extraction reference](pre_extraction.md) for the full integration plan.

## Verification of the modular refactor

The Release build passes 50 regression cases plus synthetic-motion, analytical
still/sequence and demo-generation/JSON CTests. The
AddressSanitizer/UndefinedBehaviorSanitizer build,
including float-cast overflow checks, also passes. The diagnostic JPEG remains
byte-for-byte identical. Spatial-only, motion-only and RGB-D-only builds are
checked independently. Webcam support compiles; physical camera/sensor acquisition
has not been exercised by these automated tests.

The live `optical_flow` command now uses `PixelFlowTracker` for uncalibrated camera
frames. It reuses sparse tracking but exposes only pixel displacement/velocity,
never metric pose. Its additional regression covers BGR input, downsampling,
known translations, timing, reset, texture loss and optional arrow rendering;
the current full suite has 38 cases. `motion_camera` remains the calibrated path.

## Primary analytical tensor profile

`analyticalFeatureSettings()` selects the 28-channel research bank inside the
existing `VisualFeatureExtractor`. `pre_extract` is a command in `imagination`,
not another application. The flow ends at a channel-major float32 tensor and mask:

```text
vision: shared luminance/derivatives → appearance/HOG/Harris/boundaries/chroma
        previous luminance         → dense Farneback U/V
motion: existing sparse LK         → persistent correspondences
geometry: essential pose diagnostics (unit-baseline scale)
          validated nadir/range poses → filtered metric triangulation
vision: bounded colored local map  → terrain grid + bounded IDW
terrain: surface-only measurements → metric slope/diffusion/residual roughness
geometry: current-camera projection + z-buffer → supported image-aligned geometry
vision: fixed normalization + selection + masks → C×32×32 tensor → STOP
```

`imagination.hpp` extends the existing settings/result contract. `src/vision.cpp`
coordinates the stages; `src/motion.cpp` exposes pre-model LK correspondences;
`src/geometry.cpp` recovers relative pose and projects terrain;
`src/terrain.cpp::measureSurface()` shares diffusion and residual statistics without
running landing decisions. `src/config.cpp` validates ablation/numeric settings;
`src/output.cpp` renders fixed-scale tensor diagnostics. `main.cpp` accepts still,
manifest, video/camera and explicitly synthetic inputs. Tests remain in `tests.cpp`.

No additional C++ code files were introduced. All previous build switches and
library aliases remain. With `BUILD_VISUAL_FEATURES=ON`, OpenCV video provides
dense Farneback even when sparse `BUILD_MOTION=OFF`; calibration/geometry modules
support rectification and relative-pose functions. Disabling visual features
removes those analytical dependencies. The existing sparse-only path does not
start computing dense flow.

The [pre-extraction guide](pre_extraction.md) is the authoritative channel, unit,
validity, configuration, coordinate and timing contract. The full bank is a
candidate experiment, not a claim that computing every feature is cheapest.
Learned adapters, CoHAtNet/HTransformer, semantic outputs and navigation are future
consumers of that contract, not dependencies of this implementation.
