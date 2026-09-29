# Imagination

A C++17/OpenCV research project for **GPS-denied UAV perception**.
It uses classical camera mathematics to extract visual features, estimate motion,
reconstruct terrain, and check potential landing sites.

Research direction: **sensors → analytic pre-extraction → convolution-assisted
Transformer → navigation**. The camera and geometric stages exist today. The full
sensor framework and learned model are future work; resource and safety comparisons
against CNN-ViT systems still require experiments.

The primary analytical research profile now produces a **28×32×32 float32 tensor**
with channel names, per-cell validity masks, timings and reconstruction diagnostics.
It extends the existing visual extractor; no learned adapter or Transformer is run.

```sh
./build/imagination pre_extract 00001.png output/analytical --data --debug
./build/imagination pre_extract ignored output/analytical_synthetic --synthetic --frames 80 --debug
```

Open `output/analytical/analytical_features.jpg`. The still image has no valid
motion or metric geometry; the separate synthetic sequence exercises those paths.
See the [tensor, sequence, camera and ablation guide](docs/pre_extraction.md).

## Build and try it

Install CMake, a C++17 compiler and OpenCV (`brew install cmake opencv` on macOS), then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/imagination visual_features 00001.png output/pre_extraction --data
```

Open `output/pre_extraction/visual_features.jpg`. It shows gradients, edges,
corners, keypoints, color transitions, texture, contours and boundaries. Motion is
unavailable for this still image because its timing and calibration are unknown.
Add `--benchmark 100` to profile each feature separately.

## Ten C++ files, each with one responsibility

| File | What belongs here |
| --- | --- |
| [imagination.hpp](imagination.hpp) | Public settings, results and function declarations |
| [src/internal.hpp](src/internal.hpp) | Small shared implementation declarations |
| [src/config.cpp](src/config.cpp) | Configuration validation, argument parsing and timing summaries |
| [src/geometry.cpp](src/geometry.cpp) | Camera math, RGB-D/stereo reconstruction and point clouds |
| [src/terrain.cpp](src/terrain.cpp) | Terrain grids, ultrasonic ground checks and landing analysis |
| [src/motion.cpp](src/motion.cpp) | Persistent optical-flow tracks, ego-motion and sparse triangulation |
| [src/vision.cpp](src/vision.cpp) | Selectable image features and reusable feature data |
| [src/output.cpp](src/output.cpp) | File exports and diagnostic images |
| [main.cpp](main.cpp) | Commands and complete application workflows |
| [tests.cpp](tests.cpp) | Known-input regression tests |

Documentation lives in `docs/`, editable examples in `configs/`, and generated
results in `output/`. These are ordinary supporting files, outside the code-file
budget. There is no source generation or giant combined implementation file.

Offline dataset preparation, PyTorch training, analytical-versus-RGB baseline
evaluation, ablations, and ONNX export live separately in
[training/](training/README.md). Python is not required by the onboard C++ runtime.

## Where to start as a contributor

Read the [beginner architecture guide](docs/architecture.md), then follow
`VisualFeatureExtractor::extract()` in `src/vision.cpp`. It shows the steps in order:
validate the frame, prepare it, extract selected spatial features, then update motion.

Use descriptive names and keep physical units (`_px`, `_m`, `_s`, `_rad`) visible.
Edit one module at a time, add a known-input test when behavior changes, and run the
suite before sharing your work. Keep `build/` and `output/` out of commits.

## Commands

**Live camera optical flow (no calibration needed for pixel movement):**

```sh
cmake -S . -B build -DBUILD_MOTION_CAMERA=ON
cmake --build build -j2
mkdir -p output
./build/imagination optical_flow 0 300 output/camera_flow > output/camera_flow.csv
```

Use device `0` for the default camera, and replace `300` with the desired frame
count. Create `output/` first if it does not exist (`mkdir -p output`). The CSV
streams accepted flow in working-image pixels and pixels/second, track counts,
quality and processing time. Arrow images are saved every ten frames when the
optional output directory is given. The first frame initializes tracking.
This mode reports 2-D image motion; use `motion_camera` with calibration and fresh
altitude for approximate metric UAV velocity. See [camera flow details](docs/optical_flow.md#live-uncalibrated-optical-flow).

For **IMU + ultrasonic assisted flow**, fill in measured camera calibration and
provide live sensor snapshots from your sensor acquisition program:

```sh
./build/imagination optical_flow 0 300 output/camera_flow \
  --camera configs/motion_camera.yaml \
  --imu /tmp/imu.txt --ultrasonic /tmp/range.txt
```

IMU yaw constrains rotation; ultrasonic height supplies metric scale. Missing,
stale or unsupported IMU readings suppress metric output when `--imu` is enabled.
The [sensor input guide](docs/sensor_inputs.md) defines timestamps, axes, file
formats and the direct C++ API. Hardware acquisition remains outside the tracker.

Run `./build/imagination --help` to list enabled commands. Existing names such as
`./build/visual_features` and `./build/two_view` still work.

| Command after `./build/imagination` | Purpose |
| --- | --- |
| `pre_extract INPUT OUTPUT ...` | Analytical tensor from a still, timestamped sequence, video or camera; optional exports/diagnostics |
| `optical_flow [device=0] [frames=300] [debug_directory]` | Live sparse webcam flow without calibration; pixel units only |
| `visual_features IMAGE OUTPUT [--features NAMES] [--benchmark N] [--data]` | Image features, JPEG, reusable data and profiling |
| `motion_benchmark --frames 500 [--triangulate]` | Synthetic motion and runtime measurements |
| `motion_camera ...` | Calibrated webcam tracking; enable `BUILD_MOTION_CAMERA` first |
| `two_view IMAGE1 IMAGE2 FX FY CX CY BASELINE_M` | Calibrated metric stereo |
| `two_view --demo 00001.png 00002.png` | Explicitly synthetic geometry for checking outputs |
| `metric_mapper configs/reconstruction.yaml` | Registered RGB-D with measured calibration and supplied poses |

All build switches remain: `BUILD_TWO_VIEW`, `BUILD_RGBD`, `BUILD_MOTION`,
`BUILD_MOTION_CAMERA`, `BUILD_VISUAL_FEATURES`, and `BUILD_TESTING`.
For webcam support, configure with `-DBUILD_MOTION_CAMERA=ON` and rebuild.
The YAML examples deliberately require real sensor values before flight use.

For C++ integration, include `imagination.hpp`, use namespace `metric_mapping`,
and link `Imagination::core`. Existing library aliases and generated compatibility
include paths remain available. Prefer the public header over `src/internal.hpp`.

## Reference

- [Camera features, API, ablations and future model connections](docs/pre_extraction.md)
- [Optical flow, motion equations, altitude scale and triangulation](docs/optical_flow.md)
- [Stereo, RGB-D, terrain, landing and output formats](docs/reconstruction.md)
- [Coordinate and unit conventions](docs/coordinates.md)
- [Recorded motion benchmarks and limitations](docs/motion_benchmark.md)
