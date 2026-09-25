# Reading and changing the code

The project has two real reconstruction paths and a separate synthetic demo.
Both real paths produce colored points, then share terrain mapping, landing
analysis, and file writers. An optional ultrasonic reading checks the mapped
ground before either real path recommends a landing site.

## Start here

For the complete stereo workflow, read these in order:

1. [`apps/two_view.cpp`](../apps/two_view.cpp) parses arguments and reports errors.
2. [`src/app/stereo_run.cpp`](../src/app/stereo_run.cpp) connects reconstruction,
   terrain mapping, landing analysis, and exports.
3. [`src/stereo/reconstruction.cpp`](../src/stereo/reconstruction.cpp) shows the
   three reconstruction stages: feature matching, alignment, and dense depth.
4. [`src/landing/analysis.cpp`](../src/landing/analysis.cpp) validates the input,
   measures terrain, and selects a landing site.

For RGB-D, start at [`apps/metric_mapper.cpp`](../apps/metric_mapper.cpp), then
[`src/app/rgbd_run.cpp`](../src/app/rgbd_run.cpp). Its depth and poses come from
the supplied configuration instead of stereo matching.

```text
apps/                         arguments and exit codes
  |
src/app/                      application workflows
  |-- src/rgbd/               calibrated depth + supplied camera poses
  |-- src/stereo/             two images + measured baseline
  |-- src/demo/               explicitly synthetic appearance-derived geometry
  |
src/terrain/                  colored points -> regular elevation grid
  |
src/landing/                  terrain measurements -> hazards -> ranked sites
  |
src/output/                   PLY, CSV, JSON, and preview images
```

## Where to make a change

| Concern | File |
| --- | --- |
| YAML schema and strict validation | `src/rgbd/config.cpp` |
| Camera transforms and depth back-projection | `src/rgbd/geometry.cpp` |
| Voxel fusion and cloud bounds | `src/terrain/point_cloud.cpp` |
| Grid boundaries and IDW interpolation | `src/terrain/grid.cpp` |
| ORB matching and geometric verification | `src/stereo/features.cpp` |
| Yaw correction and disparity search bounds | `src/stereo/alignment.cpp` |
| StereoSGBM, consistency checks, and metric XYZ | `src/stereo/dense_depth.cpp` |
| Smoothing, slope, roughness, and hazard clearance | `src/landing/surface.cpp` |
| Fuzzy membership functions and rule tables | `src/landing/fuzzy.cpp` |
| Hard constraints and best-site selection | `src/landing/selection.cpp` |
| Downward ultrasonic range validation and ground comparison | `src/landing/ultrasonic.cpp` |
| PLY/CSV exports and JSON escaping | `src/output/cloud_files.cpp` |
| Scalar-map coloring and panel layout | `src/output/raster.cpp` |
| Landing maps and selected-site overlay | `src/output/landing_images.cpp` |
| Landing-site JSON | `src/output/landing_json.cpp` |
| RGB-D and stereo/demo metadata | `src/output/rgbd_metadata.cpp`, `stereo_metadata.cpp` |
| Cloud/terrain debug overview | `src/output/overview.cpp` |
| Synthetic scene generation | `src/demo/synthetic_scene.cpp` |

## Interfaces and ownership

`include/metric_mapping/` is the public library interface. Its headers keep the
existing names so callers do not need to follow implementation file moves.
`types.hpp` defines the shared data: camera calibration, colored points, terrain
grids, settings, and results.

Headers under `src/` are private to the implementation. They declare only the
small records and stage functions needed between neighboring modules. For
example, `stereo_internal.hpp` describes verified feature matches and the
aligned image pair. It is not a second public API.

Algorithms do not print progress or write files. `src/app/` chooses the order
of operations and output names; `src/output/` handles serialization and
presentation. Synthetic geometry stays under `src/demo/`, separate from the
metric reconstruction algorithms.

The refactor keeps the lightweight implementation: ordinary functions and
value types, shared OpenCV image buffers, no plugin framework or new runtime
dependencies. Fuzzy scores are cached per analysis, and IDW reads only measured
cells while updating the grid in place.

## Coordinate conventions

- Distances and elevations are in metres; slope is in degrees.
- World `+Z` points upward. Grid columns increase toward `+X`; rows toward `-Y`.
- Grid validity is `255` for measured, `127` for interpolated, and `0` for unknown.
- OpenCV camera coordinates use `+X` right, `+Y` down, and `+Z` forward.
- The stereo path assumes nadir captures and uses camera 1 as the world origin.
- An unknown or incomplete landing footprint is a hazard, not a safe default.

These conventions are shared across modules; changing one requires checking
reconstruction, mapping, landing analysis, and exports together.

## Tests

The tests mirror the source domains: `rgbd_tests.cpp`, `config_tests.cpp`,
`terrain_tests.cpp`, `ultrasonic_tests.cpp`, `stereo_tests.cpp`, and `output_tests.cpp`.
`test_support.hpp` contains assertions and small fixtures; `test_main.cpp`
collects and runs the groups. Temporary files are cleaned up even when a test
throws. The Unix CLI test in `demo_json.cmake` also checks quoted filenames and
that all demo lattice cells remain measured.

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Use `-DBUILD_TWO_VIEW=OFF` for RGB-D only, or `-DBUILD_RGBD=OFF` for stereo only.
Tests remain optional, and new build directories default to Release mode.
