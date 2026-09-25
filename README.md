# Two-view metric terrain reconstruction

This C++17/OpenCV program reconstructs a dense colored terrain point cloud
from two calibrated aerial images, then produces the front-end map products
required by the landing-site paper:

```text
two calibrated, overlapping nadir images + measured baseline
    -> ORB matching and geometric verification
    -> parallel-view/yaw alignment
    -> dense StereoSGBM disparity with left-right consistency
    -> metric XYZRGB back-projection
    -> 0.1 m terrain grid
    -> bounded inverse-distance-weighting gap completion
    -> colored PLY + DEM + orthomosaic
    -> anisotropic DEM smoothing
    -> slope + roughness + nearest-hazard maps
    -> three-stage fuzzy landing-site selection
```

The paper itself uses geotagged images from several views and assumes the
colored point cloud and UAV position are already available. This executable is
a constrained two-image implementation of that input contract. It also
implements the paper's geometry-only landing analysis. Water detection is
disabled because the paper uses a neural semantic-segmentation model, which is
outside this C++/OpenCV-only proof of concept.

## Code layout

- `apps/`: short command-line entry points.
- `src/app/`: complete RGB-D, stereo, and demo workflows.
- `src/rgbd/` and `src/stereo/`: reconstruction algorithms.
- `src/terrain/`: point fusion, terrain grids, and interpolation.
- `src/landing/`: surface measurements, fuzzy scores, and site selection.
- `src/output/`: file writers and visualizations.
- `src/demo/`: synthetic scene generation.
- `tests/`: regression tests grouped by the same subjects.

Read [the source guide](docs/architecture.md) for the processing flow, a
file-by-file navigation table, and the shared coordinate conventions.
Public headers remain under `include/metric_mapping/`.

## Required capture geometry

The following are mandatory capture assumptions. The program checks image
sizes, alignment scale, epipolar error, and stereo consistency; two images alone
cannot prove the absolute camera attitude or correctness of the supplied
calibration and baseline.

- Use a perspective camera, not an orthographic camera or a 2-D map pan.
- Calibrate the camera at the exact saved image resolution.
- Undistort both images using the calibration before running this program.
- Point the camera straight down in both images.
- Keep the same altitude, pitch, roll, zoom, and resolution.
- Move the camera sideways by a measured distance in metres.
- Small yaw changes are corrected automatically.
- Capture a static, textured scene with substantial overlap.

`baseline_m` is the physical distance between the two camera centers. It is
what makes the cloud metric. The software does not invent this value.

The current `00001.png` and `00002.png` files are 512 x 512, while the earlier
principal point `(640, 360)` belongs to a larger image. These images do not come with
matching calibration or a measured baseline. Use them for the synthetic demo;
do not infer metric terrain from guessed camera values.

## Build and test

On macOS:

```bash
brew install cmake opencv
cd Imagination
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Release builds are the default; tests are opt-in. To build only the path you
need, use `-DBUILD_RGBD=OFF` for stereo only, or `-DBUILD_TWO_VIEW=OFF` for
RGB-D only. The RGB-D build requests OpenCV core, image processing, and
image codecs, plus their transitive dependencies. Feature detection and stereo
geometry modules are linked only by the stereo path. OpenCV 5 builds avoid the
unused calibration and object-detection modules.
No additional runtime dependencies are needed.

The tests cover calibrated stereo with a 1 m baseline and 20/25/35/40/60-pixel
disparities, foreground depth preservation, ridged-terrain rejection, decimal
grid boundaries, strict configuration parsing, large pixel strides, thin-grid
exports, and JSON escaping. On Unix, an additional CLI test runs the supplied
image pair in demo mode and parses its output metadata.

The dense path verifies feature geometry, aligns yaw, and reconstructs directly
from stereo disparity. It avoids a discarded sparse reconstruction, uses
single-pass grayscale warps, and stores accepted points once. IDW operates in
place using measured neighbors only. Fuzzy scores are calculated only for
quantized inputs encountered in the map, using the same rule tables and
membership functions.

For C++ callers, link `metric_stereo` for `reconstructTwoView` and
`metric_mapping` for RGB-D/terrain functions. The discarded sparse path's
`maximum_reprojection_error_px`, `minimum_parallax_degrees`, and
`maximum_distance_factor` settings were removed. Stereo metadata now names
its feature count `geometric_inliers` instead of `pose_inliers` and includes
the terrain grid's X/Y origin.

## Run

```bash
./build/two_view image1.png image2.png fx fy cx cy baseline_m
```

Every camera value must be numeric and must belong to the saved image size.
For example, do not use a 1280 x 720 principal point with 512 x 512 images.
Recalibrate after changing the aspect ratio, cropping, or resizing.

## Synthetic demo mode

When only uncalibrated images are available, use the explicitly synthetic mode
to test PLY/DEM/orthomosaic/IDW output and visualization:

```bash
./build/two_view --demo \
  00001.png \
  00002.png
```

This mode aligns the two images for color, then creates a deterministic
appearance-derived height surface with one point for every input image pixel.
For a 512 x 512 image, the completed demo PLY therefore contains 262,144
points. It writes `demo_output/` and marks both PLY headers and
`metadata.json` as synthetic. It is useful for exercising the software and
viewing terrain/rocks together, but it is **not** stereo depth and must not be
used for measurements or landing decisions. Its landing maps are generated
only to test the processing chain and are marked `demo_only`.

## Landing-site analysis

After creating the registered DEM, the program performs the paper's terrain
decision stages:

1. Ten iterations of edge-preserving anisotropic diffusion smooth the DEM.
2. A metric Sobel derivative produces local slope in degrees. The maximum
   measured gradient to an adjacent cell provides a conservative lower bound,
   so alternating ridges and sharp steps cannot cancel out.
3. Local roughness is the standard deviation of
   `original DEM - smoothed DEM` over the UAV footprint.
4. Excessive slope, excessive roughness, missing terrain, and incomplete UAV
   footprints become hazards.
5. A Euclidean distance transform produces nearest-hazard clearance.
6. The paper's three fuzzy rule tables produce terrain safety, distance, and
   global safety indices.
7. Cells satisfying every hard constraint and the global-index threshold are
   candidates. The highest global index is selected, with terrain safety and
   distance index as tie breakers.

The defaults come from Table VIII of the paper:

- Maximum slope: `10 degrees`.
- Maximum roughness: `0.02 m`.
- Minimum nearest-hazard distance: `1.5 m`.
- UAV bounding-box diagonal: `1.0 m`.
- Minimum global safety index: `44.8%`.

These values are examples from the paper, not universal flight limits. Change
`LandingAnalysisConfig` to match the dimensions and limits of the real UAV.
The program treats rocks and ground as one geometric surface: rock boundaries
and protrusions become unsafe through slope, roughness, and hazard clearance.

The local world frame is gravity-aligned under the nadir capture contract:

- `+X`: image-right from camera 1.
- `+Y`: image-up from camera 1.
- `+Z`: upward.
- Camera 1/UAV position: `(0, 0, 0)`.
- Units: metres, provided `baseline_m` and intrinsics are correct.

## Outputs

A successful run writes:

- `pointcloud_raw.ply`: measured dense StereoSGBM XYZRGB points.
- `pointcloud.ply`: homogeneous 0.1 m XY terrain grid containing measured and
  bounded-IDW interpolated points together.
- `dem.csv`: registered `+Z` elevation grid in metres.
- `orthomosaic.png`: registered vertical-view color grid.
- `validity_mask.png`: `255` measured, `127` interpolated, `0` unknown.
- `dem_preview.png`: colorized DEM.
- `disparity.png`: accepted dense stereo disparities.
- `matches.jpg`: final parallel-alignment feature inliers.
- `debug_overview.png`: orthomosaic, DEM, validity, and side-cloud views.
- `metadata.json`: intrinsics, baseline, quality diagnostics, counts, and map
  dimensions.
- `smoothed_dem.png`: anisotropic-diffusion DEM used for slope estimation.
- `slope.png`: local slope visualization, scaled from zero to twice the limit.
- `roughness.png`: local roughness, scaled from zero to twice the limit.
- `hazard_mask.png`: `255` hazard and `0` geometrically admissible terrain.
- `nearest_hazard_distance.png`: distance to the nearest hazard.
- `safety_index.png`: fuzzy slope/roughness safety index.
- `distance_index.png`: fuzzy UAV-distance/hazard-clearance index.
- `global_safety_index.png`: combined fuzzy landing suitability.
- `best_landing_site.png`: orthomosaic with hazards in red and the selected
  UAV footprint circled in green.
- `landing_analysis_overview.png`: all landing-analysis maps in one image.
- `landing_site.json`: constraints, counts, UAV position, and selected XYZ and
  scores. Demo output explicitly states that its geometry is synthetic.

Rocks, protrusions, and terrain remain in the same colored geometric cloud.
The terrain grid keeps the highest `+Z` measured point in each XY cell, so
above-ground items are not discarded. They are not assigned semantic class
labels at this stage.

View a successful cloud with MeshLab:

```bash
brew install --cask meshlab
open -a MeshLab pointcloud.ply
```

## Existing RGB-D path

The earlier modular metric RGB-D implementation remains available as
`metric_mapper`. It accepts registered metric depth and supplied poses through
`configs/reconstruction.yaml`; it is independent of the two-image stereo
executable.
