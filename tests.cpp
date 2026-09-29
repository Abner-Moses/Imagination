#include "src/internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
// Regression tests: small known examples protect each stage during edits.

// ============================================================================
// Test helpers
// ============================================================================


namespace metric_mapping::tests {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path(std::filesystem::temp_directory_path() /
               ("mapping_regression_" + std::to_string(
                   std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(path);
    }
    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::filesystem::path path;
};

inline DepthConfig metricDepthConfig()
{
    DepthConfig config;
    config.unit = DepthUnit::Meters;
    config.min_depth_m = 0.1;
    config.max_depth_m = 100.0;
    config.pixel_stride = 1;
    config.invalid_values = {0.0};
    return config;
}

inline void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

inline void requireNear(double actual,
                 double expected,
                 double tolerance,
                 const std::string& message)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", received " + std::to_string(actual));
    }
}

inline TerrainGrid makeFlatGrid(int width, int height, double resolution_m)
{
    TerrainGrid grid;
    grid.width = width;
    grid.height = height;
    grid.resolution_m = resolution_m;
    grid.minimum_x_m = -0.5 * (width - 1) * resolution_m;
    grid.maximum_y_m = 0.5 * (height - 1) * resolution_m;
    const std::size_t cells =
        static_cast<std::size_t>(width) * height;
    grid.elevation_m.assign(cells, 0.0F);
    grid.color_bgr.assign(cells, cv::Vec3b(60, 90, 120));
    grid.validity.assign(cells, 255);
    return grid;
}

inline void requireThrows(const std::function<void()>& action, const std::string& message)
{
    try { action(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

using TestCase = std::pair<std::string, std::function<void()>>;
std::vector<TestCase> rgbdTests();
std::vector<TestCase> terrainTests();
std::vector<TestCase> configTests();
std::vector<TestCase> outputTests();
std::vector<TestCase> ultrasonicTests();
#ifdef HAVE_VISUAL_FEATURES
std::vector<TestCase> visualFeatureTests();
#endif
#ifdef HAVE_MOTION
std::vector<TestCase> motionTests();
#endif
#ifdef HAVE_TWO_VIEW
std::vector<TestCase> stereoTests();
#endif
} // namespace metric_mapping::tests

// ============================================================================
// Rgbd tests
// ============================================================================

namespace metric_mapping::tests {
namespace {
void testSyntheticPlanarDepth()
{
    const CameraIntrinsics camera{3, 2, 2.0, 2.0, 1.0, 0.5};
    const cv::Mat rgb(2, 3, CV_8UC3, cv::Scalar(5, 10, 15));
    const cv::Mat depth(2, 3, CV_32F, cv::Scalar(2.0F));
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());

    require(result.world_points.size() == 6,
            "Planar frame must emit every valid pixel");
    for (const ColoredPoint& point : result.world_points) {
        requireNear(point.z, 2.0, 1e-6,
                    "Synthetic constant-depth plane must remain at Z=2 m");
    }
}

void testPixelBackProjection()
{
    const CameraIntrinsics camera{10, 10, 2.0, 4.0, 1.0, 2.0};
    const cv::Vec3d point = backProjectPixel(3.0, 4.0, 2.0, camera);
    requireNear(point[0], 2.0, 1e-12, "Back-projected X");
    requireNear(point[1], 1.0, 1e-12, "Back-projected Y");
    requireNear(point[2], 2.0, 1e-12, "Back-projected Z");
}

void testWorldTransformAndConvention()
{
    cv::Matx44d camera_to_world = cv::Matx44d::eye();
    camera_to_world(0, 0) = 0.0;
    camera_to_world(0, 1) = -1.0;
    camera_to_world(1, 0) = 1.0;
    camera_to_world(1, 1) = 0.0;
    camera_to_world(0, 3) = 1.0;
    camera_to_world(1, 3) = 2.0;
    camera_to_world(2, 3) = 3.0;

    const cv::Vec3d world =
        transformPoint(camera_to_world, cv::Vec3d(1.0, 0.0, 0.0));
    requireNear(world[0], 1.0, 1e-12, "World transform X");
    requireNear(world[1], 3.0, 1e-12, "World transform Y");
    requireNear(world[2], 3.0, 1e-12, "World transform Z");

    const cv::Matx44d world_to_camera =
        invertRigidTransform(camera_to_world);
    const cv::Matx44d recovered = cameraToWorldTransform(
        world_to_camera, PoseConvention::WorldToCamera);
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            requireNear(recovered(row, column),
                        camera_to_world(row, column), 1e-12,
                        "T_CW inversion must recover T_WC");
        }
    }
}

void testMultipleFrameOverlap()
{
    const CameraIntrinsics camera{3, 1, 1.0, 1.0, 1.0, 0.0};
    const cv::Mat rgb(1, 3, CV_8UC3, cv::Scalar(30, 20, 10));
    cv::Mat depth_first(1, 3, CV_32F, cv::Scalar(0.0F));
    cv::Mat depth_second(1, 3, CV_32F, cv::Scalar(0.0F));
    depth_first.at<float>(0, 1) = 2.0F;
    depth_second.at<float>(0, 0) = 2.0F;

    const FrameResult first = backProjectFrame(
        rgb, depth_first, camera, metricDepthConfig(), cv::Matx44d::eye());
    cv::Matx44d second_pose = cv::Matx44d::eye();
    second_pose(0, 3) = 2.0;
    const FrameResult second = backProjectFrame(
        rgb, depth_second, camera, metricDepthConfig(), second_pose);

    require(first.world_points.size() == 1 &&
                second.world_points.size() == 1,
            "Each synthetic frame must emit one point");
    requireNear(first.world_points[0].x, second.world_points[0].x, 1e-6,
                "Two views must overlap in world X");
    requireNear(first.world_points[0].y, second.world_points[0].y, 1e-6,
                "Two views must overlap in world Y");
    requireNear(first.world_points[0].z, second.world_points[0].z, 1e-6,
                "Two views must overlap in world Z");

    VoxelGridAccumulator fusion(0.1, 10);
    fusion.add(first.world_points);
    fusion.add(second.world_points);
    require(fusion.points().size() == 1,
            "Overlapping world observations must fuse into one voxel");
}

void testRgbPreservation()
{
    const CameraIntrinsics camera{1, 1, 1.0, 1.0, 0.0, 0.0};
    const cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(30, 20, 10));
    const cv::Mat depth(1, 1, CV_32F, cv::Scalar(1.0F));
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());
    require(result.world_points[0].r == 10 &&
                result.world_points[0].g == 20 &&
                result.world_points[0].b == 30,
            "PLY RGB must preserve the corresponding BGR source pixel");
}

void testMetricScale()
{
    const CameraIntrinsics camera{2, 1, 1.0, 1.0, 0.0, 0.0};
    const cv::Mat rgb(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat depth(1, 2, CV_16U, cv::Scalar(1000));
    DepthConfig millimeter_depth = metricDepthConfig();
    millimeter_depth.unit = DepthUnit::Millimeters;
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, millimeter_depth, cv::Matx44d::eye());
    const ColoredPoint& first = result.world_points[0];
    const ColoredPoint& second = result.world_points[1];
    const double distance = std::sqrt(
        std::pow(second.x - first.x, 2) +
        std::pow(second.y - first.y, 2) +
        std::pow(second.z - first.z, 2));
    requireNear(distance, 1.0, 1e-6,
                "1000 mm synthetic object must measure 1 metre");
}

void testInvalidDepthRejection()
{
    const CameraIntrinsics camera{2, 2, 1.0, 1.0, 0.5, 0.5};
    const cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(2, 2, CV_32F);
    depth.at<float>(0, 0) = 0.0F;
    depth.at<float>(0, 1) = std::numeric_limits<float>::quiet_NaN();
    depth.at<float>(1, 0) = std::numeric_limits<float>::infinity();
    depth.at<float>(1, 1) = 1.0F;
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());
    require(result.diagnostics.valid_depth_pixels == 1 &&
                result.diagnostics.invalid_depth_pixels == 3 &&
                result.world_points.size() == 1,
            "NaN, infinity, zero, and invalid depth must be rejected");
}

void testStrideBounds()
{
    const CameraIntrinsics camera{2, 2, 1, 1, 0.5, 0.5};
    const cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(0));
    const cv::Mat depth(2, 2, CV_32F, cv::Scalar(1));
    auto config = metricDepthConfig();
    for (int stride : {65536, std::numeric_limits<int>::max()}) {
        config.pixel_stride = stride;
        const auto result = backProjectFrame(rgb, depth, camera, config, cv::Matx44d::eye());
        require(result.world_points.size() == 1, "Large positive strides must emit the first pixel");
    }
    for (int stride : {0, -1}) {
        config.pixel_stride = stride;
        requireThrows([&] { backProjectFrame(rgb, depth, camera, config, cv::Matx44d::eye()); },
                      "Nonpositive strides must be rejected");
    }
}
} // namespace

std::vector<TestCase> rgbdTests()
{
    return {
        {"synthetic planar depth", testSyntheticPlanarDepth},
        {"pixel back-projection", testPixelBackProjection},
        {"world transform and T_CW inversion", testWorldTransformAndConvention},
        {"multiple-frame overlap", testMultipleFrameOverlap},
        {"RGB preservation", testRgbPreservation},
        {"metric scale", testMetricScale},
        {"invalid depth rejection", testInvalidDepthRejection},
        {"pixel stride bounds", testStrideBounds}
    };
}
} // namespace metric_mapping::tests

// ============================================================================
// Terrain tests
// ============================================================================

namespace metric_mapping::tests {
namespace {
void testDemAndOrthomosaicRegistration()
{
    const std::vector<ColoredPoint> points{
        {0.0F, 0.0F, 1.0F, 10, 20, 30},
        {1.0F, 0.0F, 2.0F, 40, 50, 60}};
    const TerrainGrid grid = createTerrainGrid(points, 1.0, 100);
    require(grid.width == 2 && grid.height == 1,
            "Known DEM must have expected dimensions");
    requireNear(grid.elevation_m[grid.index(0, 0)], 1.0, 1e-6,
                "DEM first elevation");
    requireNear(grid.elevation_m[grid.index(0, 1)], 2.0, 1e-6,
                "DEM second elevation");
    const cv::Vec3b first_bgr = grid.color_bgr[grid.index(0, 0)];
    require(first_bgr == cv::Vec3b(30, 20, 10),
            "Orthomosaic color must stay registered with DEM cell");
}

void testPositiveZElevationAndIdwMask()
{
    const std::vector<ColoredPoint> stacked{
        {0.0F, 0.0F, 1.0F, 10, 10, 10},
        {0.0F, 0.0F, 3.0F, 30, 30, 30}};
    const TerrainGrid surface = createTerrainGrid(stacked, 1.0, 10);
    requireNear(surface.elevation_m[0], 3.0, 1e-6,
                "Increasing elevation must always be +Z");

    TerrainGrid grid;
    grid.width = 3;
    grid.height = 1;
    grid.resolution_m = 1.0;
    grid.minimum_x_m = 0.0;
    grid.maximum_y_m = 0.0;
    grid.elevation_m = {1.0F,
                        std::numeric_limits<float>::quiet_NaN(), 3.0F};
    grid.color_bgr = {cv::Vec3b(10, 10, 10), cv::Vec3b(0, 0, 0),
                      cv::Vec3b(30, 30, 30)};
    grid.validity = {255, 0, 255};
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 1.1;
    idw.minimum_neighbors = 2;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 1.1;
    interpolateIdw(grid, idw);
    requireNear(grid.elevation_m[1], 2.0, 1e-6,
                "IDW must interpolate the centered elevation");
    require(grid.validity[1] == 127,
            "Interpolated cells must remain distinguishable from measured");

    const std::vector<ColoredPoint> completed = terrainGridPoints(grid);
    require(completed.size() == 3,
            "Completed terrain cloud must contain measured and IDW cells");
    requireNear(completed[1].x, 1.0, 1e-6,
                "Completed cloud must preserve grid X registration");
    requireNear(completed[1].z, 2.0, 1e-6,
                "Completed cloud must preserve interpolated elevation");

    grid.width = 5;
    grid.elevation_m.assign(5, std::numeric_limits<float>::quiet_NaN());
    grid.color_bgr.assign(5, cv::Vec3b(0, 0, 0));
    grid.validity.assign(5, 0);
    grid.elevation_m[0] = 1.0F;
    grid.validity[0] = 255;
    idw.minimum_neighbors = 1;
    interpolateIdw(grid, idw);
    require(grid.validity[1] == 127 && grid.validity[2] == 0,
            "In-place IDW must never propagate newly interpolated cells");
}

void testPaperStyleLandingAnalysis()
{
    TerrainGrid grid = makeFlatGrid(121, 121, 0.1);
    const int center_row = grid.height / 2;
    const int center_column = grid.width / 2;
    for (int row = center_row - 4; row <= center_row + 4; ++row) {
        for (int column = center_column - 4;
             column <= center_column + 4; ++column) {
            grid.elevation_m[grid.index(row, column)] = 0.8F;
            grid.color_bgr[grid.index(row, column)] =
                cv::Vec3b(20, 20, 20);
        }
    }

    const LandingAnalysisConfig config;
    const LandingAnalysis result = analyzeLandingSites(
        grid, cv::Vec3d(0.0, 0.0, 10.0), config);
    require(result.hazard_mask.at<std::uint8_t>(
                center_row, center_column - 4) != 0 &&
                result.nearest_hazard_distance_m.at<float>(
                    center_row, center_column) <
                    config.minimum_hazard_distance_m,
            "Raised obstacle edge must be hazardous and its top must fail "
            "the required hazard clearance");
    require(result.admissible_cells > 0 && result.clearance_cells > 0,
            "Flat terrain away from the obstacle must remain admissible");
    require(result.best_site.found && result.candidate_cells > 0,
            "Known flat terrain must yield at least one landing candidate");
    require(result.best_site.slope_degrees <=
                config.maximum_slope_degrees &&
                result.best_site.roughness_m <=
                    config.maximum_roughness_m &&
                result.best_site.nearest_hazard_distance_m >=
                    config.minimum_hazard_distance_m &&
                result.best_site.global_safety_index >=
                    config.minimum_global_safety_index,
            "Selected site must satisfy every configured constraint");
}

void testMetricSlopeEstimation()
{
    TerrainGrid grid = makeFlatGrid(51, 51, 0.1);
    constexpr double rise_per_metre = 0.1;
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const double x = grid.minimum_x_m +
                             column * grid.resolution_m;
            grid.elevation_m[grid.index(row, column)] =
                static_cast<float>(rise_per_metre * x);
        }
    }
    LandingAnalysisConfig config;
    config.uav_footprint_diagonal_m = 0.3;
    const LandingAnalysis result = analyzeLandingSites(
        grid, cv::Vec3d(0.0, 0.0, 10.0), config);
    const double expected_degrees =
        std::atan(rise_per_metre) * 180.0 / CV_PI;
    requireNear(result.slope_degrees.at<float>(25, 25),
                expected_degrees, 0.2,
                "Sobel DEM slope must preserve metric rise/run");
}

void testDecimalGridBoundaries()
{
    std::vector<ColoredPoint> points;
    for (int row = 0; row < 512; ++row)
        for (int column = 0; column < 512; ++column)
            points.push_back({float((column - 256) * 0.05),
                              float((256 - row) * 0.05),
                              float(row * 512 + column), 1, 2, 3});
    const auto grid = createTerrainGrid(points, 0.05, points.size());
    const auto statistics = computeGridStatistics(grid);
    require(grid.width == 512 && grid.height == 512 &&
            statistics.measured_cells == points.size() && statistics.unknown_cells == 0,
            "A complete decimal lattice must not manufacture holes");
    for (std::size_t index = 0; index < points.size(); ++index)
        requireNear(grid.elevation_m[index], points[index].z, 0.0,
                    "Every original lattice measurement must survive");
    const auto interior = createTerrainGrid(
        {{0, 0, 1, 0, 0, 0}, {0.049F, 0, 2, 0, 0, 0}, {0.1F, 0, 3, 0, 0, 0}}, 0.05, 10);
    requireNear(interior.elevation_m[0], 2, 0, "Ordinary points must retain floor-based binning");
}

void testRidgedTerrainRejection()
{
    for (bool checkerboard : {false, true}) {
        auto grid = makeFlatGrid(121, 121, 0.1);
        for (int row = 0; row < grid.height; ++row)
            for (int column = 0; column < grid.width; ++column)
                grid.elevation_m[grid.index(row, column)] =
                    ((column + (checkerboard ? row : 0)) % 2) ? 0.3F : 0.0F;
        const auto result = analyzeLandingSites(grid, {0, 0, 10});
        require(!result.best_site.found && result.candidate_cells == 0 &&
                result.hazard_mask.at<std::uint8_t>(60, 60) == 255,
                "Measured 30 cm ridges must be hazards despite Sobel cancellation");
    }
}
} // namespace

std::vector<TestCase> terrainTests()
{
    return {
        {"DEM and orthomosaic registration", testDemAndOrthomosaicRegistration},
        {"+Z elevation and IDW validity", testPositiveZElevationAndIdwMask},
        {"paper-style landing analysis", testPaperStyleLandingAnalysis},
        {"metric DEM slope estimation", testMetricSlopeEstimation},
        {"decimal grid boundaries", testDecimalGridBoundaries},
        {"ridged terrain rejection", testRidgedTerrainRejection}
    };
}
} // namespace metric_mapping::tests

// ============================================================================
// Config tests
// ============================================================================

#include <opencv2/imgcodecs.hpp>

namespace metric_mapping::tests {
namespace {
void testStrictConfigParsing()
{
    TemporaryDirectory directory;
    cv::imwrite((directory.path / "image.png").string(),
                cv::Mat(1, 1, CV_16U, cv::Scalar(1000)));
    const std::string valid = R"(%YAML:1.0
---
world_frame: "map_z_up"
images_are_rectified: 1
depth_registered_to_rgb: 1
pose_convention: "camera_to_world"
output_directory: "output"
camera: {width: 1, height: 1, fx: 2.0, fy: 2.0, cx: 0.0, cy: 0.0}
depth: {unit: "millimeters", min_depth_m: 0.1, max_depth_m: 100.0, pixel_stride: 1, invalid_values: [0, 65535]}
fusion: {voxel_size_m: 0.1, max_raw_points: 100, max_voxels: 100}
map:
  resolution_m: 0.1
  max_cells: 100
  idw: {enabled: 1, search_radius_m: 0.3, minimum_neighbors: 3, power: 2.0, maximum_interpolation_distance_m: 0.3}
uav_position_world_m: [0.0, 0.0, 10.0]
frames:
  - rgb: "image.png"
    depth: "image.png"
    pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
)";
    const auto path = directory.path / "config.yaml";
    const auto read = [&](const std::string& yaml) {
        { std::ofstream output(path); output << yaml; }
        return loadConfig(path);
    };
    requireNear(read(valid).camera.fx, 2.0, 0.0, "Valid configuration must load");
    const std::string sensor = "\nultrasonic: {distance_m: 10.0, maximum_error_m: 0.2, "
        "direction: world_down, sensor_offset_world_m: [0, 0, 0]}\n";
    requireNear(read(valid + sensor).ultrasonic->distance_m, 10, 0,
                "RGB-D must load optional sensor measurement");
    auto missing_position = valid + sensor;
    const std::string position = "uav_position_world_m: [0.0, 0.0, 10.0]\n";
    missing_position.erase(missing_position.find(position), position.size());
    requireThrows([&] { read(missing_position); }, "Sensor reading requires UAV position");
    for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
             {"fx: 2.0", "fx: \"not calibrated\""},
             {"fx: 2.0", "fx: .Inf"},
             {"images_are_rectified: 1", "images_are_rectified: 0.6"},
             {"width: 1", "width: 1.4"},
             {"width: 1", "width: 1.0e100"},
             {"pixel_stride: 1", "pixel_stride: 0"},
             {"max_raw_points: 100", "max_raw_points: 1.0e100"},
             {"max_raw_points: 100", "max_raw_points: 18446744073709551616.0"},
             {"max_cells: 100", "max_cells: 2.5"},
             {"[0, 65535]", "[0, \"invalid\"]"},
             {"[0.0, 0.0, 10.0]", "[0.0, \"invalid\", 10.0]"},
             {"pose: [1, 0", "pose: [\"invalid\", 0"}}) {
        std::string yaml = valid;
        yaml.replace(yaml.find(replacement.first), replacement.first.size(), replacement.second);
        requireThrows([&] { read(yaml); }, "Must reject " + replacement.second);
    }
}
} // namespace

std::vector<TestCase> configTests()
{
    return {
        {"strict configuration parsing", testStrictConfigParsing}
    };
}
} // namespace metric_mapping::tests

// ============================================================================
// Output tests
// ============================================================================

#include <opencv2/imgcodecs.hpp>

namespace metric_mapping::tests {
namespace {
void testOutputSerialization()
{
    TemporaryDirectory temporary;
    const auto& directory = temporary.path;

    const std::vector<ColoredPoint> points{
        {0.0F, 0.0F, 1.0F, 255, 0, 0},
        {1.0F, 0.0F, 2.0F, 0, 255, 0}};
    const CloudBounds bounds = computeBounds(points);
    const TerrainGrid grid = createTerrainGrid(points, 1.0, 100);
    const GridStatistics statistics = computeGridStatistics(grid);
    ReconstructionConfig config;
    config.camera = {2, 1, 1.0, 1.0, 0.0, 0.0};
    config.depth = metricDepthConfig();
    config.fusion.voxel_size_m = 0.1;
    config.map.resolution_m = 1.0;
    config.world_frame = "map_z_up";

    writePly(directory / "cloud.ply", points);
    writeDemCsv(directory / "dem.csv", grid);
    writeTerrainImages(directory, grid);
    const LandingAnalysisConfig landing_config;
    const cv::Vec3d uav_position(0.0, 0.0, 3.0);
    const LandingAnalysis landing = analyzeLandingSites(
        grid, uav_position, landing_config);
    writeLandingAnalysis(directory, grid, landing, landing_config,
                         uav_position, false);
    writeDebugVisualization(directory / "debug.png", points, bounds, grid,
                            {cv::Vec3d(0.0, 0.0, 2.0)}, cv::Vec3d(0.0, 0.0, 3.0));
    writeMetadata(directory / "metadata.json", config, points.size(), points,
                  bounds, grid, statistics,
                  {cv::Vec3d(0.0, 0.0, 2.0)});

    for (const std::string& filename :
         {"cloud.ply", "dem.csv", "orthomosaic.png",
          "validity_mask.png", "slope.png", "roughness.png",
          "hazard_mask.png", "nearest_hazard_distance.png",
          "safety_index.png", "distance_index.png",
          "global_safety_index.png", "best_landing_site.png",
          "landing_analysis_overview.png", "landing_site.json",
          "debug.png", "metadata.json"}) {
        const std::filesystem::path path = directory / filename;
        require(std::filesystem::is_regular_file(path) &&
                    std::filesystem::file_size(path) > 0,
                "Expected serialized output: " + filename);
    }
}

void testThinGridExports()
{
    TemporaryDirectory directory;
    for (const auto size : {cv::Size(1000, 1), cv::Size(1, 1000)}) {
        const auto grid = makeFlatGrid(size.width, size.height, 0.1);
        const auto points = terrainGridPoints(grid);
        const auto analysis = analyzeLandingSites(grid, {0, 0, 10});
        writeLandingAnalysis(directory.path, grid, analysis, {}, {0, 0, 10}, false);
        writeDebugVisualization(directory.path / "debug.png", points,
                                computeBounds(points), grid, {}, std::nullopt);
        require(!cv::imread((directory.path / "debug.png").string()).empty(),
                "Thin grid debug visualization must be readable");
        require(!cv::imread((directory.path / "landing_analysis_overview.png").string()).empty(),
                "Thin grid landing visualization must be readable");
    }
}

void testJsonEscaping()
{
    require(escapeJson("a\"b\\c\n\t\r\b\f") ==
            "a\\\"b\\\\c\\u000a\\u0009\\u000d\\u0008\\u000c",
            "JSON must escape quotes, backslashes, and control characters");
    for (unsigned char character = 0; character < 32; ++character)
        require(escapeJson(std::string(1, char(character))).size() == 6,
                "Every JSON control character must be escaped");
    require(escapeJson("terrain-é.png") == "terrain-é.png", "UTF-8 must be preserved");
}
} // namespace

std::vector<TestCase> outputTests()
{
    return {
        {"output serialization", testOutputSerialization},
        {"thin grid exports", testThinGridExports},
        {"JSON escaping", testJsonEscaping}
    };
}
} // namespace metric_mapping::tests

// ============================================================================
// Ultrasonic tests
// ============================================================================


namespace metric_mapping::tests {
namespace {
void testRangeGatesLanding()
{
    auto grid = makeFlatGrid(81, 81, 0.1);
    const cv::Vec3d uav(0, 0, 10);
    const auto original = analyzeLandingSites(grid, uav);
    require(original.best_site.found && !original.ultrasonic,
            "Flat terrain must have a landing site without a sensor");
    UltrasonicMeasurement reading{10.0, 0.2, {0, 0, 0}};
    auto checked = analyzeLandingSites(grid, uav, {}, reading);
    require(checked.ultrasonic->consistent && checked.best_site.found &&
            checked.candidate_cells == original.candidate_cells &&
            checked.best_site.row == original.best_site.row &&
            checked.best_site.column == original.best_site.column,
            "Matching range must preserve landing selection");
    reading.distance_m = 5;
    checked = analyzeLandingSites(grid, uav, {}, reading);
    require(!checked.best_site.found && checked.candidate_cells == 0 &&
            std::string(checked.ultrasonic->status()) == "mismatch",
            "Range mismatch must withhold landing recommendation");
    reading.distance_m = 10;
    for (const std::uint8_t validity : {0, 127}) {
        grid.validity[grid.index(40, 40)] = validity;
        checked = analyzeLandingSites(grid, uav, {}, reading);
        require(!checked.best_site.found && checked.candidate_cells == 0 &&
                !checked.ultrasonic->mapped_distance_m,
                "Unknown and interpolated ground must not corroborate a reading");
    }
}

void testRangeGeometryAndValidation()
{
    auto grid = makeFlatGrid(11, 11, 1);
    grid.elevation_m[grid.index(3, 6)] = 2;
    UltrasonicMeasurement reading{7.5, 0.0, {1, 2, -0.5}};
    auto check = checkGroundRange(grid, {0, 0, 10}, reading);
    require(check.consistent, "World XYZ mounting offset must select the correct cell and height");
    requireNear(*check.mapped_distance_m, 7.5, 0, "Distance must be sensor height minus ground height");
    reading.distance_m = 7.75;
    reading.maximum_error_m = 0.25;
    require(checkGroundRange(grid, {0, 0, 10}, reading).consistent,
            "Tolerance boundary must be inclusive");
    reading.distance_m = 7.751;
    require(!checkGroundRange(grid, {0, 0, 10}, reading).consistent,
            "Reading beyond tolerance must fail");
    for (const cv::Vec3d& position : {cv::Vec3d(100, 0, 10), cv::Vec3d(0, 0, 1)})
        require(!checkGroundRange(grid, position, reading).mapped_distance_m,
                "Outside map or ground above sensor must be unavailable");
    for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        reading.distance_m = invalid;
        requireThrows([&] { checkGroundRange(grid, {0, 0, 10}, reading); },
                      "Invalid distances must be rejected");
    }
    reading.distance_m = 7.5;
    reading.maximum_error_m = -0.1;
    requireThrows([&] { validateUltrasonicMeasurement(reading); }, "Reject negative tolerance");
    reading.maximum_error_m = 0.2;
    reading.sensor_offset_world_m[0] = std::numeric_limits<double>::infinity();
    requireThrows([&] { validateUltrasonicMeasurement(reading); }, "Reject nonfinite offsets");
}

void testSensorConfigAndOutput()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "sensor.yaml";
    const std::string yaml = "%YAML:1.0\n---\nultrasonic:\n  distance_m: 10.0\n"
        "  maximum_error_m: 0.2\n  direction: world_down\n"
        "  sensor_offset_world_m: [0, 0, 0]\n";
    const auto read = [&](const std::string& text) {
        { std::ofstream output(path); output << text; }
        return loadUltrasonicConfig(path);
    };
    const auto reading = read(yaml);
    requireNear(reading.distance_m, 10, 0, "Sensor YAML distance");
    for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
             {"10.0", "0.0"}, {"10.0", "\"ten\""}, {"10.0", ".Inf"},
             {"0.2", "-1.0"}, {"world_down", "camera_down"},
             {"[0, 0, 0]", "[0, 0]"}, {"distance_m:", "range:"}}) {
        auto invalid = yaml;
        invalid.replace(invalid.find(replacement.first), replacement.first.size(), replacement.second);
        requireThrows([&] { read(invalid); }, "Reject invalid sensor YAML");
    }
    auto grid = makeFlatGrid(11, 11, 1);
    for (const std::string status : {"consistent", "mismatch", "ground_unavailable"}) {
        auto input = reading;
        if (status == "mismatch") input.distance_m = 5;
        if (status == "ground_unavailable") grid.validity[grid.index(5, 5)] = 0;
        const auto analysis = analyzeLandingSites(grid, {0, 0, 10}, {}, input);
        writeLandingAnalysis(directory.path, grid, analysis, {}, {0, 0, 10}, false);
        cv::FileStorage json((directory.path / "landing_site.json").string(),
                             cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        require(json.isOpened() && std::string(json["ultrasonic"]["status"]) == status,
                "Range status must round-trip through JSON");
        if (status != "ground_unavailable")
            requireNear(double(json["ultrasonic"]["mapped_distance_m"]), 10, 0,
                        "Export mapped distance");
    }
}
} // namespace
std::vector<TestCase> ultrasonicTests()
{
    return {{"ultrasonic landing gate", testRangeGatesLanding},
            {"ultrasonic geometry and validation", testRangeGeometryAndValidation},
            {"ultrasonic configuration and JSON", testSensorConfigAndOutput}};
}
} // namespace metric_mapping::tests

// ============================================================================
// Stereo tests
// ============================================================================

#ifdef HAVE_TWO_VIEW
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace metric_mapping::tests {
namespace {
void testDenseTwoViewStereo(int foreground_disparity,
                            double yaw_degrees = 0.0, double fy = 200.0)
{
    TemporaryDirectory temporary;
    const auto& directory = temporary.path;

    constexpr int width = 320;
    constexpr int height = 240;
    cv::Mat first(height, width, CV_8UC3);
    cv::RNG random(123456U);
    random.fill(first, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(first, first, cv::Size(3, 3), 0.5);

    cv::Mat second(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const bool foreground = column >= 105 && column < 215 &&
                                    row >= 70 && row < 170;
            const int disparity = foreground ? foreground_disparity : 20;
            const int second_column = column - disparity;
            if (second_column >= 0 && second_column < width) {
                second.at<cv::Vec3b>(row, second_column) =
                    first.at<cv::Vec3b>(row, column);
            }
        }
    }

    const CameraIntrinsics camera{width, height, 200.0, fy, 159.5, 119.5};
    if (yaw_degrees != 0.0) {
        const double angle = yaw_degrees * CV_PI / 180.0;
        const cv::Matx33d rotation(std::cos(angle), -std::sin(angle), 0,
                                   std::sin(angle), std::cos(angle), 0, 0, 0, 1);
        const cv::Matx33d intrinsic(camera.fx, 0, camera.cx,
                                    0, camera.fy, camera.cy, 0, 0, 1);
        cv::Mat rotated;
        cv::warpPerspective(second, rotated, intrinsic * rotation * intrinsic.inv(), second.size());
        second = rotated;
    }

    const std::filesystem::path first_path = directory / "first.png";
    const std::filesystem::path second_path = directory / "second.png";
    require(cv::imwrite(first_path.string(), first) &&
                cv::imwrite(second_path.string(), second),
            "Synthetic stereo images must be writable");

    const TwoViewResult result = reconstructTwoView(
        first_path, second_path, camera, 1.0);
    require(result.points.size() > 500,
            "Known stereo pair must create a dense cloud");

    std::vector<double> depths;
    depths.reserve(result.points.size());
    for (const ColoredPoint& point : result.points)
        depths.push_back(-point.z);
    std::sort(depths.begin(), depths.end());
    const double median_depth = depths[depths.size() / 2];
    requireNear(median_depth, 10.0, 1.0,
                "Known 20 px disparity must reconstruct near 10 m depth");
    std::size_t foreground_points = 0;
    std::size_t correct_foreground_points = 0;
    for (const auto& point : result.points) {
        const double z = -point.z;
        const double u = point.x * camera.fx / z + camera.cx;
        const double v = -point.y * camera.fy / z + camera.cy;
        if (u > 125 && u < 160 && v > 90 && v < 150) {
            ++foreground_points;
            if (std::abs(z - 200.0 / foreground_disparity) < 0.5)
                ++correct_foreground_points;
        }
    }
    require(foreground_points > 1000 &&
            correct_foreground_points > 0.95 * foreground_points,
            "Raised foreground must retain its true depth, not background depth");
}
} // namespace
std::vector<TestCase> stereoTests()
{
    return {{"dense calibrated two-view stereo", [] {
        for (int disparity : {25, 35, 40, 60})
            testDenseTwoViewStereo(disparity);
        testDenseTwoViewStereo(35, 3.0, 150.0);
    }}};
}
} // namespace metric_mapping::tests
#endif

// ============================================================================
// Motion tests
// ============================================================================

#ifdef HAVE_MOTION
#include <opencv2/imgproc.hpp>

namespace metric_mapping::tests {
namespace {
const CameraIntrinsics camera{320, 240, 240, 200, 159.5, 119.5};

MotionEstimate pair(double dx, double dy, double yaw = 0, double height = 2,
                    double scale = 1, OpticalFlowSettings settings = {})
{
    SparseFlowTracker tracker(camera, settings);
    const auto first = demo::motionTexture();
    cv::Mat second;
    demo::warpMotion(first, second, camera, dx, dy, yaw, scale);
    tracker.processFrame(first, 0, AltitudeSample{height * scale, 0});
    return tracker.processFrame(second, 0.1, AltitudeSample{height, 0.1});
}

void testTranslationAndScale()
{
    for (double height : {1.0, 3.0}) {
        const auto result = pair(3, -2, 0, height);
        require(result.valid && result.planar_velocity_mps.has_value(), "Translated frame must yield metric motion");
        requireNear(result.median_flow_px.x, 3, 0.08, "Pixel X translation");
        requireNear(result.median_flow_px.y, -2, 0.08, "Pixel Y translation");
        requireNear((*result.planar_displacement_m)[0], -height * 3 / camera.fx, 0.001, "Camera X sign and altitude scale");
        requireNear((*result.planar_displacement_m)[1], -height * 2 / camera.fy, 0.001, "Camera Y sign and altitude scale");
        requireNear((*result.planar_velocity_mps)[0], -height * 30 / camera.fx, 0.01, "Velocity uses seconds");
        require(result.translation_direction.has_value(), "Nonzero translation has direction");
    }
    const auto scaled = pair(2, 1, 0, 2, 1.04);
    require(scaled.valid && scaled.planar_displacement_m.has_value(), "Height change is supported with two readings");
    requireNear(scaled.image_scale, 1.04, 0.003, "Image scale from horizontal plane");
    requireNear((*scaled.planar_displacement_m)[0], -2 * 2 / camera.fx, 0.001, "Current height scales transform translation");
}

void testYawSeparation()
{
    const auto rotated = pair(0, 0, 0.025);
    require(rotated.valid, "Small yaw must be observable");
    requireNear(rotated.yaw_delta_rad, 0.025, 0.001, "Yaw sign in z-up frame");
    requireNear(cv::norm(*rotated.planar_displacement_m), 0, 0.001, "Pure yaw must not invent translation");
    const auto combined = pair(2, -1, 0.025);
    require(combined.valid, "Yaw plus translation must fit");
    requireNear((*combined.planar_displacement_m)[0],
                -2 * (std::cos(0.025)*2/camera.fx - std::sin(0.025)/camera.fy), 0.001,
                "Rotation-compensated translation with unequal focal lengths");
}

void testOutlierConsensus()
{
    using namespace motion_detail;
    std::vector<TrackedFeature> tracks;
    for (int row = 0; row < 8; ++row) for (int col = 0; col < 10; ++col) {
        const cv::Point2f p(float(20 + col * 28), float(20 + row * 27));
        const bool moving = col < 3;
        tracks.push_back({std::uint64_t(tracks.size()), 2, p, p + cv::Point2f(moving ? -6 : 3, moving ? 5 : -2)});
    }
    GeometryScratch scratch; scratch.reserve(100, 20);
    MotionEstimate result; result.tracked_points = tracks.size();
    estimateMotion(tracks, camera, {}, scratch, result);
    require(result.valid && result.accepted_points == 56, "30 percent moving tracks must not dominate geometry");
    requireNear(result.median_flow_px.x, 3, 1e-5, "Dominant consensus X");
    MotionEstimate assisted; assisted.tracked_points = tracks.size();
    estimateMotion(tracks, camera, {}, scratch, assisted, 0.0);
    require(assisted.valid && assisted.imu_used && assisted.accepted_points == 56,
            "Fixed IMU yaw retains moving-object consensus rejection");
    // Exercise the actual LK frontend with a moving textured patch as well.
    const auto first = demo::motionTexture(); cv::Mat second, moving;
    demo::warpMotion(first, second, camera, 3, -2);
    demo::warpMotion(first, moving, camera, -5, 4);
    moving(cv::Rect(50, 60, 65, 70)).copyTo(second(cv::Rect(50, 60, 65, 70)));
    SparseFlowTracker tracker(camera);
    tracker.processFrame(first, 0);
    const auto observed = tracker.processFrame(second, 0.1);
    require(observed.valid, "Minority moving image patch should be rejected");
    requireNear(observed.median_flow_px.x, 3, 0.1, "LK moving-patch rejection");
}

void testPersistenceAndTextureLoss()
{
    SparseFlowTracker tracker(camera);
    const auto first = demo::motionTexture();
    tracker.processFrame(first, 0, AltitudeSample{2, 0});
    const auto original = tracker.tracks(); cv::Mat frame;
    for (int i = 1; i <= 8; ++i) {
        demo::warpMotion(first, frame, camera, i * 0.4, i * -0.2);
        const auto result = tracker.processFrame(frame, i * 0.1, AltitudeSample{2, i * 0.1});
        require(result.valid && result.replenished_points == 0 && !result.detection_ran,
                "Healthy persistent tracks must not trigger corner detection");
        require(tracker.tracks().front().id == original.front().id && tracker.tracks().front().age == std::uint32_t(i + 1),
                "Track identity and age must survive");
        require(tracker.tracks().size() <= 100, "Feature budget is hard-limited");
    }
    tracker.reset();
    const cv::Mat blank(240, 320, CV_8U, cv::Scalar(100));
    for (int i = 0; i < 10; ++i) {
        const auto result = tracker.processFrame(blank, i * 0.1);
        require(!result.valid && !result.planar_displacement_m && tracker.tracks().empty(),
                "No texture must fail cleanly");
        require(result.detection_ran == (i % 5 == 0), "Textureless retries must be throttled");
    }
}

void testTimingAndAltitudeGates()
{
    SparseFlowTracker tracker(camera);
    const auto image = demo::motionTexture();
    tracker.processFrame(image, 0, AltitudeSample{2, 0});
    auto result = tracker.processFrame(image, 0.1);
    require(result.valid && !result.planar_displacement_m && !result.translation_direction,
            "Stationary image is valid without stale metric scale or a fabricated direction");
    result = tracker.processFrame(image, 0.3, AltitudeSample{2, 0});
    require(!result.planar_displacement_m, "Stale altitude must not produce metric output");
    requireThrows([&] { tracker.processFrame(image, 0.3); }, "Duplicate timestamp must fail");
    result = tracker.processFrame(image, 1.0, AltitudeSample{2, 1.0});
    require(!result.valid && std::string(result.status) == "frame_gap", "Large gap must reseed without velocity");
    const auto segment = result.segment_id;
    result = tracker.processFrame(image, 1.1, AltitudeSample{4, 1.1});
    require(result.valid && !result.planar_displacement_m && result.segment_id != segment,
            "Inconsistent altitude ratio must break metric integration");
    UltrasonicMeasurement sensor{1.9, 0.2, {0, 0, -0.1}};
    requireNear(altitudeFromUltrasonic(sensor, 1).height_m, 2, 1e-12, "Sensor mounting offset to camera height");
    requireThrows([&] { tracker.processFrame(cv::Mat(10, 10, CV_8U), 2); }, "Calibration mismatch must fail");
}

void testResizeAndBackwardTracking()
{
    const CameraIntrinsics input{640, 480, 480, 400, 319.5, 239.5};
    OpticalFlowSettings settings; settings.forward_backward = true;
    SparseFlowTracker tracker(input, settings);
    requireNear(tracker.workingCamera().cx, 159.5, 0, "Resize must scale pixel centers correctly");
    const auto first = demo::motionTexture({640, 480}); cv::Mat second;
    demo::warpMotion(first, second, input, 6, -4);
    tracker.processFrame(first, 0, AltitudeSample{2, 0});
    const auto result = tracker.processFrame(second, 0.1, AltitudeSample{2, 0.1});
    require(result.valid, "Resized forward-backward tracking");
    requireNear(result.median_flow_px.x, 3, 0.1, "Flow units are working pixels");
    requireNear((*result.planar_displacement_m)[0], -2 * 6 / input.fx, 0.001, "Resizing preserves metric scale");
    const CameraIntrinsics small{160, 120, 120, 100, 79.5, 59.5};
    require(SparseFlowTracker(small).workingCamera().width == 160, "Never upscale frames");
    settings.window_size = 14;
    requireThrows([&] { SparseFlowTracker invalid(input, settings); }, "Reject invalid window");
}

void testTriangulationAndSparseMap()
{
    const NadirPose first{{0, 0, 0}, 0}, second{{0.3, 0.1, 0}, 0.03};
    const cv::Vec3d truth(0.2, -0.1, -2);
    const auto project = [&](const NadirPose& pose) {
        const auto p = motion_detail::nadirRotation(pose.yaw_rad).t() * (truth - pose.position_m);
        return cv::Point2f(float(camera.fx * p[0] / p[2] + camera.cx), float(camera.fy * p[1] / p[2] + camera.cy));
    };
    const auto a = project(first), b = project(second);
    const auto point = triangulateTrack(a, b, camera, first, second);
    require(point.has_value(), "Well-conditioned pair triangulates");
    requireNear(cv::norm(point->world_m - truth), 0, 1e-5, "Ray intersection recovers known 3-D point");
    require(!triangulateTrack(a, a, camera, first, first), "Zero baseline must fail");
    const NadirPose tiny{{0.00001, 0, 0}, 0};
    require(!triangulateTrack(a, project(tiny), camera, first, tiny), "Tiny baseline must fail");
    require(!triangulateTrack(a, a + cv::Point2f(20, 0), camera, first, NadirPose{{0.3, 0, 0}, 0}),
            "Intersection behind cameras must fail");
    require(!triangulateTrack(a, a, camera, first, NadirPose{{0.3, 0, 0}, 0}), "Parallel rays must fail");
    require(!triangulateTrack(a, b + cv::Point2f(0, 20), camera, first, second), "Bad reprojection must fail");
    OpticalFlowSettings settings; settings.triangulation = true; settings.maximum_landmarks = 8;
    SparseFlowTracker tracker(camera, settings);
    const auto texture = demo::motionTexture(); cv::Mat image;
    std::size_t created = 0;
    for (int i = 0; i < 25; ++i) {
        demo::warpMotion(texture, image, camera, -i * 2, 0);
        const auto result = tracker.processFrame(image, i * 0.1, AltitudeSample{2, i * 0.1});
        created += result.new_landmarks;
        require(result.triangulation_attempts <= 4 && tracker.landmarks().size() <= 8,
                "Triangulation and map budgets must hold");
        if (i % 5) require(result.triangulation_attempts == 0, "Triangulation cadence");
    }
    require(created >= 4, "Persistent tracked points must reach sparse map");
    for (const auto& landmark : tracker.landmarks())
        requireNear(landmark.point.world_m[2], -2, 0.05, "Integrated sparse landmarks have correct depth");
    tracker.processFrame(image, 2.5);
    require(tracker.landmarks().empty(), "Lost scale clears landmarks rather than joining incompatible poses");
}

void testQualityAndRecovery()
{
    const cv::Mat blank(240, 320, CV_8U, cv::Scalar(100));
    const auto texture = demo::motionTexture();
    cv::Mat patch = blank.clone();
    texture(cv::Rect(10, 10, 50, 45)).copyTo(patch(cv::Rect(10, 10, 50, 45)));
    SparseFlowTracker tracker(camera);
    tracker.processFrame(patch, 0);
    require(!tracker.processFrame(patch, 0.1).valid, "Concentrated texture must not yield confident ego-motion");
    // Recovery must eventually seed the new scene, without stale metrics.
    MotionEstimate recovered;
    for (int i = 2; i <= 12; ++i)
        recovered = tracker.processFrame(texture, i * 0.1, AltitudeSample{2, i * 0.1});
    require(recovered.valid && recovered.planar_velocity_mps.has_value(), "Texture recovery restores estimates");
    require(!tracker.processFrame(texture, 1.3, AltitudeSample{2, 1.4}).planar_displacement_m,
            "Future range must not set metric scale");
    require(!tracker.processFrame(texture, 1.4, AltitudeSample{-2, 1.4}).planar_displacement_m,
            "Invalid range must not set metric scale");
    cv::Mat color; cv::cvtColor(texture, color, cv::COLOR_GRAY2BGR);
    require(tracker.processFrame(color, 1.5).valid, "BGR input uses grayscale tracking");
    cv::Mat alpha; cv::cvtColor(texture, alpha, cv::COLOR_GRAY2BGRA);
    require(tracker.processFrame(alpha, 1.6).valid, "BGRA input uses grayscale tracking");
    const auto debug = tracker.debugImage(recovered);
    require(debug.size() == texture.size() && debug.type() == CV_8UC3, "Explicit debug image");
}
} // namespace
std::vector<TestCase> motionTests()
{
    return {{"flow translation and altitude scaling", testTranslationAndScale},
            {"flow yaw separation", testYawSeparation},
            {"flow moving-object consensus", testOutlierConsensus},
            {"flow persistence and no texture", testPersistenceAndTextureLoss},
            {"flow timing and altitude gates", testTimingAndAltitudeGates},
            {"flow resizing and backward check", testResizeAndBackwardTracking},
            {"flow triangulation and bounded map", testTriangulationAndSparseMap},
            {"flow quality gates and recovery", testQualityAndRecovery}};
}
} // namespace metric_mapping::tests
#endif

// ============================================================================
// Visual Features tests
// ============================================================================

#ifdef HAVE_VISUAL_FEATURES
#include <opencv2/imgproc.hpp>

namespace metric_mapping::tests {
namespace {
VisualFeatureSettings selected(const std::string& names)
{
    VisualFeatureSettings s; s.features = selectVisualFeatures(names); return s;
}
void gradientsAndAblation()
{
    cv::Mat image(64, 96, CV_8UC3);
    for (int y=0; y<image.rows; ++y) for (int x=0; x<image.cols; ++x)
        image.at<cv::Vec3b>(y,x) = cv::Vec3b(x*2,x*2,x*2);
    VisualFeatureExtractor extractor(selected("gradients,edge_magnitude,edge_orientation"));
    const auto& f = extractor.extract(image,0);
    requireNear(f.gradient_x.at<float>(30,30),2.0/255,1e-6,"Normalized gradient");
    requireNear(f.gradient_y.at<float>(30,30),0,1e-6,"Vertical gradient");
    requireNear(f.edge_orientation_rad.at<float>(30,30),0,1e-5,"Gradient direction");
    require(f.edge_orientation_valid.at<uchar>(30,30)==255,"Defined orientation");
    const auto& blank = extractor.extract(cv::Mat(image.size(),CV_8UC3,cv::Scalar::all(80)),1);
    require(cv::countNonZero(blank.edge_orientation_valid)==0,"Flat image has no orientation");
    VisualFeatureExtractor none(selected("none"));
    const auto& n = none.extract(image,0);
    require(n.computed.none() && visualFeaturePlanes(n).empty(),"Ablation removes outputs");
    for (std::size_t i=1;i<visual_stage_count;++i) require(!n.stages[i].ran,"Disabled stages do not run");
}
void textureAndColor()
{
    cv::Mat stripes(64,96,CV_8UC3);
    for(int y=0;y<64;++y) for(int x=0;x<96;++x) stripes.at<cv::Vec3b>(y,x)=cv::Vec3b::all((x/8)%2?220:20);
    VisualFeatureExtractor texture(selected("texture_orientation,local_binary_texture"));
    const auto& f=texture.extract(stripes,0);
    require(f.texture_orientation_valid.at<uchar>(30,31)!=0,"Stripe orientation valid");
    requireNear(f.texture_orientation_rad.at<float>(30,31),CV_PI/2,1e-5,"Vertical stripe tangent");
    requireNear(f.texture_coherence.at<float>(30,31),1,1e-5,"Stripe coherence");
    require(f.local_binary_valid.at<uchar>(0,0)==0,"LBP border invalid");
    cv::Mat patch(64,96,CV_8UC3,cv::Scalar::all(100));
    patch.at<cv::Vec3b>(29,29)=cv::Vec3b::all(200);
    patch.at<cv::Vec3b>(29,30)=cv::Vec3b::all(0);
    const auto& p=texture.extract(patch,1);
    require(p.local_binary_texture.at<uchar>(30,30)==253,"LBP clockwise NW-first bit order");
    VisualFeatureExtractor color(selected("color_transitions,gradients"));
    cv::Mat step(64,96,CV_8UC3,cv::Scalar(0,0,150));
    step.colRange(48,96).setTo(cv::Scalar(0,76,0));
    const auto& c=color.extract(step,0);
    require(c.color_transition_magnitude.at<float>(30,47)>0.3,"Chromatic edge exposed");
    require(std::abs(c.gradient_x.at<float>(30,47))<0.005,"Nearly equal luminance edge");
    step.setTo(cv::Scalar::all(20)); step.colRange(48,96).setTo(cv::Scalar::all(220));
    require(cv::norm(color.extract(step,1).color_transition_magnitude)==0,"Achromatic change has no opponent transition");
}
void pointsAndBoundaries()
{
    cv::Mat image(240,320,CV_8UC3,cv::Scalar::all(0));
    cv::RNG rng(123);
    for(int y=15;y<230;y+=25) for(int x=15;x<310;x+=25)
        cv::circle(image,{x,y},5,cv::Scalar::all(rng.uniform(100,255)),-1);
    auto s=selected("corners,keypoints,contours,shape_boundaries"); s.maximum_points=40;
    s.maximum_contours=8; s.maximum_contour_points=100;
    VisualFeatureExtractor extractor(s); const auto& f=extractor.extract(image,0);
    require(!f.corners.empty() && f.corners.size()<=40,"Bounded corners");
    require(!f.keypoints.empty() && f.keypoints.size()<=40,"Bounded FAST points");
    require(!f.contours.empty() && f.contours.size()<=8 && f.contours_truncated,"Contour budget explicit");
    std::size_t points=0; for(const auto& c:f.contours) points+=c.size();
    require(points<=100 && cv::countNonZero(f.shape_boundaries)>0,"Bounded shape output");
    bool left=false,right=false; for(const auto& k:f.corners){left|=k.pt.x<100;right|=k.pt.x>220;}
    require(left&&right,"Spatially distributed corners");
}
void coordinatesAndExport()
{
    VisualFeatureSettings s; s.profiling=true;
    VisualFeatureExtractor extractor(s);
    cv::Mat image(480,640,CV_8UC3,cv::Scalar(40,80,120));
    const auto& f=extractor.extract(image,2);
    require(f.working_size==cv::Size(320,240),"Working resolution");
    requireNear(f.pixel_to_source(0,0),2,0,"Coordinate scale");
    requireNear(f.pixel_to_source(0,2),0.5,0,"Pixel center offset");
    require(!f.motion && !f.computed.test(std::size_t(VisualFeature::Motion)),"Still image does not invent motion");
    TemporaryDirectory dir; const auto file=dir.path/"features.yml.gz";
    writeVisualFeatures(file,f,s); cv::FileStorage in(file.string(),cv::FileStorage::READ);
    require(in.isOpened() && int(in["schema_version"])==1 && in["planes"].size()==10,"Structured export");
    cv::Mat gray; in["gray"]>>gray;
    require(cv::norm(gray,f.gray)==0,"Export preserves values");
    const auto canvas=renderVisualFeatures(image,f,"No verified temporal input");
    require(canvas.type()==CV_8UC3 && canvas.cols>=1200 && canvas.rows>=900,"Diagnostic panel layout");
    require(extractor.extract(cv::Mat(32,40,CV_8UC3,cv::Scalar::all(0)),3).working_size==cv::Size(40,32),"Never upscale");
    requireThrows([&]{extractor.extract(cv::Mat(),4);},"Reject empty image");
    requireThrows([&]{selectVisualFeatures("gradients,wrong");},"Reject unknown ablation");
    s.tensor_window=4; requireThrows([&]{VisualFeatureExtractor bad(s);},"Reject invalid tensor window");
}
void temporalIntegration()
{
#ifdef HAVE_MOTION
    const CameraIntrinsics camera{320,240,240,200,159.5,119.5};
    auto settings=selected("motion"); VisualFeatureExtractor extractor(settings,camera);
    const auto gray=demo::motionTexture(); cv::Mat first,second,warped;
    cv::cvtColor(gray,first,cv::COLOR_GRAY2BGR);
    demo::warpMotion(gray,warped,camera,3,-2); cv::cvtColor(warped,second,cv::COLOR_GRAY2BGR);
    extractor.extract(first,0,AltitudeSample{2,0});
    const auto& f=extractor.extract(second,0.1,AltitudeSample{2,0.1});
    require(f.motion && f.motion->valid && f.motion->planar_velocity_mps,"Integrated metric motion");
    requireNear(f.motion->median_flow_px.x,3,0.1,"Temporal flow uses previous frame");
    require(!f.tracks.empty() && f.gradient_x.empty(),"Motion ablation avoids spatial extraction");
    requireThrows([&]{extractor.extract(second,0.1);},"Reject repeated temporal timestamp");
    extractor.resetMotion(); require(!extractor.extract(first,0).motion->valid,"Reset clears temporal history");
    settings.motion.require_imu = true;
    VisualFeatureExtractor assisted(settings, camera);
    assisted.extract(first, 0, AltitudeSample{2,0}, ImuSample{});
    const auto& sensor_frame = assisted.extract(second, 0.1, AltitudeSample{2,0.1}, ImuSample{0,0,0,0.1});
    require(sensor_frame.motion && sensor_frame.motion->imu_used && sensor_frame.motion->planar_velocity_mps,
            "Pre-extraction forwards IMU and ultrasonic samples into temporal motion");
#endif
}
}
std::vector<TestCase> visualFeatureTests()
{
    return {{"visual gradients and ablation",gradientsAndAblation},{"visual texture and color",textureAndColor},
        {"visual points and boundaries",pointsAndBoundaries},{"visual coordinates and export",coordinatesAndExport},
        {"visual temporal integration",temporalIntegration}};
}
}
#endif

// ============================================================================
// Run every enabled test
// ============================================================================


using namespace metric_mapping::tests;

// Shared behavior must stay consistent across visual extraction and motion.
void testSharedCameraResize()
{
    using namespace metric_mapping;
    const CameraIntrinsics camera{641, 479, 470, 465, 320, 239};
    const auto size = detail::fitImageSize({camera.width, camera.height}, 320, 240);
    require(size == cv::Size(320, 239), "Aspect ratio and rounding for an odd-sized image");
    require(detail::fitImageSize({40, 32}, 320, 240) == cv::Size(40, 32), "Never upscale");
    const auto resized = detail::resizeCamera(camera, size);
    const auto transform = detail::pixelToSource({camera.width, camera.height}, size);
    const cv::Vec3d working_pixel(51.25, 83.5, 1);
    const auto source_pixel = transform * working_pixel;
    requireNear((working_pixel[0] - resized.cx) / resized.fx,
                (source_pixel[0] - camera.cx) / camera.fx, 1e-14, "Resize preserves the X camera ray");
    requireNear((working_pixel[1] - resized.cy) / resized.fy,
                (source_pixel[1] - camera.cy) / camera.fy, 1e-14, "Resize preserves the Y camera ray");
    requireThrows([] { detail::fitImageSize({0, 10}, 320, 240); }, "Reject empty source size");
    requireThrows([&] { detail::resizeCamera(camera, {0, 10}); }, "Reject empty working size");
}

#ifdef HAVE_MOTION
void testImuAndUltrasonicGeometry()
{
    using namespace metric_mapping;
    const CameraIntrinsics camera{320, 240, 240, 200, 159.5, 119.5};
    const auto first = demo::motionTexture();
    for (double height : {1.0, 3.0}) {
        OpticalFlowSettings settings; settings.require_imu = true;
        SparseFlowTracker tracker(camera, settings);
        const double yaw = 0.025;
        cv::Mat second; demo::warpMotion(first, second, camera, 3, -2, yaw);
        tracker.processFrame(first, 0, AltitudeSample{height, 0}, ImuSample{0, 0, 3.13, 0});
        const auto result = tracker.processFrame(second, 0.1, AltitudeSample{height, 0.1},
            ImuSample{0, 0, std::remainder(3.13 + yaw, 2 * CV_PI), 0.1});
        require(result.valid && result.imu_used && result.planar_velocity_mps, "Fresh IMU and range enable metric motion");
        requireNear(result.yaw_delta_rad, yaw, 1e-12, "IMU yaw wrap and coordinate sign");
        requireNear((*result.planar_displacement_m)[0],
                    -height * (std::cos(yaw)*3/camera.fx - std::sin(yaw)*2/camera.fy), 0.001,
                    "IMU compensated horizontal X motion");
        requireNear((*result.planar_displacement_m)[1],
                    height * (-std::sin(yaw)*3/camera.fx - std::cos(yaw)*2/camera.fy), 0.001,
                    "IMU compensated horizontal Y motion");
        requireNear((*result.planar_velocity_mps)[0], (*result.planar_displacement_m)[0]/0.1, 1e-12,
                    "Sensor-assisted velocity uses camera time");
        tracker.reset();
        demo::warpMotion(first, second, camera, 0, 0, yaw);
        tracker.processFrame(first, 0, AltitudeSample{height, 0}, ImuSample{});
        const auto rotated = tracker.processFrame(second, 0.1, AltitudeSample{height, 0.1}, ImuSample{0,0,yaw,0.1});
        require(rotated.valid && rotated.planar_displacement_m, "Pure IMU yaw remains observable");
        requireNear(cv::norm(*rotated.planar_displacement_m), 0, 0.001, "Pure yaw does not become translation");
    }
}

void testSensorFailuresAndRecovery()
{
    using namespace metric_mapping;
    const CameraIntrinsics camera{320, 240, 240, 200, 159.5, 119.5};
    const auto first = demo::motionTexture(); cv::Mat second;
    demo::warpMotion(first, second, camera, 3, -2, 0.025);
    OpticalFlowSettings settings; settings.require_imu = true;
    const auto pair = [&](std::optional<ImuSample> imu, std::optional<AltitudeSample> height,
                          double timestamp = 0.1) {
        SparseFlowTracker tracker(camera, settings);
        tracker.processFrame(first, 0, AltitudeSample{2, 0}, ImuSample{});
        return tracker.processFrame(second, timestamp, height, imu);
    };
    for (const auto& sample : {ImuSample{0,0,0.025,-0.1}, ImuSample{0,0,0.025,0.2},
                               ImuSample{0.2,0,0.025,0.1}, ImuSample{0.02,0,0.025,0.1},
                               ImuSample{0,0,std::numeric_limits<double>::quiet_NaN(),0.1}}) {
        const auto result = pair(sample, AltitudeSample{2,0.1});
        require(!result.imu_used && !result.planar_displacement_m && !result.pose,
                "Stale/future/tilted/invalid IMU must not enable metric motion or pose");
    }
    require(!pair(std::nullopt, AltitudeSample{2,0.1}).planar_velocity_mps, "Required IMU cannot be silently omitted");
    const auto repeated = pair(ImuSample{}, AltitudeSample{2,0.01}, 0.01);
    require(std::string(repeated.imu_status)=="imu_not_new" && !repeated.planar_displacement_m,
            "A repeated IMU packet cannot represent a new frame's attitude");
    const auto stale_range = pair(ImuSample{0,0,0.025,0.1}, AltitudeSample{2,-1});
    require(stale_range.valid && stale_range.imu_used && !stale_range.planar_displacement_m,
            "IMU alone does not provide metric scale");
    const auto mismatch = pair(ImuSample{0,0,-0.1,0.1}, AltitudeSample{2,0.1});
    require(!mismatch.valid && !mismatch.planar_displacement_m, "Image consensus rejects wrong IMU yaw");

    SparseFlowTracker tracker(camera, settings);
    tracker.processFrame(first, 0, AltitudeSample{2,0}, ImuSample{});
    tracker.processFrame(second, 0.1, AltitudeSample{2,0.1});
    require(!tracker.processFrame(second, 0.2, AltitudeSample{2,0.2}, ImuSample{0,0,0,0.2}).planar_displacement_m,
            "Recovery needs a fresh attitude pair");
    require(tracker.processFrame(second, 0.3, AltitudeSample{2,0.3}, ImuSample{0,0,0,0.3}).planar_displacement_m.has_value(),
            "Metric estimation recovers with a fresh sensor pair");
}

void testSensorSnapshotAdapters()
{
    using namespace metric_mapping;
    TemporaryDirectory directory;
    const auto imu = directory.path / "imu.txt", range = directory.path / "range.txt";
    { std::ofstream file(imu); file << "12.5 0.01 -0.02 0.4\n"; }
    { std::ofstream file(range); file << "12.5 2.3 0.1\n"; }
    const auto attitude = detail::readImuSnapshot(imu);
    const auto altitude = detail::readAltitudeSnapshot(range);
    require(attitude && altitude, "Read actual sensor snapshot formats");
    requireNear(attitude->yaw_rad, 0.4, 0, "IMU yaw field");
    requireNear(altitude->height_m, 2.2, 1e-12, "Ultrasonic mounting offset");
    { std::ofstream file(imu); file << "12.5 0.01\n"; }
    { std::ofstream file(range); file << "12.5 2.3 0.1 extra\n"; }
    require(!detail::readImuSnapshot(imu) && !detail::readAltitudeSnapshot(range), "Reject torn/extra sensor records");
    require(!detail::readImuSnapshot(directory.path/"absent"), "Missing sensor file is unavailable");
}

void testUncalibratedPixelFlow()
{
    using namespace metric_mapping;
    for (const auto size : {cv::Size(320, 240), cv::Size(640, 480)}) {
        OpticalFlowSettings settings;
        settings.profiling = true;
        PixelFlowTracker tracker(size, settings);
        cv::Mat first, second;
        cv::cvtColor(demo::motionTexture(size), first, cv::COLOR_GRAY2BGR);
        const cv::Matx23d translation(1, 0, 3, 0, 1, -2);
        cv::warpAffine(first, second, translation, size, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
        const auto initial = tracker.processFrame(first, 0);
        require(!initial.valid && !initial.velocity_px_s, "First frame has no correspondence");
        const auto flow = tracker.processFrame(second, 0.1);
        const double scale = 320.0 / size.width;
        require(flow.valid && flow.velocity_px_s && !tracker.tracks().empty(), "Uncalibrated BGR frames produce flow");
        require(tracker.workingSize() == cv::Size(320, 240), "Pixel flow respects resolution budget");
        requireNear(flow.median_flow_px.x, 3 * scale, 0.12, "Horizontal image displacement");
        requireNear(flow.median_flow_px.y, -2 * scale, 0.12, "Vertical image displacement");
        requireNear(flow.velocity_px_s->x, 30 * scale, 1.2, "Image velocity uses frame timing");
        require(tracker.debugImage().size() == tracker.workingSize(), "Optional flow arrow image");
        requireThrows([&] { tracker.processFrame(second, 0.1); }, "Reject repeated timestamps");
        tracker.reset();
        const cv::Mat blank(size, CV_8UC3, cv::Scalar::all(80));
        tracker.processFrame(blank, 0);
        require(!tracker.processFrame(blank, 0.1).valid, "No texture does not invent flow");
    }
}
#endif

void testSharedInputAndTimingRules()
{
    using namespace metric_mapping::detail;
    require(parseInteger("0", 0, 2) == 0 && parseInteger("2", 0, 2) == 2, "Integer endpoints");
    requireThrows([] { parseInteger("2x"); }, "Reject numeric prefixes with trailing text");
    requireThrows([] { parseInteger("99999999999999999999"); }, "Reject integer overflow");
    requireThrows([] { parseInteger("-1", 0, 2); }, "Reject out-of-range option");
    std::vector<double> times;
    for (int value = 20; value >= 1; --value) times.push_back(value);
    const auto summary = summarizeTimes(times);
    requireNear(summary.mean, 10.5, 0, "Timing mean");
    requireNear(summary.median, 11, 0, "Existing upper median convention");
    requireNear(summary.p95, 19, 0, "Nearest-rank p95");
    requireNear(summary.worst, 20, 0, "Worst latency");
    times = {3};
    requireNear(summarizeTimes(times).p95, 3, 0, "One timing sample");
    times.clear();
    requireThrows([&] { summarizeTimes(times); }, "Reject empty benchmark");
    times = {std::numeric_limits<double>::quiet_NaN()};
    requireThrows([&] { summarizeTimes(times); }, "Reject invalid timing sample");
}

#ifdef HAVE_VISUAL_FEATURES
namespace metric_mapping::tests {
namespace {
cv::Mat analyticPlane(const VisualFeatureFrame& f,const std::string& name)
{
    const auto& names=f.feature_tensor.channel_names;
    const auto it=std::find(names.begin(),names.end(),name);
    require(it!=names.end(),"Missing analytical channel: "+name);
    return f.feature_tensor.plane(std::size_t(it-names.begin()));
}
bool stageRan(const VisualFeatureFrame& f,const std::string& name)
{
    for(const auto& t:f.analytical_stages)if(t.name==name)return t.ran;
    throw std::runtime_error("Missing timing stage: "+name);
}
VisualFeatureSettings analyticSettings(const std::string& names="all")
{
    auto s=analyticalFeatureSettings();s.analytical.features=selectAnalyticFeatures(names);return s;
}
void analyticShapeAndOrder()
{
    VisualFeatureExtractor extractor(analyticSettings());
    const auto& f=extractor.extract(cv::Mat(180,320,CV_8UC3,cv::Scalar(30,80,160)),0);
    const std::vector<std::string> expected={"Y","Cb","Cr","Gx","Gy","GradientMagnitude",
        "HOG_0","HOG_1","HOG_2","HOG_3","HOG_4","HOG_5","HOG_6","HOG_7","HOG_8",
        "HarrisResponse","CannyEdge","ContourMap","ChromaGradientCb","ChromaGradientCr",
        "OpticalFlowU","OpticalFlowV","Depth","DepthGradientX","DepthGradientY","Slope","Roughness","GeometryConfidence"};
    require(f.feature_tensor.channel_names==expected,"Exact 28-channel order");
    const auto& tensor=f.feature_tensor.values;
    require(tensor.dims==3&&tensor.size[0]==28&&tensor.size[1]==32&&tensor.size[2]==32&&tensor.type()==CV_32F&&tensor.isContinuous(),"CHW float32 tensor");
    require(!f.analytical.geometry_valid&&!f.analytical.scale_valid&&!f.analytical.dense_flow_valid,"Unavailable first-frame geometry and flow");
    require(cv::countNonZero(analyticPlane(f,"GeometryConfidence"))==0,"Unsupported geometry confidence zero");
    require(cv::countNonZero(f.feature_tensor.mask(22))==0,"Missing depth mask invalid");
    require(cv::countNonZero(f.feature_tensor.mask(0))==1024,"Appearance remains available");
    require(cv::checkRange(tensor),"Tensor is finite even when geometry absent");
    auto s=analyticSettings("appearance");s.analytical.batch_dimension=true;
    VisualFeatureExtractor batch(s);const auto& b=batch.extract(f.bgr,0);
    require(b.feature_tensor.values.dims==4&&b.feature_tensor.values.size[0]==1&&b.feature_tensor.values.size[1]==3,"Optional batch dimension");
}
void analyticEdgesAndHog()
{
    auto s=analyticSettings("gradients,hog,canny,contours");
    cv::Mat vertical(256,256,CV_8UC3,cv::Scalar(0,0,0));vertical.colRange(128,256).setTo(cv::Scalar(255,255,255));
    VisualFeatureExtractor extractor(s);const auto& v=extractor.extract(vertical,0);
    require(cv::norm(analyticPlane(v,"Gx"),cv::NORM_L1)>100*cv::norm(analyticPlane(v,"Gy"),cv::NORM_L1),"Vertical edge is Gx");
    require(cv::sum(analyticPlane(v,"HOG_0"))[0]>20,"Unsigned horizontal gradient HOG bin");
    const auto edge=analyticPlane(v,"CannyEdge");require(cv::countNonZero(edge)>=30,"Canny edge survives occupancy pooling");
    require(cv::countNonZero(analyticPlane(v,"ContourMap"))>=30,"Contour raster connectivity retained");
    cv::Mat horizontal;cv::transpose(vertical,horizontal);const auto& h=extractor.extract(horizontal,0.1);
    require(cv::norm(analyticPlane(h,"Gy"),cv::NORM_L1)>100*cv::norm(analyticPlane(h,"Gx"),cv::NORM_L1),"Horizontal edge is Gy");
    require(cv::sum(analyticPlane(h,"HOG_4"))[0]>10&&cv::sum(analyticPlane(h,"HOG_5"))[0]>10,"90 degree gradient interpolates HOG bins 4 and 5");
    cv::Mat diagonal(256,256,CV_8UC3);
    for(int y=0;y<256;++y)for(int x=0;x<256;++x){auto z=std::uint8_t((x+y)%32<16?220:20);diagonal.at<cv::Vec3b>(y,x)={z,z,z};}
    const auto& d=extractor.extract(diagonal,0.2);
    require(cv::sum(analyticPlane(d,"HOG_2"))[0]>2*cv::sum(analyticPlane(d,"HOG_0"))[0],"45 degree pattern HOG orientation");
}
void analyticHarrisAndChroma()
{
    auto s=analyticSettings("harris,chroma,gradients");
    VisualFeatureExtractor extractor(s);
    cv::Mat corner(256,256,CV_8UC3,cv::Scalar(0,0,0));corner(cv::Rect(128,128,128,128)).setTo(cv::Scalar(255,255,255));
    const auto& f=extractor.extract(corner,0);double maximum;cv::Point peak;
    cv::minMaxLoc(analyticPlane(f,"HarrisResponse"),nullptr,&maximum,nullptr,&peak);
    require(maximum>0&&std::abs(peak.x-16)<=1&&std::abs(peak.y-16)<=1,"Harris peaks at corner");
    cv::Mat color(256,256,CV_8UC3,cv::Scalar(0,0,200));color.colRange(128,256).setTo(cv::Scalar(0,102,0));
    const auto& c=extractor.extract(color,0.1);
    require(cv::norm(analyticPlane(c,"ChromaGradientCr"),cv::NORM_L1)>10*cv::norm(analyticPlane(c,"GradientMagnitude"),cv::NORM_L1),"Similar-luminance chromatic edge");
}
void analyticNormalizationAndAblation()
{
    auto s=analyticSettings("appearance");VisualFeatureExtractor e(s);
    cv::Mat a(256,256,CV_8UC3,cv::Scalar(30,80,160));
    const auto y=analyticPlane(e.extract(a,0),"Y").at<float>(5,5);
    a(cv::Rect(128,128,128,128)).setTo(cv::Scalar(255,255,255));
    const auto& f=e.extract(a,0.1);requireNear(analyticPlane(f,"Y").at<float>(5,5),y,1e-7,"Color normalization independent of scene extrema");
    requireNear(y,(0.114*30+0.587*80+0.299*160)/255,1e-5,"Known luminance scaling");
    for(const char* stage:{"sobel","hog","harris","canny","dense_flow","sparse_tracking","pose","diffusion","projection"})require(!stageRan(f,stage),std::string("Disabled stage: ")+stage);
    s=analyticSettings("hog");VisualFeatureExtractor hog_only(s);const auto& h=hog_only.extract(a,0);
    require(h.feature_tensor.channel_names.size()==9&&stageRan(h,"sobel")&&!stageRan(h,"harris"),"Dependency-driven HOG-only ablation");
    s=analyticSettings("none");VisualFeatureExtractor none(s);require(none.extract(a,0).feature_tensor.values.empty(),"Empty bank supported");
    s=analyticSettings();s.analytical.enable_geometry=false;VisualFeatureExtractor no_geometry(s);
    const auto& ng=no_geometry.extract(a,0);
    require(!stageRan(ng,"sparse_tracking")&&ng.feature_tensor.channel_names.size()==22,"Geometry off omits channels and sparse work");
}
void analyticDenseFlow()
{
    cv::Mat image(256,256,CV_8UC3);cv::RNG rng(42);rng.fill(image,cv::RNG::UNIFORM,0,255);
    cv::GaussianBlur(image,image,{3,3},0.6);
    VisualFeatureExtractor e(analyticSettings("flow"));
    const auto& first=e.extract(image,0);require(!first.analytical.dense_flow_valid&&cv::countNonZero(first.feature_tensor.mask(0))==0,"First dense frame unavailable");
    const auto& identical=e.extract(image,0.1);
    require(identical.analytical.dense_flow_valid&&cv::norm(analyticPlane(identical,"OpticalFlowU")(cv::Rect(4,4,24,24)),cv::NORM_INF)<0.01,"Identical frames zero dense flow");
    cv::Mat shifted;cv::warpAffine(image,shifted,cv::Mat(cv::Matx23d(1,0,3,0,1,-2)),image.size());
    const auto& moved=e.extract(shifted,0.2);
    requireNear(cv::mean(analyticPlane(moved,"OpticalFlowU")(cv::Rect(4,4,24,24)))[0]*16,3,0.3,"Dense U translation");
    requireNear(cv::mean(analyticPlane(moved,"OpticalFlowV")(cv::Rect(4,4,24,24)))[0]*16,-2,0.3,"Dense V translation");
    require(!e.extract(shifted,2).analytical.dense_flow_valid,"Frame gap invalidates dense flow");
    e.resetMotion();require(!e.extract(shifted,3).analytical.dense_flow_valid,"Reset clears dense history");
}
void analyticSurfaceAndProjection()
{
    auto grid=makeFlatGrid(15,15,0.1);auto flat=measureSurface(grid);
    requireNear(flat.slope_rad.at<float>(7,7),0,1e-6,"Flat slope");requireNear(flat.roughness_m.at<float>(7,7),0,1e-6,"Flat roughness");
    for(int y=0;y<15;++y)for(int x=0;x<15;++x)grid.elevation_m[grid.index(y,x)]=float(0.2*x*0.1-0.1*y*0.1);
    auto plane=measureSurface(grid);requireNear(plane.slope_rad.at<float>(7,7),std::atan(std::hypot(0.2,0.1)),1e-5,"Metric inclined-plane slope");
    requireNear(plane.gradient_y.at<float>(7,7),0.1,1e-5,"Terrain row to world Y sign");
    const float smooth_rough=plane.roughness_m.at<float>(7,7);
    for(int y=0;y<15;++y)for(int x=0;x<15;++x)grid.elevation_m[grid.index(y,x)]+=((x+y)%2?0.01F:-0.01F);
    require(measureSurface(grid).roughness_m.at<float>(7,7)>smooth_rough+0.005,"High-frequency residual roughness increases");
    grid=makeFlatGrid(15,15,0.1);flat=measureSurface(grid);const CameraIntrinsics k{64,64,40,40,31.5,31.5};
    auto projected=detail::projectSurface(grid,flat,cv::Mat(15,15,CV_32F,cv::Scalar(0.8)),k,detail::nadirTransform({{0,0,2},0}),{64,64},10,true);
    requireNear(projected.planes[0].at<float>(32,32),2,1e-6,"Current-view metric depth");
    requireNear(projected.planes[5].at<float>(32,32),0.8,1e-6,"Projected confidence");
    require(projected.masks[0].at<unsigned char>(0,0)==0,"Unsupported image space stays unknown");
    requireNear(projected.planes[1].at<float>(32,32),0,1e-6,"Flat current-view depth gradient");
    // Known isolated point: +X right, +Y up; no DEM resize can pass this shift test.
    auto one=makeFlatGrid(1,1,0.1);one.minimum_x_m=0.5;one.maximum_y_m=0.25;
    auto single=detail::projectSurface(one,measureSurface(one),cv::Mat(1,1,CV_32F,cv::Scalar(1)),k,detail::nadirTransform({{0,0,2},0}),{64,64},10,false);
    require(single.masks[0].at<unsigned char>(27,42)==255,"Known XYZ projects to correct pixel");
    auto moved=detail::projectSurface(one,measureSurface(one),cv::Mat(1,1,CV_32F,cv::Scalar(1)),k,detail::nadirTransform({{0.5,0,2},0}),{64,64},10,false);
    require(moved.masks[0].at<unsigned char>(27,32)==255,"Projection follows current camera position");
    auto overlap=makeFlatGrid(2,1,0.001);
    overlap.minimum_x_m=0;overlap.maximum_y_m=0;
    overlap.elevation_m={0,0.5F};
    const auto occluded=detail::projectSurface(overlap,measureSurface(overlap),
        cv::Mat(1,2,CV_32F,cv::Scalar(1)),k,detail::nadirTransform({{0,0,2},0}),{64,64},10,false);
    requireNear(occluded.planes[0].at<float>(32,32),1.5,1e-6,"Z-buffer keeps nearest projected sample");
}
void analyticRelativeGeometry()
{
    const CameraIntrinsics k{320,240,240,240,159.5,119.5};std::vector<TrackedFeature> tracks;std::vector<cv::Vec3d> truth;
    const double angle=0.025;const cv::Matx33d r(std::cos(angle),0,std::sin(angle),0,1,0,-std::sin(angle),0,std::cos(angle));
    const cv::Vec3d t(-0.4,0.02,0.03);cv::RNG rng(197);
    const auto project=[&](cv::Vec3d p){return cv::Point2f(float(k.fx*p[0]/p[2]+k.cx),float(k.fy*p[1]/p[2]+k.cy));};
    for(int i=0;i<90;++i){
        const cv::Vec3d p(rng.uniform(-1.3,1.3),rng.uniform(-0.9,0.9),rng.uniform(3.0,5.0));
        truth.push_back(p);auto q=project(r*p+t);if(i<15)q={float(rng.uniform(20,300)),float(rng.uniform(20,220))};
        tracks.push_back({std::uint64_t(i),2,project(p),q});
    }
    const auto pose=estimateRelativePose(tracks,k);
    require(pose.valid&&!pose.scale_valid,"Essential pose valid, metric scale explicitly unknown");
    require(pose.translation_direction.dot(t/cv::norm(t))>0.99,"Essential translation direction");
    require(cv::norm(cv::Mat(pose.rotation_21-r))<0.02,"Recovered rotation");
    require(std::count(pose.inliers.begin(),pose.inliers.begin()+15,1)<4,"Incorrect correspondences rejected");
    require(pose.relative_points.size()>10&&pose.mean_reprojection_error_px<0.5,"Filtered relative triangulation");
    for(std::size_t i=0,j=0;i<tracks.size()&&j<pose.relative_points.size();++i)if(pose.inliers[i]){
        require(cv::norm(pose.relative_points[j]*cv::norm(t)-truth[i])<0.05,"Triangulated relative structure matches known scale");++j;
    }
    for(auto& track:tracks)track.current_px=track.previous_px;
    require(!estimateRelativePose(tracks,k).valid,"No-parallax pose unavailable");
}
void analyticCalibrationAndExport()
{
    auto s=analyticSettings("appearance,gradients");s.distortion={0,0,0,0,0};const CameraIntrinsics k{320,240,240,200,159.5,119.5};
    VisualFeatureExtractor e(s,k);const auto& f=e.extract(cv::Mat(240,320,CV_8UC3,cv::Scalar(20,50,90)),0);
    requireNear(f.camera->fx,192,1e-9,"Analytical resize fx");requireNear(f.camera->fy,200*256.0/240,1e-9,"Analytical resize fy");
    requireNear(f.camera->cx,127.5,1e-9,"Pixel-center principal point");
    TemporaryDirectory temp;writeVisualFeatures(temp.path/"features.yml",f,s);
    cv::FileStorage file((temp.path/"features.yml").string(),cv::FileStorage::READ);cv::Mat tensor;file["analytical"]["tensor"]>>tensor;
    require(tensor.dims==3&&tensor.size[0]==6&&cv::norm(tensor,f.feature_tensor.values)==0,"Tensor export round trip");
    require(renderAnalyticalFeatures(f).type()==CV_8UC3,"Headless diagnostic rendering");
    {std::ofstream config(temp.path/"bad.yml");config << "%YAML:1.0\nflow_scal_px: 16\n";}
    requireThrows([&]{loadAnalyticalConfig(temp.path/"bad.yml");},"Misspelled settings rejected");
    auto bad=s;bad.analytical.flow_scale_px=0;requireThrows([&]{VisualFeatureExtractor invalid(bad,k);},"Reject zero normalization scale");
}
void analyticMapLifecycle()
{
#ifdef HAVE_MOTION
    auto s=analyticSettings("depth,depth_gradients,slope,roughness,geometry_confidence");
    s.analytical.maximum_map_points=24;const CameraIntrinsics k{256,256,220,220,127.5,127.5};
    VisualFeatureExtractor e(s,k);const auto texture=demo::motionTexture({256,256});cv::Mat gray,bgr;
    bool geometry=false;const VisualFeatureFrame* latest=nullptr;
    for(int i=0;i<70;++i){demo::warpMotion(texture,gray,k,0.7*i,0.12*i);cv::cvtColor(gray,bgr,cv::COLOR_GRAY2BGR);
        latest=&e.extract(bgr,0.1*i,AltitudeSample{2,0.1*i});geometry|=latest->analytical.geometry_valid;
        require(latest->local_map.size()<=24,"Rolling point budget");}
    require(geometry&&latest->analytical.scale_valid&&!latest->local_map.empty(),"Sparse tracker connects to bounded colored map and current-view geometry");
    const auto& lost=e.extract(bgr,7.1);
    require(!lost.analytical.scale_valid&&!lost.analytical.geometry_valid&&lost.local_map.empty(),"Scale loss clears metric map segment");
    require(cv::countNonZero(analyticPlane(lost,"GeometryConfidence"))==0,"Scale loss confidence zero");
    e.resetMotion();const auto& fresh=e.extract(bgr,8,AltitudeSample{2,8});require(!fresh.analytical.geometry_valid,"Reset does not reuse prior geometry");
    s.analytical.map_age_s=0.05;VisualFeatureExtractor short_lived(s,k);bool had_points=false;
    for(int i=0;i<25;++i){
        demo::warpMotion(texture,gray,k,0.7*i,0.12*i);cv::cvtColor(gray,bgr,cv::COLOR_GRAY2BGR);
        const auto& f=short_lived.extract(bgr,0.1*i,AltitudeSample{2,0.1*i});
        had_points|=!f.local_map.empty();
        if(f.analytical.num_triangulated_points==0)require(f.local_map.empty(),"Age pruning does not reinsert old tracker landmarks");
    }
    require(had_points,"Age-pruning trial actually reconstructed points");
#endif
}
}
std::vector<TestCase> analyticalTests()
{
    return {{"analytical tensor shape order validity",analyticShapeAndOrder},
        {"analytical edges dense HOG contours",analyticEdgesAndHog},
        {"analytical Harris chroma boundaries",analyticHarrisAndChroma},
        {"analytical normalization ablation",analyticNormalizationAndAblation},
        {"analytical camera dense flow",analyticDenseFlow},
        {"analytical surface and current-view projection",analyticSurfaceAndProjection},
        {"analytical essential pose relative triangulation outliers",analyticRelativeGeometry},
        {"analytical calibration tensor export",analyticCalibrationAndExport},
        {"analytical rolling map lifecycle",analyticMapLifecycle}};
}
} // namespace metric_mapping::tests
#endif

int main(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--check-demo-json") {
        try {
            cv::FileStorage metadata(argv[2], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            require(metadata.isOpened(), "Demo metadata must be valid JSON");
            require(static_cast<std::string>(metadata["input_image_1"]) == argv[3],
                    "Escaped demo path must round-trip exactly");
            require(static_cast<int>(metadata["raw_points"]) == 262144 &&
                    static_cast<int>(metadata["measured_cells"]) == 262144 &&
                    static_cast<int>(metadata["idw_interpolated_cells"]) == 0,
                    "Demo must preserve every original lattice measurement");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    std::vector<TestCase> tests;
    tests.push_back({"shared camera resize preserves rays", testSharedCameraResize});
    tests.push_back({"shared input and timing rules", testSharedInputAndTimingRules});
    for (const auto& group : {rgbdTests(), terrainTests(), configTests(), outputTests(), ultrasonicTests()})
        tests.insert(tests.end(), group.begin(), group.end());
#ifdef HAVE_TWO_VIEW
    const auto stereo = stereoTests();
    tests.insert(tests.end(), stereo.begin(), stereo.end());
#endif
#ifdef HAVE_MOTION
    tests.push_back({"IMU and ultrasonic geometry", testImuAndUltrasonicGeometry});
    tests.push_back({"sensor failures and recovery", testSensorFailuresAndRecovery});
    tests.push_back({"sensor snapshot adapters", testSensorSnapshotAdapters});
    tests.push_back({"uncalibrated pixel optical flow", testUncalibratedPixelFlow});
    const auto motion = motionTests();
    tests.insert(tests.end(), motion.begin(), motion.end());
#endif

    int failures = 0;
#ifdef HAVE_VISUAL_FEATURES
    const auto analytical = analyticalTests();
    tests.insert(tests.end(), analytical.begin(), analytical.end());
    const auto visual = visualFeatureTests();
    tests.insert(tests.end(), visual.begin(), visual.end());
#endif
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what()
                      << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << tests.size() << " tests passed\n";
    return 0;
}
