#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace metric_mapping {

enum class DepthUnit {
    Meters,
    Millimeters
};

enum class PoseConvention {
    CameraToWorld,
    WorldToCamera
};

struct CameraIntrinsics {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct DepthConfig {
    DepthUnit unit = DepthUnit::Meters;
    double min_depth_m = 0.0;
    double max_depth_m = 0.0;
    int pixel_stride = 1;
    std::vector<double> invalid_values;
};

struct FusionConfig {
    double voxel_size_m = 0.0;
    std::size_t max_raw_points = 0;
    std::size_t max_voxels = 0;
};

struct IdwConfig {
    bool enabled = false;
    double search_radius_m = 0.0;
    int minimum_neighbors = 0;
    double power = 2.0;
    double maximum_interpolation_distance_m = 0.0;
};

struct MapConfig {
    double resolution_m = 0.0;
    std::size_t max_cells = 0;
    IdwConfig idw;
};

struct FrameSpec {
    std::filesystem::path rgb_path;
    std::filesystem::path depth_path;
    cv::Matx44d supplied_pose = cv::Matx44d::eye();
};

struct ReconstructionConfig {
    std::filesystem::path config_path;
    std::filesystem::path output_directory;
    std::string world_frame;
    CameraIntrinsics camera;
    DepthConfig depth;
    FusionConfig fusion;
    MapConfig map;
    PoseConvention pose_convention = PoseConvention::CameraToWorld;
    bool images_are_rectified = false;
    bool depth_registered_to_rgb = false;
    std::optional<cv::Vec3d> uav_position_world_m;
    std::vector<FrameSpec> frames;
};

struct ColoredPoint {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

struct CloudBounds {
    cv::Vec3d minimum{0.0, 0.0, 0.0};
    cv::Vec3d maximum{0.0, 0.0, 0.0};
};

struct FrameDiagnostics {
    int width = 0;
    int height = 0;
    std::size_t valid_depth_pixels = 0;
    std::size_t invalid_depth_pixels = 0;
    std::size_t emitted_points = 0;
    double minimum_depth_m = std::numeric_limits<double>::quiet_NaN();
    double maximum_depth_m = std::numeric_limits<double>::quiet_NaN();
    double median_depth_m = std::numeric_limits<double>::quiet_NaN();
};

struct FrameResult {
    std::vector<ColoredPoint> world_points;
    FrameDiagnostics diagnostics;
};

// Grid convention: column increases with world +X, row increases with world
// -Y, and each elevation value is world +Z in metres.
struct TerrainGrid {
    int width = 0;
    int height = 0;
    double resolution_m = 0.0;
    double minimum_x_m = 0.0;
    double maximum_y_m = 0.0;
    std::vector<float> elevation_m;
    std::vector<cv::Vec3b> color_bgr;
    // 0 = unknown, 127 = IDW-interpolated, 255 = directly measured.
    std::vector<std::uint8_t> validity;

    std::size_t index(int row, int column) const
    {
        return static_cast<std::size_t>(row) *
                   static_cast<std::size_t>(width) +
               static_cast<std::size_t>(column);
    }
};

struct GridStatistics {
    std::size_t measured_cells = 0;
    std::size_t interpolated_cells = 0;
    std::size_t unknown_cells = 0;
};

// Defaults reproduce the constraint values reported in Table VIII of
// Iratni & Diaf. They must be changed to match the real UAV before flight.
struct LandingAnalysisConfig {
    int diffusion_iterations = 10;
    double diffusion_conductance_m = 0.05;
    double diffusion_time_step = 0.2;
    double uav_footprint_diagonal_m = 1.0;
    double maximum_slope_degrees = 10.0;
    double maximum_roughness_m = 0.02;
    double minimum_hazard_distance_m = 1.5;
    double minimum_global_safety_index = 44.8;
};

struct LandingSite {
    bool found = false;
    int row = -1;
    int column = -1;
    cv::Vec3d world_m{0.0, 0.0, 0.0};
    double slope_degrees = std::numeric_limits<double>::quiet_NaN();
    double roughness_m = std::numeric_limits<double>::quiet_NaN();
    double nearest_hazard_distance_m =
        std::numeric_limits<double>::quiet_NaN();
    double spatial_distance_m = std::numeric_limits<double>::quiet_NaN();
    double safety_index = std::numeric_limits<double>::quiet_NaN();
    double distance_index = std::numeric_limits<double>::quiet_NaN();
    double global_safety_index = std::numeric_limits<double>::quiet_NaN();
};

struct LandingAnalysis {
    cv::Mat smoothed_dem_m;
    cv::Mat slope_degrees;
    cv::Mat roughness_m;
    // 255 = hazard, 0 = geometrically admissible terrain.
    cv::Mat hazard_mask;
    cv::Mat nearest_hazard_distance_m;
    cv::Mat spatial_distance_m;
    cv::Mat safety_index;
    cv::Mat distance_index;
    cv::Mat global_safety_index;
    LandingSite best_site;
    std::size_t hazard_cells = 0;
    std::size_t admissible_cells = 0;
    std::size_t clearance_cells = 0;
    std::size_t candidate_cells = 0;
    double maximum_global_safety_index = 0.0;
};

}  // namespace metric_mapping
