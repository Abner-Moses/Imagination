#pragma once
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
// Imagination API: data records first, then the functions that use them.
// Coordinates and assumptions are explained alongside each interface and in README.md.

// ============================================================================
// Types
// ============================================================================

#include <opencv2/core.hpp>


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
    int maximum_neighbors = 0; // 0 preserves the legacy unlimited-neighbor behavior.
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

// A synchronized reading from a sensor pointing vertically down (world -Z).
struct UltrasonicMeasurement {
    double distance_m = 0.0;
    double maximum_error_m = 0.2;
    cv::Vec3d sensor_offset_world_m{0.0, 0.0, 0.0};
};

struct GroundRangeCheck {
    UltrasonicMeasurement measurement;
    std::optional<double> mapped_distance_m;
    bool consistent = false;
    const char* status() const {
        return !mapped_distance_m ? "ground_unavailable" :
               consistent ? "consistent" : "mismatch";
    }
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
    std::optional<UltrasonicMeasurement> ultrasonic;
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
    std::optional<GroundRangeCheck> ultrasonic;
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

// ============================================================================
// Config
// ============================================================================


namespace metric_mapping {

ReconstructionConfig loadConfig(const std::filesystem::path& config_path);
UltrasonicMeasurement loadUltrasonicConfig(const std::filesystem::path& config_path);
std::string depthUnitName(DepthUnit unit);
std::string poseConventionName(PoseConvention convention);

}  // namespace metric_mapping

// ============================================================================
// Geometry
// ============================================================================

#include <opencv2/core.hpp>


namespace metric_mapping {

void validateCamera(const CameraIntrinsics& camera);
void validateRigidTransform(const cv::Matx44d& transform,
                            const std::string& name);
cv::Matx44d invertRigidTransform(const cv::Matx44d& transform);
cv::Matx44d cameraToWorldTransform(const cv::Matx44d& supplied_pose,
                                   PoseConvention convention);
cv::Vec3d backProjectPixel(double u,
                           double v,
                           double depth_m,
                           const CameraIntrinsics& camera);
cv::Vec3d transformPoint(const cv::Matx44d& transform,
                         const cv::Vec3d& point);
FrameResult backProjectFrame(const cv::Mat& rgb,
                             const cv::Mat& depth,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world);
FrameResult reconstructFrame(const std::filesystem::path& rgb_path,
                             const std::filesystem::path& depth_path,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world);

}  // namespace metric_mapping

// ============================================================================
// Point Cloud
// ============================================================================


namespace metric_mapping {

class VoxelGridAccumulator {
public:
    VoxelGridAccumulator(double voxel_size_m, std::size_t max_voxels);

    void add(const ColoredPoint& point);
    void add(const std::vector<ColoredPoint>& points);
    std::vector<ColoredPoint> points() const;
    std::size_t voxelCount() const;

private:
    struct Key {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;

        bool operator==(const Key& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const;
    };

    struct Accumulator {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        std::size_t count = 0;
    };

    double voxel_size_m_ = 0.0;
    std::size_t max_voxels_ = 0;
    std::unordered_map<Key, Accumulator, KeyHash> voxels_;
};

CloudBounds computeBounds(const std::vector<ColoredPoint>& points);

}  // namespace metric_mapping

// ============================================================================
// Terrain
// ============================================================================


namespace metric_mapping {

TerrainGrid createTerrainGrid(const std::vector<ColoredPoint>& points,
                              double resolution_m,
                              std::size_t max_cells);
void interpolateIdw(TerrainGrid& grid, const IdwConfig& config);
GridStatistics computeGridStatistics(const TerrainGrid& grid);
std::vector<ColoredPoint> terrainGridPoints(const TerrainGrid& grid);
LandingAnalysis analyzeLandingSites(
    const TerrainGrid& grid,
    const cv::Vec3d& uav_position_world_m,
    const LandingAnalysisConfig& config = {},
    const std::optional<UltrasonicMeasurement>& ultrasonic = std::nullopt);

}  // namespace metric_mapping

// ============================================================================
// Ultrasonic
// ============================================================================

namespace metric_mapping {
void validateUltrasonicMeasurement(const UltrasonicMeasurement& measurement);
GroundRangeCheck checkGroundRange(const TerrainGrid& grid,
                                 const cv::Vec3d& uav_position_world_m,
                                 const UltrasonicMeasurement& measurement);
} // namespace metric_mapping

// ============================================================================
// Io
// ============================================================================


namespace metric_mapping {

std::string escapeJson(const std::string& value);

void writePly(const std::filesystem::path& path,
              const std::vector<ColoredPoint>& points,
              const std::string& coordinate_comment =
                  "coordinates are metres in map_z_up world frame");
void writeDemCsv(const std::filesystem::path& path,
                 const TerrainGrid& grid);
void writeTerrainImages(const std::filesystem::path& output_directory,
                        const TerrainGrid& grid);
void writeLandingAnalysis(
    const std::filesystem::path& output_directory,
    const TerrainGrid& grid,
    const LandingAnalysis& analysis,
    const LandingAnalysisConfig& config,
    const cv::Vec3d& uav_position_world_m,
    bool demo_only);
void writeMetadata(const std::filesystem::path& path,
                   const ReconstructionConfig& config,
                   std::size_t raw_point_count,
                   const std::vector<ColoredPoint>& filtered_points,
                   const CloudBounds& bounds,
                   const TerrainGrid& grid,
                   const GridStatistics& grid_statistics,
                   const std::vector<cv::Vec3d>& camera_positions_world_m);

}  // namespace metric_mapping

// ============================================================================
// Visualization
// ============================================================================


namespace metric_mapping {

cv::Mat fitImage(const cv::Mat& image, cv::Size box);

void writeDebugVisualization(
    const std::filesystem::path& path,
    const std::vector<ColoredPoint>& points,
    const CloudBounds& bounds,
    const TerrainGrid& grid,
    const std::vector<cv::Vec3d>& camera_positions_world_m,
    const std::optional<cv::Vec3d>& uav_position_world_m);

}  // namespace metric_mapping

// ============================================================================
// Two View
// ============================================================================

#include <opencv2/core.hpp>


namespace metric_mapping {

struct TwoViewSettings {
    int maximum_features = 8000;
    float lowe_ratio = 0.75F;
    double ransac_probability = 0.999;
    double ransac_threshold_px = 1.0;
    double minimum_depth_m = 0.2;
    double maximum_depth_m = 200.0;
    double maximum_rectified_epipolar_error_px = 2.0;
    double left_right_disparity_tolerance_px = 1.0;
    int stereo_block_size = 7;
};

struct TwoViewDiagnostics {
    std::size_t features_image_1 = 0;
    std::size_t features_image_2 = 0;
    std::size_t good_matches = 0;
    std::size_t ransac_inliers = 0;
    std::size_t alignment_inliers = 0;
    std::size_t dense_stereo_points = 0;
    std::size_t final_points = 0;
    double median_rectified_epipolar_error_px = 0.0;
};

struct TwoViewResult {
    std::vector<ColoredPoint> points;
    cv::Mat matches_image;
    cv::Mat disparity_preview;
    cv::Matx33d rotation_21 = cv::Matx33d::eye();
    cv::Vec3d translation_21{0.0, 0.0, 0.0};
    std::vector<cv::Vec3d> camera_positions_world_m;
    TwoViewDiagnostics diagnostics;
};

// Reconstructs a dense metric cloud using a known camera-center baseline.
// The output frame assumes camera 1 is nadir-facing: +X right in image 1,
// +Y toward the top of image 1, and +Z upward, with camera 1 at the origin.
TwoViewResult reconstructTwoView(
    const std::filesystem::path& image_1_path,
    const std::filesystem::path& image_2_path,
    const CameraIntrinsics& camera,
    double baseline_m,
    const TwoViewSettings& settings = {});

}  // namespace metric_mapping

// ============================================================================
// Motion Geometry
// ============================================================================

namespace metric_mapping {

// Local z-up coordinates, yaw counterclockwise about +Z. At yaw=0,
// camera right = +X, camera down = -Y, optical axis = -Z.
struct NadirPose {
    cv::Vec3d position_m{0, 0, 0};
    double yaw_rad = 0;
};

struct AltitudeSample {
    double height_m = 0; // Camera center above the locally horizontal ground.
    double timestamp_s = 0;
};

// Attitude from the IMU/flight controller, transformed into the CAMERA mount's
// nadir reference: roll=pitch=0 means downward-facing, yaw is CCW about world +Z.
// Radians, same monotonic clock as frames. Not raw accelerometer measurements.
struct ImuSample {
    double roll_rad = 0, pitch_rad = 0, yaw_rad = 0;
    double timestamp_s = 0;
};

// Sensor offset must be relative to the CAMERA center, in world axes.
// Horizontal offsets are harmless only under the horizontal-plane assumption.
AltitudeSample altitudeFromUltrasonic(const UltrasonicMeasurement& reading,
                                     double timestamp_s);

struct TriangulationSettings {
    double minimum_baseline_m = 0.05;
    double minimum_parallax_rad = CV_PI / 180.0;
    double maximum_reprojection_error_px = 1.0;
    double maximum_depth_m = 100.0;
};

struct TriangulatedPoint {
    cv::Vec3d world_m{0, 0, 0};
    double parallax_rad = 0;
    double reprojection_error_px = 0;
};

// Closest-point ray intersection; rejects weak baseline/parallax, negative
// depth, excessive depth, and reprojection error. No dense matrix solver.
std::optional<TriangulatedPoint> triangulateTrack(
    const cv::Point2f& first, const cv::Point2f& second,
    const CameraIntrinsics& camera, const NadirPose& first_pose,
    const NadirPose& second_pose, const TriangulationSettings& settings = {});

} // namespace metric_mapping

// ============================================================================
// Optical Flow
// ============================================================================


namespace metric_mapping {

struct OpticalFlowSettings {
    int maximum_width = 320, maximum_height = 240; // Never upscale.
    int maximum_tracks = 100, replenish_below = 60, minimum_inliers = 12;
    int grid_columns = 5, grid_rows = 4;
    int detection_interval = 5; // Minimum spacing, including textureless retries.
    double corner_quality = 0.01, corner_distance_px = 7;
    int corner_block_size = 3;
    bool retain_correspondences = false; // LK survivors before the nadir model gate.
    int window_size = 15, pyramid_levels = 1, lk_iterations = 12;
    double lk_epsilon = 0.03, maximum_lk_error = 20;
    bool forward_backward = false; // Optional second LK pass.
    double maximum_forward_backward_error_px = 0.75;
    double model_error_px = 1.5, minimum_inlier_fraction = 0.6;
    double minimum_coverage = 0.3, minimum_quality = 0.2;
    double maximum_yaw_rad = 0.15, maximum_scale_change = 0.2;
    double maximum_frame_gap_s = 0.5, maximum_altitude_age_s = 0.15;
    double altitude_scale_tolerance = 0.05;
    bool require_imu = false; // If true, missing/stale IMU suppresses metric output.
    double maximum_imu_age_s = 0.02;
    double maximum_imu_tilt_rad = 5 * CV_PI / 180;
    double maximum_imu_tilt_change_rad = 0.3 * CV_PI / 180;
    bool triangulation = false, profiling = false;
    int triangulation_interval = 5, minimum_track_age = 5;
    int maximum_triangulations_per_frame = 4, maximum_landmarks = 64;
    TriangulationSettings triangulation_limits;
};

struct TrackedFeature {
    std::uint64_t id = 0;
    std::uint32_t age = 1;
    cv::Point2f previous_px, current_px; // Working-resolution pixels.
};

struct SparseLandmark {
    std::uint64_t track_id = 0;
    TriangulatedPoint point;
};

struct MotionTimings {
    double preprocessing_ms = 0, detection_ms = 0, flow_ms = 0;
    double filtering_ms = 0, geometry_ms = 0, triangulation_ms = 0, total_ms = 0;
};

struct MotionEstimate {
    bool valid = false;
    const char* status = "initializing";
    const char* metric_status = "altitude_unavailable";
    const char* imu_status = "not_supplied";
    bool imu_used = false;
    double dt_s = 0;
    cv::Point2f median_flow_px{0, 0}; // Raw accepted image displacement.
    cv::Vec2d normalized_displacement{0, 0}; // Rotation-compensated, prior z-up axes.
    std::optional<cv::Vec2d> translation_direction; // Absent near zero motion.
    std::optional<cv::Vec2d> planar_displacement_m, planar_velocity_mps;
    double yaw_delta_rad = 0, image_scale = 1;
    double quality = 0, coverage = 0, inlier_fraction = 0, residual_px = 0;
    std::size_t tracked_points = 0, accepted_points = 0, active_tracks = 0;
    std::size_t replenished_points = 0, triangulation_attempts = 0, new_landmarks = 0;
    bool consensus_fallback = false, detection_ran = false;
    // Dead reckoning only, never a globally registered map pose. A lost metric
    // interval starts a new segment and clears its landmarks/anchor poses.
    std::uint64_t segment_id = 0;
    std::optional<NadirPose> pose;
    MotionTimings timing;
};

class SparseFlowTracker {
public:
    explicit SparseFlowTracker(const CameraIntrinsics& camera,
                               const OpticalFlowSettings& settings = {});
    ~SparseFlowTracker();
    SparseFlowTracker(const SparseFlowTracker&) = delete;
    SparseFlowTracker& operator=(const SparseFlowTracker&) = delete;

    // Images must be rectified CV_8UC1/BGR/BGRA at the calibrated input size.
    // Timestamps must increase. Missing/stale altitude never reuses old scale.
    MotionEstimate processFrame(const cv::Mat& frame, double timestamp_s,
                                std::optional<AltitudeSample> altitude = std::nullopt,
                                std::optional<ImuSample> imu = std::nullopt);
    void reset();
    const CameraIntrinsics& workingCamera() const;
    const std::vector<TrackedFeature>& tracks() const;
    const std::vector<TrackedFeature>& correspondences() const;
    const std::vector<SparseLandmark>& landmarks() const;
    // Only this explicit call draws/copies a debug image. No GUI dependency.
    cv::Mat debugImage(const MotionEstimate& estimate) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metric_mapping

// ============================================================================
// Uncalibrated image-space optical flow (no metric motion or camera pose).
// ============================================================================

namespace metric_mapping {
struct PixelFlowEstimate {
    bool valid = false;
    const char* status = "initializing";
    double dt_s = 0;
    cv::Point2f median_flow_px{0, 0}; // Working pixels; +x right, +y down.
    std::optional<cv::Point2f> velocity_px_s;
    std::size_t active_tracks = 0, tracked_points = 0, accepted_points = 0;
    double quality = 0, processing_ms = 0;
};

class PixelFlowTracker {
public:
    explicit PixelFlowTracker(cv::Size input_size, OpticalFlowSettings settings = {});
    PixelFlowEstimate processFrame(const cv::Mat& frame, double timestamp_s);
    const std::vector<TrackedFeature>& tracks() const { return tracker_.tracks(); }
    cv::Size workingSize() const;
    cv::Mat debugImage() const { return tracker_.debugImage(last_); }
    void reset() { tracker_.reset(); last_ = {}; }
private:
    SparseFlowTracker tracker_;
    MotionEstimate last_;
};
} // namespace metric_mapping

// ============================================================================
// Visual Features
// ============================================================================


namespace metric_mapping {
// The analytical bank is an optional mode of VisualFeatureExtractor, not a
// second extractor. Legacy comparison planes remain independently selectable.
enum class AnalyticFeature : std::size_t {
    Appearance, Gradients, Hog, Harris, Canny, Contours, Chroma, DenseFlow,
    Depth, DepthGradients, Slope, Roughness, GeometryConfidence, Count
};
constexpr std::size_t analytic_feature_count = std::size_t(AnalyticFeature::Count);
using AnalyticSelection = std::bitset<analytic_feature_count>;
const char* analyticFeatureName(AnalyticFeature feature);
AnalyticSelection selectAnalyticFeatures(const std::string& names);

struct SurfaceSettings {
    int diffusion_iterations = 5, roughness_window = 5;
    double diffusion_time_step = 0.2, diffusion_conductance_m = 0.05;
    bool gradients = true, slope = true, roughness = true;
};
struct SurfaceFeatures {
    cv::Mat elevation_m, valid, gradient_valid, roughness_valid;
    cv::Mat gradient_x, gradient_y, slope_rad, smoothed_m, roughness_m;
    double gradients_ms = 0, slope_ms = 0, diffusion_ms = 0, roughness_ms = 0;
};
// Nonsemantic surface measurements; does not run any landing decisions.
SurfaceFeatures measureSurface(const TerrainGrid&, const SurfaceSettings& = {});

struct RelativePose {
    bool valid = false, scale_valid = false;
    cv::Matx33d rotation_21 = cv::Matx33d::eye();
    cv::Vec3d translation_direction{0, 0, 0}; // X_cam2 = R21 X_cam1 + scale*t.
    double scale = 0, confidence = 0;
    std::vector<std::uint8_t> inliers;
    // In camera-1 coordinates, unit-baseline units, NEVER metres.
    std::vector<cv::Vec3d> relative_points;
    double mean_reprojection_error_px = 0;
    double pose_ms = 0, triangulation_ms = 0;
};
struct RelativePoseSettings {
    int minimum_inliers = 12, maximum_iterations = 100, maximum_points = 32;
    double threshold_px = 1, minimum_parallax_rad = CV_PI / 180;
    double maximum_reprojection_error_px = 1, maximum_relative_depth = 100;
};
RelativePose estimateRelativePose(const std::vector<TrackedFeature>&,
                                 const CameraIntrinsics&, const RelativePoseSettings& = {});

struct AnalyticalSettings {
    bool enabled = false; // Existing API behavior remains unchanged unless enabled.
    AnalyticSelection features = AnalyticSelection{}.set();
    cv::Size working_size{256, 256}, grid_size{32, 32};
    bool batch_dimension = false;
    int gaussian_kernel = 3;
    double gaussian_sigma = 1, harris_k = 0.04, harris_scale = 0.01;
    double flow_scale_px = 16, depth_scale_m = 20, depth_gradient_scale = 5;
    double roughness_scale_m = 0.1;
    int flow_levels = 2, flow_window = 15, flow_iterations = 2, flow_poly_n = 5;
    double flow_pyramid_scale = 0.5, flow_poly_sigma = 1.2;
    bool enable_geometry = true;
    int pose_interval = 5;
    RelativePoseSettings relative_pose;
    std::size_t maximum_map_points = 256, maximum_grid_cells = 4096;
    double map_radius_m = 5, map_age_s = 10, voxel_size_m = 0.05, grid_resolution_m = 0.1;
    IdwConfig idw{true, 0.3, 3, 2, 0.3, 12};
    SurfaceSettings surface;
    bool has(AnalyticFeature f) const {
        const bool geometric = f >= AnalyticFeature::Depth && f <= AnalyticFeature::GeometryConfidence;
        return features.test(std::size_t(f)) && (enable_geometry || !geometric);
    }
};
struct LocalMapPoint {
    ColoredPoint point;
    std::uint64_t track_id = 0;
    double timestamp_s = 0, confidence = 0, reprojection_error_px = 0;
    std::uint32_t observations = 1;
};
struct AnalyticalTensor {
    // Contiguous channel-major storage: ((channel * height) + row) * width + col.
    // Optional batch dimension has size 1 and does not change storage ordering.
    cv::Mat values, valid; // CV_32F / CV_8U, dimensions C,H,W or 1,C,H,W.
    std::vector<std::string> channel_names;
    std::vector<bool> channel_valid;
    cv::Mat plane(std::size_t channel) const;
    cv::Mat mask(std::size_t channel) const;
};
struct AnalyticalDiagnostics {
    bool calibration_valid = false, scale_valid = false, geometry_valid = false, dense_flow_valid = false;
    std::string geometry_status = "unavailable";
    std::size_t num_sparse_tracks = 0, num_valid_tracks = 0, num_pose_inliers = 0;
    std::size_t num_triangulated_points = 0, num_valid_3d_points = 0, local_map_point_count = 0;
    double geometry_coverage_percent = 0, mean_reprojection_error = 0;
    std::size_t payload_bytes = 0;
};
struct NamedStageTiming { std::string name; bool ran = false; double elapsed_ms = 0; };

enum class VisualFeature : std::size_t {
    Gradients, EdgeMagnitude, EdgeOrientation, Corners, Keypoints,
    ColorTransitions, TextureOrientation, LocalBinaryTexture, Contours, ShapeBoundaries, Motion, Count
};
constexpr std::size_t visual_feature_count = std::size_t(VisualFeature::Count);
using VisualFeatureSelection = std::bitset<visual_feature_count>;
const char* visualFeatureName(VisualFeature feature);
VisualFeatureSelection selectVisualFeatures(const std::string& comma_separated); // all, none, or names

struct VisualFeatureSettings {
    VisualFeatureSelection features = VisualFeatureSelection{}.set();
    int maximum_width = 320, maximum_height = 240;
    int maximum_points = 100, grid_columns = 5, grid_rows = 4;
    float corner_quality = 0.01F, point_spacing_px = 6;
    int fast_threshold = 12, tensor_window = 5;
    float minimum_gradient = 0.005F, minimum_texture_energy = 0.00001F;
    float minimum_texture_coherence = 0.2F;
    double canny_low = 20, canny_high = 60, contour_epsilon_px = 1;
    int minimum_contour_points = 5, maximum_contours = 64, maximum_contour_points = 2048;
    bool profiling = false;
    OpticalFlowSettings motion;
    AnalyticalSettings analytical;
    std::vector<double> distortion; // OpenCV k1,k2,p1,p2[,k3...]; empty means rectified.
    bool has(VisualFeature feature) const { return features.test(std::size_t(feature)); }
};

enum class VisualStage : std::size_t {
    Preparation, Derivatives, Edges, StructureTensor, Corners, Keypoints,
    ColorTransitions, LocalBinaryTexture, Boundaries, Motion, Count
};
constexpr std::size_t visual_stage_count = std::size_t(VisualStage::Count);
const char* visualStageName(VisualStage stage);
struct VisualStageTiming { bool ran = false; double elapsed_ms = 0; };

// A borrowed, reusable per-frame result. Coordinates are working-image pixels
// (+x right, +y down). Units and masks are described by visualFeaturePlanes().
struct VisualFeatureFrame {
    int schema_version = 1;
    std::uint64_t frame_id = 0;
    double timestamp_s = 0;
    cv::Size source_size, working_size;
    // Maps working pixel centers back to source: p_src = A * [u,v,1].
    cv::Matx23d pixel_to_source = cv::Matx23d::eye();
    std::optional<CameraIntrinsics> camera;
    VisualFeatureSelection requested, computed;
    cv::Mat bgr, gray; // 8-bit, fixed channel order; raw input is never normalized per frame.
    cv::Mat gradient_x, gradient_y, edge_magnitude, edge_orientation_rad, edge_orientation_valid;
    cv::Mat corner_response;
    std::vector<cv::KeyPoint> corners, keypoints; // Shi-Tomasi / FAST; no descriptors.
    cv::Mat color_transition_magnitude;
    cv::Mat texture_orientation_rad, texture_coherence, texture_orientation_valid;
    cv::Mat local_binary_texture, local_binary_valid;
    std::vector<std::vector<cv::Point>> contours; // Canny traces; no semantic objects.
    cv::Mat shape_boundaries; // Rasterized simplified traces, not semantic segmentation.
    bool contours_truncated = false;
    const char* motion_status = "disabled";
    std::optional<MotionEstimate> motion;
    std::vector<TrackedFeature> tracks; // Includes new seeds; age>1 identifies correspondences.
    std::array<VisualStageTiming, visual_stage_count> stages{};
    double total_ms = 0;
    AnalyticalTensor feature_tensor;
    AnalyticalDiagnostics analytical;
    std::vector<NamedStageTiming> analytical_stages;
    RelativePose relative_pose;
    std::vector<TrackedFeature> pose_correspondences; // Same indexing as relative_pose.inliers.
    std::vector<LocalMapPoint> local_map;
    TerrainGrid terrain;
    SurfaceFeatures surface;
    cv::Mat geometry_confidence; // Output-grid confidence, including unsupported zeros.
};

struct VisualFeaturePlane {
    const char* name;
    const char* units;
    cv::Mat values; // Shallow view; validity mask may be empty when all values are defined.
    cv::Mat valid;
};
// Explicitly requested representation adapters: no allocation in extract() for these.
std::vector<VisualFeaturePlane> visualFeaturePlanes(const VisualFeatureFrame& frame);
std::size_t visualFeaturePayloadBytes(const VisualFeatureFrame& frame); // Logical data, not peak RSS.

class VisualFeatureExtractor {
public:
    explicit VisualFeatureExtractor(const VisualFeatureSettings& settings = {},
                                    std::optional<CameraIntrinsics> camera = std::nullopt);
    ~VisualFeatureExtractor();
    VisualFeatureExtractor(const VisualFeatureExtractor&) = delete;
    VisualFeatureExtractor& operator=(const VisualFeatureExtractor&) = delete;
    // OpenCV BGR8 input. Timestamps are finite; calibrated motion requires
    // strictly increasing timestamps and rectified, genuinely sequential frames.
    // Result and shallow Mat copies expire/are overwritten on the next call.
    const VisualFeatureFrame& extract(const cv::Mat& bgr, double timestamp_s,
                                     std::optional<AltitudeSample> altitude = std::nullopt,
                                     std::optional<ImuSample> imu = std::nullopt);
    void resetMotion();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Diagnostic/serialization are explicit off-path operations.
cv::Mat renderVisualFeatures(const cv::Mat& original_bgr, const VisualFeatureFrame& frame,
                            const std::string& motion_note = "");
void writeVisualFeatures(const std::filesystem::path& path, const VisualFeatureFrame& frame,
                         const VisualFeatureSettings& settings);
// Compact training cache export: tensor, validity, registry, scale, and map pose only.
void writeAnalyticalTensor(const std::filesystem::path& path, const VisualFeatureFrame& frame,
                           const VisualFeatureSettings& settings);
// Primary research profile: analytical bank on, legacy comparison bank off.
VisualFeatureSettings analyticalFeatureSettings();
VisualFeatureSettings loadAnalyticalConfig(const std::filesystem::path& path);
std::optional<CameraIntrinsics> loadVisualCalibration(const std::filesystem::path& path,
                                                     std::vector<double>& distortion);
cv::Mat renderAnalyticalFeatures(const VisualFeatureFrame& frame);
} // namespace metric_mapping
