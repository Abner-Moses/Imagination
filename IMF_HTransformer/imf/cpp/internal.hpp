#pragma once
#include "imagination.hpp"
#include <chrono>
#include <fstream>

namespace metric_mapping::detail {
// Common rules used by both camera pipelines and by the command-line tools.
cv::Size fitImageSize(cv::Size source, int maximum_width, int maximum_height);
cv::Matx23d pixelToSource(cv::Size source, cv::Size working);
CameraIntrinsics resizeCamera(const CameraIntrinsics& camera, cv::Size working);
int parseInteger(const std::string& text,
                 int minimum = std::numeric_limits<int>::min(),
                 int maximum = std::numeric_limits<int>::max());
struct TimingSummary { double mean, median, p95, worst; };
TimingSummary summarizeTimes(std::vector<double>& times); // Sorts the supplied samples.
#ifdef HAVE_MOTION
std::optional<ImuSample> readImuSnapshot(const std::filesystem::path& path);
std::optional<AltitudeSample> readAltitudeSnapshot(const std::filesystem::path& path);
#endif
} // namespace metric_mapping::detail

// Shared implementation details. Applications should include imagination.hpp.
// --- Application / test support: rgbd app ---

namespace metric_mapping::app {
void runRgbd(const std::filesystem::path& config_path);
}

// --- Application / test support: stereo app ---

namespace metric_mapping::app {
void runStereo(const std::filesystem::path& image_1_path,
               const std::filesystem::path& image_2_path,
               const CameraIntrinsics& camera, double baseline_m,
               const std::optional<UltrasonicMeasurement>& ultrasonic = std::nullopt);
void runSyntheticDemo(const std::filesystem::path& image_1_path,
                      const std::filesystem::path& image_2_path);
void printLandingSummary(const LandingAnalysis& analysis);
} // namespace metric_mapping::app

// --- Application / test support: motion sequence ---

namespace metric_mapping::demo {
cv::Mat motionTexture(cv::Size size = {320, 240});
// Warps a fixed original, avoiding cumulative interpolation blur in tests.
void warpMotion(const cv::Mat& source, cv::Mat& destination,
                const CameraIntrinsics& camera, double dx_px, double dy_px,
                double yaw_rad = 0, double scale = 1);
} // namespace metric_mapping::demo

// --- Application / test support: motion internal ---


namespace metric_mapping::motion_detail {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Similarity {
    double a = 1, b = 0; // sR = [a,-b; b,a]
    cv::Vec2d t{0, 0};
    cv::Vec2d apply(const cv::Vec2d& p) const {
        return {a * p[0] - b * p[1] + t[0], b * p[0] + a * p[1] + t[1]};
    }
};
struct GeometryScratch {
    std::vector<cv::Vec2d> first, second;
    std::vector<float> values, errors;
    std::vector<std::uint8_t> mask, trial;
    std::vector<int> cells;
    void reserve(int count, int cell_count);
};
float median(std::vector<float>& values);
Similarity estimateMotion(const std::vector<TrackedFeature>& tracks,
                          const CameraIntrinsics& camera, const OpticalFlowSettings& settings,
                          GeometryScratch& scratch, MotionEstimate& result,
                          std::optional<double> imu_yaw_delta = std::nullopt);
void validateSettings(const OpticalFlowSettings& settings);
cv::Matx33d nadirRotation(double yaw);
} // namespace metric_mapping::motion_detail

// --- Grid Internal ---

namespace metric_mapping::detail {
// Shared precondition for grid processing; no file I/O or reconstruction here.
void validateGrid(const TerrainGrid& grid);
void validateAnalyticalSettings(const VisualFeatureSettings& settings);
struct ProjectedGeometry {
    // Depth, depth d/d(image-plane metre) X/Y, slope, roughness, confidence.
    std::array<cv::Mat, 6> planes, masks;
};
void depthGradients(ProjectedGeometry&, const CameraIntrinsics& output_camera);
ProjectedGeometry projectSurface(const TerrainGrid&, const SurfaceFeatures&,
                                 const cv::Mat& confidence, const CameraIntrinsics&,
                                 const cv::Matx44d& camera_to_world,
                                 cv::Size output_size, double maximum_depth_m,
                                 bool depth_gradients);
cv::Matx44d nadirTransform(const NadirPose& pose);
}

// --- Output Internal ---

namespace metric_mapping::detail {
void requireWritable(const std::ofstream& stream, const std::filesystem::path& path);
cv::Mat terrainValidMask(const TerrainGrid& grid);
cv::Mat colorizeFloat(const cv::Mat& values, const cv::Mat& valid_mask,
                      double minimum, double maximum);
cv::Mat colorizeDem(const cv::Mat& dem, const cv::Mat& valid_mask);
void writeImage(const std::filesystem::path& path, const cv::Mat& image);
cv::Mat labeledPanel(const cv::Mat& image, const std::string& title);
void writeLandingJson(const std::filesystem::path& output_directory,
                      const LandingAnalysis& analysis, const LandingAnalysisConfig& config,
                      const cv::Vec3d& uav_position_world_m, bool demo_only);
} // namespace metric_mapping::detail

// --- Stereo metadata ---

namespace metric_mapping::detail {
void writeTwoViewMetadata(const std::filesystem::path& path,
                          const CameraIntrinsics& camera, double baseline_m,
                          const TwoViewResult& result, const TerrainGrid& grid,
                          const GridStatistics& statistics, std::size_t completed_point_count);
void writeDemoMetadata(const std::filesystem::path& output_directory,
                       const std::filesystem::path& image_1_path,
                       const std::filesystem::path& image_2_path,
                       std::size_t alignment_matches, std::size_t raw_points,
                       std::size_t completed_points, const GridStatistics& statistics);
} // namespace metric_mapping::detail

// --- Synthetic demonstration ---

namespace metric_mapping::app {
// Appearance-derived geometry for exercising exports, never metric stereo.
struct SyntheticScene {
    std::vector<ColoredPoint> points;
    cv::Mat matches_image;
    std::size_t alignment_matches = 0;
};
SyntheticScene makeSyntheticScene(const std::filesystem::path& image_1_path,
                                  const std::filesystem::path& image_2_path);
} // namespace metric_mapping::app
