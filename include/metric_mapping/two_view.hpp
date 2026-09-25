#pragma once

#include "metric_mapping/types.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <filesystem>
#include <vector>

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
