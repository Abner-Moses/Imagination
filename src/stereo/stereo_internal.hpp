#pragma once
#include "metric_mapping/two_view.hpp"

namespace metric_mapping::detail {
struct StereoImages {
    cv::Mat first, second;
    cv::Mat gray_first, gray_second;
};
struct FeatureMatches {
    std::vector<cv::KeyPoint> first, second;
    std::vector<cv::DMatch> verified;
};
struct RectifiedPair {
    cv::Mat gray_first, gray_second;
    cv::Mat valid_first, valid_second;
    cv::Mat to_original_pixels;
    cv::Vec3d camera_2_center;
    int minimum_disparity = 0;
    int disparity_count = 0;
};

// Each stage retains the previous stage's geometric checks.
FeatureMatches findFeatureMatches(const StereoImages& images, const CameraIntrinsics& camera,
                                  const TwoViewSettings& settings, TwoViewDiagnostics& diagnostics);
RectifiedPair alignPair(const StereoImages& images, const FeatureMatches& features,
                        const CameraIntrinsics& camera, double baseline_m,
                        const TwoViewSettings& settings, TwoViewResult& result);
void reconstructDepth(const StereoImages& images, const RectifiedPair& aligned,
                      const CameraIntrinsics& camera, double baseline_m,
                      const TwoViewSettings& settings, TwoViewResult& result);
} // namespace metric_mapping::detail
