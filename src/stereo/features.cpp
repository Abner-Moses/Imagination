#include "stereo_internal.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <stdexcept>
#include <utility>

namespace metric_mapping::detail {
FeatureMatches findFeatureMatches(const StereoImages& images, const CameraIntrinsics& camera,
                                  const TwoViewSettings& settings, TwoViewDiagnostics& diagnostics)
{
    const cv::Mat& gray_1 = images.gray_first;
    const cv::Mat& gray_2 = images.gray_second;
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(settings.maximum_features);
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    orb->detectAndCompute(gray_1, cv::noArray(), keypoints_1,
                          descriptors_1);
    orb->detectAndCompute(gray_2, cv::noArray(), keypoints_2,
                          descriptors_2);
    if (descriptors_1.empty() || descriptors_2.empty())
        throw std::runtime_error("ORB found no usable descriptors");

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (const auto& neighbors : knn_matches) {
        if (neighbors.size() == 2 &&
            neighbors[0].distance <
                settings.lowe_ratio * neighbors[1].distance)
            good_matches.push_back(neighbors[0]);
    }
    if (good_matches.size() < 8)
        throw std::runtime_error(
            "Too few feature matches after the Lowe ratio test");

    std::vector<cv::Point2f> pixels_1;
    std::vector<cv::Point2f> pixels_2;
    pixels_1.reserve(good_matches.size());
    pixels_2.reserve(good_matches.size());
    for (const cv::DMatch& match : good_matches) {
        pixels_1.push_back(keypoints_1[match.queryIdx].pt);
        pixels_2.push_back(keypoints_2[match.trainIdx].pt);
    }

    const cv::Matx33d camera_matrix(camera.fx, 0.0, camera.cx,
                                    0.0, camera.fy, camera.cy,
                                    0.0, 0.0, 1.0);
    cv::Mat pose_mask;
    cv::Mat essential = cv::findEssentialMat(
        pixels_1, pixels_2, camera_matrix, cv::RANSAC,
        settings.ransac_probability, settings.ransac_threshold_px,
        1000, pose_mask);
    if (essential.empty())
        throw std::runtime_error("Essential-matrix estimation failed");
    const std::size_t ransac_inliers =
        static_cast<std::size_t>(cv::countNonZero(pose_mask));
    if (ransac_inliers < 8)
        throw std::runtime_error("Too few geometrically valid feature matches");
    std::vector<cv::DMatch> final_matches;
    for (std::size_t index = 0; index < good_matches.size(); ++index)
        if (pose_mask.at<std::uint8_t>(static_cast<int>(index)))
            final_matches.push_back(good_matches[index]);

    diagnostics.features_image_1 = keypoints_1.size();
    diagnostics.features_image_2 = keypoints_2.size();
    diagnostics.good_matches = good_matches.size();
    diagnostics.ransac_inliers = ransac_inliers;
    return {std::move(keypoints_1), std::move(keypoints_2), std::move(final_matches)};
}
} // namespace metric_mapping::detail
