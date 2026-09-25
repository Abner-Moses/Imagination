#include "stereo_internal.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metric_mapping::detail {
namespace {
double median(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(),
                                            values.begin() + middle);
        result = (*lower + result) * 0.5;
    }
    return result;
}

} // namespace

RectifiedPair alignPair(const StereoImages& images, const FeatureMatches& features,
                        const CameraIntrinsics& camera, double baseline_m,
                        const TwoViewSettings& settings, TwoViewResult& result)
{
    const auto& image_1 = images.first;
    const auto& image_2 = images.second;
    const auto& gray_1 = images.gray_first;
    const auto& gray_2 = images.gray_second;
    const auto& keypoints_1 = features.first;
    const auto& keypoints_2 = features.second;
    const auto& final_matches = features.verified;
    const cv::Matx33d camera_matrix(camera.fx, 0, camera.cx,
                                    0, camera.fy, camera.cy, 0, 0, 1);
    // A single mostly planar terrain pair is degenerate for unconstrained
    // essential-matrix pose recovery. For the dense metric path we enforce
    // the documented capture contract: both cameras remain nadir-facing with
    // no pitch/roll change. A robust partial-affine fit accounts for small yaw
    // about the optical axis. Image displacement determines only the baseline
    // direction; the caller supplies its measured magnitude.
    std::vector<cv::Point2f> affine_pixels_1;
    std::vector<cv::Point2f> affine_pixels_2;
    affine_pixels_1.reserve(final_matches.size());
    affine_pixels_2.reserve(final_matches.size());
    for (const cv::DMatch& match : final_matches) {
        affine_pixels_1.push_back(keypoints_1[match.queryIdx].pt);
        affine_pixels_2.push_back(keypoints_2[match.trainIdx].pt);
    }
    std::vector<cv::Point2f> normalized_1, normalized_2;
    cv::perspectiveTransform(affine_pixels_1, normalized_1, camera_matrix.inv());
    cv::perspectiveTransform(affine_pixels_2, normalized_2, camera_matrix.inv());
    cv::Mat affine_mask;
    cv::Mat affine = cv::estimateAffinePartial2D(
        normalized_1, normalized_2, affine_mask, cv::RANSAC,
        3.0 / std::max(camera.fx, camera.fy), 2000, 0.99, 10);
    if (affine.empty())
        throw std::runtime_error(
            "Could not estimate the parallel-view image alignment");
    std::vector<cv::DMatch> alignment_matches;
    alignment_matches.reserve(final_matches.size());
    for (std::size_t index = 0; index < final_matches.size(); ++index) {
        if (affine_mask.at<std::uint8_t>(static_cast<int>(index)) != 0)
            alignment_matches.push_back(final_matches[index]);
    }
    if (alignment_matches.size() < 8U)
        throw std::runtime_error(
            "Too few matches support the parallel-view alignment");
    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2,
                    alignment_matches, result.matches_image,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::Mat affine_64;
    affine.convertTo(affine_64, CV_64F);
    const double affine_scale = std::hypot(
        affine_64.at<double>(0, 0), affine_64.at<double>(1, 0));
    if (!std::isfinite(affine_scale) ||
        std::abs(affine_scale - 1.0) > 0.1) {
        throw std::runtime_error(
            "The two images differ in scale; keep the same altitude, zoom, "
            "and resolution for both captures");
    }
    const double cosine = affine_64.at<double>(0, 0) / affine_scale;
    const double sine = affine_64.at<double>(1, 0) / affine_scale;
    const cv::Matx33d rotation_21(cosine, -sine, 0.0,
                              sine, cosine, 0.0,
                              0.0, 0.0, 1.0);

    const cv::Matx33d image_2_to_camera_1 =
        camera_matrix * rotation_21.t() * camera_matrix.inv();
    std::vector<cv::Point2f> aligned_pixels_2;
    cv::perspectiveTransform(affine_pixels_2, aligned_pixels_2,
                             image_2_to_camera_1);

    std::vector<double> horizontal_residuals;
    std::vector<double> vertical_residuals;
    horizontal_residuals.reserve(final_matches.size());
    vertical_residuals.reserve(final_matches.size());
    for (std::size_t index = 0; index < final_matches.size(); ++index) {
        if (!affine_mask.empty() &&
            affine_mask.at<std::uint8_t>(static_cast<int>(index)) == 0) {
            continue;
        }
        horizontal_residuals.push_back(
            affine_pixels_1[index].x - aligned_pixels_2[index].x);
        vertical_residuals.push_back(
            affine_pixels_1[index].y - aligned_pixels_2[index].y);
    }
    const double median_horizontal_residual =
        median(horizontal_residuals);
    const double median_vertical_residual =
        median(vertical_residuals);
    const cv::Vec3d baseline_direction(
        median_horizontal_residual / camera.fx,
        median_vertical_residual / camera.fy, 0.0);
    const double direction_norm = cv::norm(baseline_direction);
    if (!std::isfinite(direction_norm) || direction_norm < 1e-6) {
        throw std::runtime_error(
            "The parallel views have too little image displacement for "
            "dense stereo");
    }
    const cv::Vec3d camera_2_center =
        baseline_m * baseline_direction / direction_norm;
    const cv::Vec3d translation_21 = -(rotation_21 * camera_2_center);

    const double alignment_angle_degrees =
        std::atan2(median_vertical_residual,
                   median_horizontal_residual) *
        180.0 / CV_PI;
    const cv::Mat matching_rotation = cv::getRotationMatrix2D(
        cv::Point2f(static_cast<float>(camera.cx),
                    static_cast<float>(camera.cy)),
        alignment_angle_degrees, 1.0);
    cv::Mat inverse_matching_rotation;
    cv::invertAffineTransform(matching_rotation,
                              inverse_matching_rotation);

    // Match grayscale images and combine camera-2 warps to avoid a second
    // resampling pass and full-size intermediate color images.
    const cv::Matx33d matching_transform(
        matching_rotation.at<double>(0, 0), matching_rotation.at<double>(0, 1),
        matching_rotation.at<double>(0, 2), matching_rotation.at<double>(1, 0),
        matching_rotation.at<double>(1, 1), matching_rotation.at<double>(1, 2),
        0, 0, 1);
    const cv::Matx33d matching_transform_2 = matching_transform * image_2_to_camera_1;
    cv::Mat matching_gray_1;
    cv::Mat matching_gray_2;
    cv::warpAffine(gray_1, matching_gray_1, matching_rotation,
                   image_1.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT);
    cv::warpPerspective(gray_2, matching_gray_2, matching_transform_2,
                        image_2.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::Mat source_mask(image_1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat matching_mask_1;
    cv::Mat matching_mask_2;
    cv::warpAffine(source_mask, matching_mask_1, matching_rotation,
                   image_1.size(), cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT);
    cv::warpPerspective(source_mask, matching_mask_2, matching_transform_2,
                        image_2.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

    const auto transformPixel = [&](const cv::Point2f& point) {
        return cv::Point2f(
            static_cast<float>(matching_rotation.at<double>(0, 0) *
                                   point.x +
                               matching_rotation.at<double>(0, 1) *
                                   point.y +
                               matching_rotation.at<double>(0, 2)),
            static_cast<float>(matching_rotation.at<double>(1, 0) *
                                   point.x +
                               matching_rotation.at<double>(1, 1) *
                                   point.y +
                               matching_rotation.at<double>(1, 2)));
    };
    std::vector<double> epipolar_errors;
    std::vector<double> sparse_disparities;
    epipolar_errors.reserve(final_matches.size());
    sparse_disparities.reserve(final_matches.size());
    for (std::size_t index = 0; index < final_matches.size(); ++index) {
        const cv::Point2f first = transformPixel(affine_pixels_1[index]);
        const cv::Point2f second = transformPixel(aligned_pixels_2[index]);
        const double error = std::abs(first.y - second.y);
        epipolar_errors.push_back(error);
        // Foreground depths need not fit the dominant surface's affine model.
        // Keep every positive disparity satisfying the epipolar constraint.
        if (error <= settings.maximum_rectified_epipolar_error_px &&
            first.x - second.x > 0.5F)
            sparse_disparities.push_back(first.x - second.x);
    }
    const double median_epipolar_error = median(epipolar_errors);
    if (!std::isfinite(median_epipolar_error) ||
        median_epipolar_error >
            settings.maximum_rectified_epipolar_error_px) {
        throw std::runtime_error(
            "Parallel-view alignment is not accurate enough for dense "
            "matching (median epipolar error " +
            std::to_string(median_epipolar_error) + " px)");
    }

    if (sparse_disparities.size() < 8)
        throw std::runtime_error("Too few positive, epipolar-consistent disparities");
    const auto disparity_bounds = std::minmax_element(
        sparse_disparities.begin(), sparse_disparities.end());
    const double low_disparity = *disparity_bounds.first;
    const double high_disparity = *disparity_bounds.second;
    const int minimum_disparity = std::max(0,
        static_cast<int>(std::floor(low_disparity)) - 4);
    const int requested_range = static_cast<int>(std::ceil(high_disparity)) + 5 -
                                minimum_disparity;
    const int number_of_disparities = ((requested_range + 15) / 16) * 16;
    if (minimum_disparity + number_of_disparities >= matching_gray_1.cols ||
        minimum_disparity + number_of_disparities > 2047)
        throw std::runtime_error("Stereo disparity range exceeds the image or matcher limits");
    result.rotation_21 = rotation_21;
    result.translation_21 = translation_21;
    result.camera_positions_world_m = {{0, 0, 0},
        {camera_2_center[0], -camera_2_center[1], -camera_2_center[2]}};
    result.diagnostics.alignment_inliers = alignment_matches.size();
    result.diagnostics.median_rectified_epipolar_error_px = median_epipolar_error;
    return {matching_gray_1, matching_gray_2, matching_mask_1, matching_mask_2,
            inverse_matching_rotation, camera_2_center, minimum_disparity, number_of_disparities};
}
} // namespace metric_mapping::detail
