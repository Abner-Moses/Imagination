#include "stereo_internal.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metric_mapping::detail {
void reconstructDepth(const StereoImages& images, const RectifiedPair& aligned,
                      const CameraIntrinsics& camera, double baseline_m,
                      const TwoViewSettings& settings, TwoViewResult& result)
{
    const auto& image_1 = images.first;
    const auto& image_2 = images.second;
    const auto& matching_gray_1 = aligned.gray_first;
    const auto& matching_gray_2 = aligned.gray_second;
    const auto& matching_mask_1 = aligned.valid_first;
    const auto& matching_mask_2 = aligned.valid_second;
    const auto& inverse_matching_rotation = aligned.to_original_pixels;
    const auto& camera_2_center = aligned.camera_2_center;
    const int minimum_disparity = aligned.minimum_disparity;
    const int number_of_disparities = aligned.disparity_count;
    const int right_minimum_disparity =
        -minimum_disparity - number_of_disparities + 1;

    const int block_size = settings.stereo_block_size;
    const auto create_matcher = [&](int minimum) {
        return cv::StereoSGBM::create(
            minimum, number_of_disparities, block_size,
            8 * block_size * block_size,
            32 * block_size * block_size, 1, 31, 15, 200, 2,
            cv::StereoSGBM::MODE_SGBM_3WAY);
    };
    const cv::Ptr<cv::StereoSGBM> left_matcher =
        create_matcher(minimum_disparity);
    const cv::Ptr<cv::StereoSGBM> right_matcher =
        create_matcher(right_minimum_disparity);
    cv::Mat disparity_left_raw;
    cv::Mat disparity_right_raw;
    left_matcher->compute(matching_gray_1, matching_gray_2,
                          disparity_left_raw);
    right_matcher->compute(matching_gray_2, matching_gray_1,
                           disparity_right_raw);

    const cv::Matx33d camera_1_to_world(1.0, 0.0, 0.0,
                                        0.0, -1.0, 0.0,
                                        0.0, 0.0, -1.0);
    result.points.reserve(image_1.total());
    cv::Mat disparity_valid(image_1.size(), CV_8UC1, cv::Scalar(0));
    float preview_minimum = std::numeric_limits<float>::infinity();
    float preview_maximum = -std::numeric_limits<float>::infinity();
    const short invalid_left = static_cast<short>(
        (minimum_disparity - 1) * 16);
    const short invalid_right = static_cast<short>(
        (right_minimum_disparity - 1) * 16);
    const cv::Vec3d unit_baseline = camera_2_center / baseline_m;
    const double effective_focal = std::hypot(
        camera.fx * unit_baseline[0],
        camera.fy * unit_baseline[1]);

    for (int row = 0; row < image_1.rows; ++row) {
        for (int column = 0; column < image_1.cols; ++column) {
            if (matching_mask_1.at<std::uint8_t>(row, column) == 0)
                continue;
            const short left_raw =
                disparity_left_raw.at<short>(row, column);
            if (left_raw <= invalid_left)
                continue;
            const float left_disparity = left_raw / 16.0F;
            if (left_disparity <= 0.5F)
                continue;
            const int right_column = cvRound(column - left_disparity);
            if (right_column < 0 || right_column >= image_2.cols ||
                matching_mask_2.at<std::uint8_t>(row, right_column) == 0) {
                continue;
            }
            const short right_raw =
                disparity_right_raw.at<short>(row, right_column);
            if (right_raw <= invalid_right)
                continue;
            const float right_disparity = right_raw / 16.0F;
            if (std::abs(left_disparity + right_disparity) >
                settings.left_right_disparity_tolerance_px) {
                continue;
            }

            const double depth_m = effective_focal * baseline_m /
                                   left_disparity;
            if (!std::isfinite(depth_m) ||
                depth_m < settings.minimum_depth_m ||
                depth_m > settings.maximum_depth_m) {
                continue;
            }
            const double original_u =
                inverse_matching_rotation.at<double>(0, 0) * column +
                inverse_matching_rotation.at<double>(0, 1) * row +
                inverse_matching_rotation.at<double>(0, 2);
            const double original_v =
                inverse_matching_rotation.at<double>(1, 0) * column +
                inverse_matching_rotation.at<double>(1, 1) * row +
                inverse_matching_rotation.at<double>(1, 2);
            const int source_x = cvRound(original_u);
            const int source_y = cvRound(original_v);
            if (source_x < 0 || source_x >= image_1.cols ||
                source_y < 0 || source_y >= image_1.rows) {
                continue;
            }
            const cv::Vec3d point_camera_1(
                (original_u - camera.cx) * depth_m / camera.fx,
                (original_v - camera.cy) * depth_m / camera.fy,
                depth_m);
            const cv::Vec3d point_world =
                camera_1_to_world * point_camera_1;
            const cv::Vec3b bgr = image_1.at<cv::Vec3b>(source_y, source_x);
            result.points.push_back({static_cast<float>(point_world[0]),
                                     static_cast<float>(point_world[1]),
                                     static_cast<float>(point_world[2]),
                                     bgr[2], bgr[1], bgr[0]});
            disparity_valid.at<std::uint8_t>(row, column) = 255;
            preview_minimum = std::min(preview_minimum, left_disparity);
            preview_maximum = std::max(preview_maximum, left_disparity);
        }
    }
    if (result.points.size() < 500U)
        throw std::runtime_error(
            "Dense stereo produced too few consistent depth samples; check overlap, texture, and calibration");

    const double preview_span =
        std::max(static_cast<double>(preview_maximum - preview_minimum),
                 1e-6);
    cv::Mat disparity_gray;
    disparity_left_raw.convertTo(
        disparity_gray, CV_8U, 255.0 / (16.0 * preview_span),
        -255.0 * preview_minimum / preview_span);
    cv::applyColorMap(disparity_gray, result.disparity_preview,
                      cv::COLORMAP_TURBO);
    result.disparity_preview.setTo(cv::Scalar(0, 0, 0),
                                   disparity_valid == 0);

    result.diagnostics.dense_stereo_points = result.points.size();
    result.diagnostics.final_points = result.points.size();
}
} // namespace metric_mapping::detail
