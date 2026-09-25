#include "stereo_internal.hpp"
#include "metric_mapping/geometry.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

namespace metric_mapping {
namespace {
void validateInputs(const cv::Mat& image_1,
                    const cv::Mat& image_2,
                    const CameraIntrinsics& camera,
                    double baseline_m,
                    const TwoViewSettings& settings)
{
    if (image_1.empty() || image_2.empty())
        throw std::runtime_error("Could not load both input images");
    if (image_1.size() != image_2.size())
        throw std::runtime_error("The two images must have the same size");
    if (camera.width != image_1.cols || camera.height != image_1.rows)
        throw std::runtime_error(
            "Camera width/height do not match the input images");
    validateCamera(camera);
    if (!std::isfinite(baseline_m) || baseline_m <= 0.0)
        throw std::runtime_error(
            "baseline_m must be the measured positive distance between "
            "camera centers");
    for (double value : {double(settings.lowe_ratio), settings.ransac_probability,
                         settings.ransac_threshold_px, settings.minimum_depth_m,
                         settings.maximum_depth_m,
                         settings.maximum_rectified_epipolar_error_px,
                         settings.left_right_disparity_tolerance_px})
        if (!std::isfinite(value))
            throw std::runtime_error("Two-view settings must be finite");
    if (settings.maximum_features < 100 || settings.lowe_ratio <= 0.0F ||
        settings.lowe_ratio >= 1.0F ||
        settings.ransac_probability <= 0.0 ||
        settings.ransac_probability >= 1.0 ||
        settings.ransac_threshold_px <= 0.0 ||
        settings.minimum_depth_m <= 0.0 ||
        settings.maximum_depth_m <= settings.minimum_depth_m ||
        settings.maximum_rectified_epipolar_error_px <= 0.0 ||
        settings.left_right_disparity_tolerance_px <= 0.0 ||
        settings.stereo_block_size < 3 ||
        settings.stereo_block_size > 255 ||
        settings.stereo_block_size % 2 == 0)
        throw std::runtime_error("Invalid two-view reconstruction settings");
}

} // namespace

TwoViewResult reconstructTwoView(const std::filesystem::path& image_1_path,
                                 const std::filesystem::path& image_2_path,
                                 const CameraIntrinsics& camera, double baseline_m,
                                 const TwoViewSettings& settings)
{
    detail::StereoImages images;
    images.first = cv::imread(image_1_path.string(), cv::IMREAD_COLOR);
    images.second = cv::imread(image_2_path.string(), cv::IMREAD_COLOR);
    validateInputs(images.first, images.second, camera, baseline_m, settings);
    cv::cvtColor(images.first, images.gray_first, cv::COLOR_BGR2GRAY);
    cv::cvtColor(images.second, images.gray_second, cv::COLOR_BGR2GRAY);

    TwoViewResult result;
    const auto features = detail::findFeatureMatches(images, camera, settings, result.diagnostics);
    const auto aligned = detail::alignPair(images, features, camera, baseline_m, settings, result);
    detail::reconstructDepth(images, aligned, camera, baseline_m, settings, result);
    return result;
}
} // namespace metric_mapping
