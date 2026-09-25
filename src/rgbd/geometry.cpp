#include "metric_mapping/geometry.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace metric_mapping {
namespace {

double rawDepthAt(const cv::Mat& depth, int row, int column)
{
    switch (depth.depth()) {
    case CV_16U:
        return depth.at<std::uint16_t>(row, column);
    case CV_32F:
        return depth.at<float>(row, column);
    case CV_64F:
        return depth.at<double>(row, column);
    default:
        throw std::runtime_error(
            "Unsupported depth image type. Use single-channel uint16 PNG "
            "depth or float32/float64 TIFF/EXR depth");
    }
}

bool isExplicitlyInvalid(double raw_value,
                         const std::vector<double>& invalid_values)
{
    for (double invalid : invalid_values) {
        if ((std::isnan(invalid) && std::isnan(raw_value)) ||
            raw_value == invalid) {
            return true;
        }
    }
    return false;
}

double toMeters(double raw_value, DepthUnit unit)
{
    return unit == DepthUnit::Meters ? raw_value : raw_value / 1000.0;
}

}  // namespace

void validateCamera(const CameraIntrinsics& camera)
{
    if (camera.width <= 0 || camera.height <= 0 ||
        !std::isfinite(camera.fx) || !std::isfinite(camera.fy) ||
        !std::isfinite(camera.cx) || !std::isfinite(camera.cy) ||
        camera.fx <= 0.0 || camera.fy <= 0.0) {
        throw std::runtime_error(
            "Camera calibration is invalid: width, height, fx, and fy must "
            "be positive finite values");
    }
    if (camera.cx < 0.0 || camera.cx >= camera.width || camera.cy < 0.0 ||
        camera.cy >= camera.height) {
        throw std::runtime_error(
            "Camera principal point lies outside the calibrated image; check "
            "that calibration and image resolution match");
    }
}

void validateRigidTransform(const cv::Matx44d& transform,
                            const std::string& name)
{
    for (double value : transform.val) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(name + " contains a non-finite value");
        }
    }

    constexpr double tolerance = 1e-5;
    if (std::abs(transform(3, 0)) > tolerance ||
        std::abs(transform(3, 1)) > tolerance ||
        std::abs(transform(3, 2)) > tolerance ||
        std::abs(transform(3, 3) - 1.0) > tolerance) {
        throw std::runtime_error(
            name + " is not homogeneous: final row must be [0, 0, 0, 1]");
    }

    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotation(row, column) = transform(row, column);
        }
    }

    const cv::Matx33d identity_test = rotation.t() * rotation;
    double maximum_error = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const double expected = row == column ? 1.0 : 0.0;
            maximum_error = std::max(
                maximum_error,
                std::abs(identity_test(row, column) - expected));
        }
    }

    const double determinant = cv::determinant(cv::Mat(rotation));
    if (maximum_error > tolerance || std::abs(determinant - 1.0) > tolerance) {
        throw std::runtime_error(
            name + " rotation is not orthonormal with determinant +1");
    }
}

cv::Matx44d invertRigidTransform(const cv::Matx44d& transform)
{
    validateRigidTransform(transform, "Rigid transform");

    cv::Matx33d rotation;
    cv::Vec3d translation;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotation(row, column) = transform(row, column);
        }
        translation[row] = transform(row, 3);
    }

    const cv::Matx33d inverse_rotation = rotation.t();
    const cv::Vec3d inverse_translation = -inverse_rotation * translation;
    cv::Matx44d inverse = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            inverse(row, column) = inverse_rotation(row, column);
        }
        inverse(row, 3) = inverse_translation[row];
    }
    return inverse;
}

cv::Matx44d cameraToWorldTransform(const cv::Matx44d& supplied_pose,
                                   PoseConvention convention)
{
    validateRigidTransform(supplied_pose, "Supplied camera pose");
    const cv::Matx44d camera_to_world =
        convention == PoseConvention::CameraToWorld
            ? supplied_pose
            : invertRigidTransform(supplied_pose);
    validateRigidTransform(camera_to_world, "T_WC");
    return camera_to_world;
}

cv::Vec3d backProjectPixel(double u,
                           double v,
                           double depth_m,
                           const CameraIntrinsics& camera)
{
    validateCamera(camera);
    if (!std::isfinite(depth_m) || depth_m <= 0.0) {
        throw std::runtime_error(
            "Back-projection requires positive finite metric depth");
    }
    return cv::Vec3d((u - camera.cx) * depth_m / camera.fx,
                     (v - camera.cy) * depth_m / camera.fy,
                     depth_m);
}

cv::Vec3d transformPoint(const cv::Matx44d& transform,
                         const cv::Vec3d& point)
{
    const cv::Vec4d homogeneous(point[0], point[1], point[2], 1.0);
    const cv::Vec4d transformed = transform * homogeneous;
    return cv::Vec3d(transformed[0], transformed[1], transformed[2]);
}

FrameResult backProjectFrame(const cv::Mat& rgb,
                             const cv::Mat& depth,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world)
{
    validateCamera(camera);
    validateRigidTransform(camera_to_world, "T_WC");

    if (rgb.type() != CV_8UC3 || depth_config.pixel_stride < 1 ||
        !std::isfinite(depth_config.min_depth_m) ||
        !std::isfinite(depth_config.max_depth_m) ||
        depth_config.min_depth_m <= 0.0 ||
        depth_config.max_depth_m <= depth_config.min_depth_m)
        throw std::runtime_error("Invalid RGB type or depth configuration");
    if (depth.channels() != 1) {
        throw std::runtime_error("Depth image must be single-channel");
    }
    if (rgb.cols != camera.width || rgb.rows != camera.height ||
        depth.cols != camera.width || depth.rows != camera.height) {
        throw std::runtime_error(
            "RGB/depth dimensions do not match the calibrated camera size " +
            std::to_string(camera.width) + "x" +
            std::to_string(camera.height));
    }

    FrameResult result;
    result.diagnostics.width = rgb.cols;
    result.diagnostics.height = rgb.rows;
    std::vector<double> valid_depths;
    valid_depths.reserve(depth.total());
    const std::size_t stride = static_cast<std::size_t>(depth_config.pixel_stride);
    result.world_points.reserve(
        (1U + (depth.rows - 1U) / stride) *
        (1U + (depth.cols - 1U) / stride));

    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            const double raw_depth = rawDepthAt(depth, v, u);
            const double depth_m = toMeters(raw_depth, depth_config.unit);
            const bool valid =
                std::isfinite(raw_depth) &&
                !isExplicitlyInvalid(raw_depth,
                                     depth_config.invalid_values) &&
                std::isfinite(depth_m) && depth_m > 0.0 &&
                depth_m >= depth_config.min_depth_m &&
                depth_m <= depth_config.max_depth_m;

            if (!valid) {
                ++result.diagnostics.invalid_depth_pixels;
                continue;
            }

            ++result.diagnostics.valid_depth_pixels;
            valid_depths.push_back(depth_m);

            if (u % depth_config.pixel_stride != 0 ||
                v % depth_config.pixel_stride != 0) {
                continue;
            }

            const cv::Vec3d camera_point =
                backProjectPixel(u, v, depth_m, camera);
            const cv::Vec3d world_point =
                transformPoint(camera_to_world, camera_point);
            if (!std::isfinite(world_point[0]) ||
                !std::isfinite(world_point[1]) ||
                !std::isfinite(world_point[2])) {
                throw std::runtime_error(
                    "A validated depth and pose produced a non-finite world "
                    "point");
            }

            const cv::Vec3b bgr = rgb.at<cv::Vec3b>(v, u);
            result.world_points.push_back(
                {static_cast<float>(world_point[0]),
                 static_cast<float>(world_point[1]),
                 static_cast<float>(world_point[2]),
                 bgr[2], bgr[1], bgr[0]});
        }
    }

    if (valid_depths.empty()) {
        throw std::runtime_error(
            "Depth frame contains no valid metric samples after configured "
            "validation");
    }
    if (result.world_points.empty()) {
        throw std::runtime_error(
            "Depth frame has valid values but pixel_stride emitted no points");
    }

    const auto middle = valid_depths.begin() + valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), middle, valid_depths.end());
    result.diagnostics.minimum_depth_m =
        *std::min_element(valid_depths.begin(), valid_depths.end());
    result.diagnostics.maximum_depth_m =
        *std::max_element(valid_depths.begin(), valid_depths.end());
    result.diagnostics.median_depth_m = *middle;
    result.diagnostics.emitted_points = result.world_points.size();
    return result;
}

FrameResult reconstructFrame(const std::filesystem::path& rgb_path,
                             const std::filesystem::path& depth_path,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world)
{
    const cv::Mat rgb = cv::imread(rgb_path.string(), cv::IMREAD_COLOR);
    const cv::Mat depth = cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
    if (rgb.empty()) {
        throw std::runtime_error("Could not load RGB image: " +
                                 rgb_path.string());
    }
    if (depth.empty()) {
        throw std::runtime_error("Could not load depth image: " +
                                 depth_path.string());
    }
    return backProjectFrame(rgb, depth, camera, depth_config,
                            camera_to_world);
}

}  // namespace metric_mapping
