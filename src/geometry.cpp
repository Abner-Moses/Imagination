#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#ifdef HAVE_TWO_VIEW
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#endif

// --- Geometry ---

namespace metric_mapping::detail {
cv::Size fitImageSize(cv::Size source, int maximum_width, int maximum_height)
{
    if (source.width <= 0 || source.height <= 0 || maximum_width <= 0 || maximum_height <= 0)
        throw std::runtime_error("Image dimensions must be positive");
    const double scale = std::min({1.0, double(maximum_width) / source.width,
                                  double(maximum_height) / source.height});
    return {std::max(1, int(std::lround(source.width * scale))),
            std::max(1, int(std::lround(source.height * scale)))};
}

cv::Matx23d pixelToSource(cv::Size source, cv::Size working)
{
    if (source.width <= 0 || source.height <= 0 || working.width <= 0 || working.height <= 0)
        throw std::runtime_error("Image dimensions must be positive");
    const double sx = double(source.width) / working.width;
    const double sy = double(source.height) / working.height;
    // Resize maps pixel CENTERS: source_u = sx * (working_u + 0.5) - 0.5.
    return {sx, 0, (sx - 1) / 2, 0, sy, (sy - 1) / 2};
}

CameraIntrinsics resizeCamera(const CameraIntrinsics& camera, cv::Size working)
{
    const auto transform = pixelToSource({camera.width, camera.height}, working);
    auto result = camera;
    result.width = working.width;
    result.height = working.height;
    result.fx /= transform(0, 0);
    result.fy /= transform(1, 1);
    result.cx = (camera.cx + 0.5) / transform(0, 0) - 0.5;
    result.cy = (camera.cy + 0.5) / transform(1, 1) - 0.5;
    return result;
}
} // namespace metric_mapping::detail

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

// --- Point Cloud ---

namespace metric_mapping {

std::size_t VoxelGridAccumulator::KeyHash::operator()(const Key& key) const
{
    const std::size_t hx = std::hash<std::int64_t>{}(key.x);
    const std::size_t hy = std::hash<std::int64_t>{}(key.y);
    const std::size_t hz = std::hash<std::int64_t>{}(key.z);
    return hx ^ (hy + 0x9e3779b9U + (hx << 6U) + (hx >> 2U)) ^
           (hz + 0x9e3779b9U + (hy << 6U) + (hy >> 2U));
}

VoxelGridAccumulator::VoxelGridAccumulator(double voxel_size_m,
                                           std::size_t max_voxels)
    : voxel_size_m_(voxel_size_m), max_voxels_(max_voxels)
{
    if (!std::isfinite(voxel_size_m_) || voxel_size_m_ <= 0.0 ||
        max_voxels_ == 0) {
        throw std::runtime_error(
            "Voxel size and maximum voxel count must be positive");
    }
}

void VoxelGridAccumulator::add(const ColoredPoint& point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
        throw std::runtime_error("Cannot fuse a non-finite point");
    }

    const Key key{
        static_cast<std::int64_t>(std::floor(point.x / voxel_size_m_)),
        static_cast<std::int64_t>(std::floor(point.y / voxel_size_m_)),
        static_cast<std::int64_t>(std::floor(point.z / voxel_size_m_))};

    auto iterator = voxels_.find(key);
    if (iterator == voxels_.end()) {
        if (voxels_.size() >= max_voxels_) {
            throw std::runtime_error(
                "Voxel map exceeded fusion.max_voxels; increase the limit or "
                "increase voxel_size_m");
        }
        iterator = voxels_.emplace(key, Accumulator{}).first;
    }

    Accumulator& accumulator = iterator->second;
    accumulator.x += point.x;
    accumulator.y += point.y;
    accumulator.z += point.z;
    accumulator.r += point.r;
    accumulator.g += point.g;
    accumulator.b += point.b;
    ++accumulator.count;
}

void VoxelGridAccumulator::add(const std::vector<ColoredPoint>& points)
{
    for (const ColoredPoint& point : points) {
        add(point);
    }
}

std::vector<ColoredPoint> VoxelGridAccumulator::points() const
{
    std::vector<ColoredPoint> filtered;
    filtered.reserve(voxels_.size());

    for (const auto& entry : voxels_) {
        const Accumulator& accumulator = entry.second;
        const double inverse_count =
            1.0 / static_cast<double>(accumulator.count);
        const auto color = [&](double sum) {
            return static_cast<std::uint8_t>(std::clamp(
                std::lround(sum * inverse_count), 0L, 255L));
        };
        filtered.push_back(
            {static_cast<float>(accumulator.x * inverse_count),
             static_cast<float>(accumulator.y * inverse_count),
             static_cast<float>(accumulator.z * inverse_count),
             color(accumulator.r),
             color(accumulator.g),
             color(accumulator.b)});
    }

    std::sort(filtered.begin(), filtered.end(),
              [](const ColoredPoint& left, const ColoredPoint& right) {
                  if (left.z != right.z)
                      return left.z < right.z;
                  if (left.y != right.y)
                      return left.y < right.y;
                  return left.x < right.x;
              });
    return filtered;
}

std::size_t VoxelGridAccumulator::voxelCount() const
{
    return voxels_.size();
}

CloudBounds computeBounds(const std::vector<ColoredPoint>& points)
{
    if (points.empty()) {
        throw std::runtime_error("Cannot compute bounds of an empty cloud");
    }

    CloudBounds bounds;
    bounds.minimum = cv::Vec3d(points.front().x,
                               points.front().y,
                               points.front().z);
    bounds.maximum = bounds.minimum;

    for (const ColoredPoint& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
            throw std::runtime_error("Cannot compute bounds of a non-finite point");
        bounds.minimum[0] = std::min(bounds.minimum[0],
                                     static_cast<double>(point.x));
        bounds.minimum[1] = std::min(bounds.minimum[1],
                                     static_cast<double>(point.y));
        bounds.minimum[2] = std::min(bounds.minimum[2],
                                     static_cast<double>(point.z));
        bounds.maximum[0] = std::max(bounds.maximum[0],
                                     static_cast<double>(point.x));
        bounds.maximum[1] = std::max(bounds.maximum[1],
                                     static_cast<double>(point.y));
        bounds.maximum[2] = std::max(bounds.maximum[2],
                                     static_cast<double>(point.z));
    }
    return bounds;
}

}  // namespace metric_mapping

#ifdef HAVE_TWO_VIEW
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

// --- Features ---

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

// --- Alignment ---

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

// --- Dense Depth ---

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

// --- Reconstruction ---

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

// --- Synthetic Scene ---

namespace metric_mapping::app {
SyntheticScene makeSyntheticScene(const std::filesystem::path& image_1_path,
                                  const std::filesystem::path& image_2_path)
{
    using namespace metric_mapping;
    const cv::Mat image_1 = cv::imread(image_1_path.string(),
                                       cv::IMREAD_COLOR);
    const cv::Mat image_2 = cv::imread(image_2_path.string(),
                                       cv::IMREAD_COLOR);
    if (image_1.empty() || image_2.empty())
        throw std::runtime_error("Could not load both demo images");
    if (image_1.size() != image_2.size())
        throw std::runtime_error("Demo images must have the same size");

    cv::Mat gray_1;
    cv::Mat gray_2;
    cv::cvtColor(image_1, gray_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_2, gray_2, cv::COLOR_BGR2GRAY);
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    orb->detectAndCompute(gray_1, cv::noArray(), keypoints_1,
                          descriptors_1);
    orb->detectAndCompute(gray_2, cv::noArray(), keypoints_2,
                          descriptors_2);
    if (descriptors_1.empty() || descriptors_2.empty())
        throw std::runtime_error("Demo alignment found no descriptors");

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> neighbors;
    matcher.knnMatch(descriptors_1, descriptors_2, neighbors, 2);
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pixels_1;
    std::vector<cv::Point2f> pixels_2;
    for (const auto& pair : neighbors) {
        if (pair.size() == 2 && pair[0].distance < 0.75F * pair[1].distance) {
            good_matches.push_back(pair[0]);
            pixels_1.push_back(keypoints_1[pair[0].queryIdx].pt);
            pixels_2.push_back(keypoints_2[pair[0].trainIdx].pt);
        }
    }
    if (good_matches.size() < 8U)
        throw std::runtime_error("Too few demo image matches");

    cv::Mat homography_mask;
    const cv::Mat image_2_to_image_1 = cv::findHomography(
        pixels_2, pixels_1, cv::RANSAC, 3.0, homography_mask);
    if (image_2_to_image_1.empty())
        throw std::runtime_error("Could not align the two demo images");

    cv::Mat aligned_2;
    cv::warpPerspective(image_2, aligned_2, image_2_to_image_1,
                        image_1.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);
    cv::Mat source_mask(image_1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat aligned_mask;
    cv::warpPerspective(source_mask, aligned_mask, image_2_to_image_1,
                        image_1.size(), cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT);
    cv::Mat average;
    cv::addWeighted(image_1, 0.5, aligned_2, 0.5, 0.0, average);
    cv::Mat blended = image_1.clone();
    average.copyTo(blended, aligned_mask);

    std::vector<cv::DMatch> inlier_matches;
    for (std::size_t index = 0; index < good_matches.size(); ++index) {
        if (homography_mask.at<std::uint8_t>(static_cast<int>(index)) != 0)
            inlier_matches.push_back(good_matches[index]);
    }
    cv::Mat matches_image;
    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2,
                    inlier_matches, matches_image, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::Mat gray_float;
    cv::cvtColor(blended, gray_float, cv::COLOR_BGR2GRAY);
    gray_float.convertTo(gray_float, CV_32F, 1.0 / 255.0);
    cv::Mat broad;
    cv::Mat local;
    cv::GaussianBlur(gray_float, broad, cv::Size(0, 0), 14.0);
    cv::GaussianBlur(gray_float, local, cv::Size(0, 0), 2.0);
    const float broad_mean = static_cast<float>(cv::mean(broad)[0]);

    constexpr int pixel_stride = 1;
    constexpr double xy_resolution_m = 0.05;
    std::vector<ColoredPoint> synthetic_points;
    synthetic_points.reserve(image_1.total() /
                             (pixel_stride * pixel_stride));
    for (int row = 0; row < image_1.rows; row += pixel_stride) {
        for (int column = 0; column < image_1.cols;
             column += pixel_stride) {
            const float low_frequency = broad.at<float>(row, column) -
                                        broad_mean;
            const float local_contrast = std::abs(
                gray_float.at<float>(row, column) -
                local.at<float>(row, column));
            const float synthetic_height = std::clamp(
                0.35F + 0.8F * low_frequency +
                    2.5F * local_contrast,
                0.0F, 1.5F);
            const cv::Vec3b bgr = blended.at<cv::Vec3b>(row, column);
            synthetic_points.push_back({
                static_cast<float>((column - 0.5 * image_1.cols) *
                                   xy_resolution_m),
                static_cast<float>((0.5 * image_1.rows - row) *
                                   xy_resolution_m),
                synthetic_height, bgr[2], bgr[1], bgr[0]});
        }
    }

    return {std::move(synthetic_points), matches_image, inlier_matches.size()};
}
} // namespace metric_mapping::app

#endif

#ifdef HAVE_VISUAL_FEATURES
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
namespace metric_mapping {
RelativePose estimateRelativePose(const std::vector<TrackedFeature>& tracks,
                                 const CameraIntrinsics& camera, const RelativePoseSettings& s)
{
    validateCamera(camera);
    if (s.minimum_inliers < 8 || s.maximum_iterations < 1 || s.maximum_iterations > 10000 ||
        s.maximum_points < 0 || s.maximum_points > 1000 || !std::isfinite(s.threshold_px) || s.threshold_px <= 0 ||
        !std::isfinite(s.minimum_parallax_rad) || s.minimum_parallax_rad <= 0 || s.minimum_parallax_rad >= CV_PI/2 ||
        !std::isfinite(s.maximum_reprojection_error_px) || s.maximum_reprojection_error_px <= 0 ||
        !std::isfinite(s.maximum_relative_depth) || s.maximum_relative_depth <= 0)
        throw std::runtime_error("Invalid relative pose settings");
    RelativePose out;
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    const auto finish = [&] {
        out.pose_ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count()-out.triangulation_ms;
        return out;
    };
    out.inliers.assign(tracks.size(), 0);
    std::vector<cv::Point2d> a,b;
    std::vector<std::size_t> indices;
    for (std::size_t i=0;i<tracks.size();++i) {
        const auto& t=tracks[i];
        if(t.age<2) continue;
        const auto valid=[&](cv::Point2f p){return std::isfinite(p.x)&&std::isfinite(p.y)&&p.x>=0&&p.y>=0&&p.x<camera.width&&p.y<camera.height;};
        if(!valid(t.previous_px)||!valid(t.current_px)) continue;
        a.emplace_back((t.previous_px.x-camera.cx)/camera.fx,(t.previous_px.y-camera.cy)/camera.fy);
        b.emplace_back((t.current_px.x-camera.cx)/camera.fx,(t.current_px.y-camera.cy)/camera.fy);
        indices.push_back(i);
    }
    if(a.size()<std::size_t(s.minimum_inliers)) return finish();
    cv::Mat mask, rotation, translation;
    const auto essential=cv::findEssentialMat(a,b,1.0,{0,0},cv::RANSAC,0.999,
        s.threshold_px/std::max(camera.fx,camera.fy),s.maximum_iterations,mask);
    if(essential.rows!=3||essential.cols!=3) return finish();
    cv::recoverPose(essential,a,b,rotation,translation,1.0,{0,0},mask);
    if(cv::countNonZero(mask)<s.minimum_inliers) return finish();
    cv::Matx33d r; rotation.copyTo(cv::Mat(r,false));
    cv::Vec3d t(translation.at<double>(0),translation.at<double>(1),translation.at<double>(2));
    const cv::Matx34d p1(1,0,0,0, 0,1,0,0, 0,0,1,0);
    const cv::Matx34d p2(r(0,0),r(0,1),r(0,2),t[0],r(1,0),r(1,1),r(1,2),t[1],r(2,0),r(2,1),r(2,2),t[2]);
    cv::Mat homogeneous;
    const auto triangulation_start=Clock::now();
    cv::triangulatePoints(p1,p2,a,b,homogeneous);
    double error_sum=0; std::size_t accepted=0;
    for(std::size_t i=0;i<a.size();++i) {
        if(!mask.at<unsigned char>(int(i))) continue;
        const double w=homogeneous.at<double>(3,int(i));
        if(!std::isfinite(w)||std::abs(w)<1e-12) continue;
        cv::Vec3d x(homogeneous.at<double>(0,int(i))/w,homogeneous.at<double>(1,int(i))/w,homogeneous.at<double>(2,int(i))/w);
        cv::Vec3d y=r*x+t;
        if(!std::isfinite(cv::norm(x))||x[2]<=0||y[2]<=0||x[2]>s.maximum_relative_depth||y[2]>s.maximum_relative_depth) continue;
        cv::Vec3d ray1(a[i].x,a[i].y,1),ray2=r.t()*cv::Vec3d(b[i].x,b[i].y,1);
        const double angle=std::acos(std::clamp(ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2)),-1.0,1.0));
        if(angle<s.minimum_parallax_rad) continue;
        const double e1=std::hypot(camera.fx*(x[0]/x[2]-a[i].x),camera.fy*(x[1]/x[2]-a[i].y));
        const double e2=std::hypot(camera.fx*(y[0]/y[2]-b[i].x),camera.fy*(y[1]/y[2]-b[i].y));
        if(std::max(e1,e2)>s.maximum_reprojection_error_px) continue;
        out.inliers[indices[i]]=1; ++accepted; error_sum+=std::max(e1,e2);
        if(out.relative_points.size()<std::size_t(s.maximum_points)) out.relative_points.push_back(x);
    }
    out.triangulation_ms=std::chrono::duration<double,std::milli>(Clock::now()-triangulation_start).count();
    if(accepted<std::size_t(s.minimum_inliers)) {out.relative_points.clear();return finish();}
    out.valid=true; out.rotation_21=r; out.translation_direction=t;
    out.mean_reprojection_error_px=error_sum/accepted;
    out.confidence=double(accepted)/a.size()/(1+out.mean_reprojection_error_px/s.threshold_px);
    // Unit baseline is NOT metric scale. No range/IMU heuristic is applied here.
    return finish();
}
} // namespace metric_mapping

namespace metric_mapping::detail {
cv::Matx44d nadirTransform(const NadirPose& p)
{
    const double c=std::cos(p.yaw_rad),s=std::sin(p.yaw_rad);
    return {c,s,0,p.position_m[0], s,-c,0,p.position_m[1], 0,0,-1,p.position_m[2], 0,0,0,1};
}
void depthGradients(ProjectedGeometry& out,const CameraIntrinsics& k)
{
        cv::Sobel(out.planes[0],out.planes[1],CV_32F,1,0,3,1.0/8);
        cv::Sobel(out.planes[0],out.planes[2],CV_32F,0,1,3,1.0/8);
        cv::erode(out.masks[0],out.masks[1],cv::Mat::ones(3,3,CV_8U),{-1,-1},1,cv::BORDER_CONSTANT,0);
        out.masks[1].copyTo(out.masks[2]);
        for(int y=0;y<out.planes[0].rows;++y)for(int x=0;x<out.planes[0].cols;++x){
            const double depth=out.planes[0].at<float>(y,x);
            const bool valid=out.masks[1].at<unsigned char>(y,x);
            out.planes[1].at<float>(y,x)=valid?float(out.planes[1].at<float>(y,x)*k.fx/depth):0;
            out.planes[2].at<float>(y,x)=valid?float(out.planes[2].at<float>(y,x)*k.fy/depth):0;
        }
}
ProjectedGeometry projectSurface(const TerrainGrid& grid, const SurfaceFeatures& surface,
                                 const cv::Mat& confidence, const CameraIntrinsics& camera,
                                 const cv::Matx44d& camera_to_world, cv::Size size,
                                 double maximum_depth, bool gradients)
{
    validateGrid(grid); validateCamera(camera); validateRigidTransform(camera_to_world,"projection pose");
    if(size.width<1||size.height<1||!std::isfinite(maximum_depth)||maximum_depth<=0 ||
       confidence.type()!=CV_32F||confidence.size()!=cv::Size(grid.width,grid.height))
        throw std::runtime_error("Invalid geometry projection inputs");
    ProjectedGeometry out;
    for(int i=0;i<6;++i) {out.planes[i]=cv::Mat::zeros(size,CV_32F);out.masks[i]=cv::Mat::zeros(size,CV_8U);}
    // Confidence is always defined: zero explicitly means no trustworthy support.
    out.masks[5].setTo(255);
    const auto k=resizeCamera(camera,size);
    const auto world_to_camera=invertRigidTransform(camera_to_world);
    struct Vertex { cv::Point2d pixel; double z=0,confidence=0,slope=0,roughness=0; bool slope_valid=false,roughness_valid=false; };
    std::vector<Vertex> vertices(grid.elevation_m.size());
    cv::Mat zbuffer(size,CV_64F,cv::Scalar(std::numeric_limits<double>::infinity()));
    for(int row=0;row<grid.height;++row) for(int col=0;col<grid.width;++col) {
        const auto i=grid.index(row,col);auto& v=vertices[i];
        if(!grid.validity[i]||!std::isfinite(grid.elevation_m[i])) continue;
        const auto p=transformPoint(world_to_camera,{grid.minimum_x_m+col*grid.resolution_m,
            grid.maximum_y_m-row*grid.resolution_m,grid.elevation_m[i]});
        if(p[2]<=1e-6||p[2]>maximum_depth||!std::isfinite(cv::norm(p)))continue;
        v.z=p[2];v.pixel={k.fx*p[0]/p[2]+k.cx,k.fy*p[1]/p[2]+k.cy};
        v.confidence=std::clamp(double(confidence.at<float>(row,col)),0.0,1.0);
        v.slope_valid=!surface.slope_rad.empty()&&surface.gradient_valid.at<unsigned char>(row,col);
        v.roughness_valid=!surface.roughness_m.empty()&&surface.roughness_valid.at<unsigned char>(row,col);
        if(v.slope_valid)v.slope=surface.slope_rad.at<float>(row,col);
        if(v.roughness_valid)v.roughness=surface.roughness_m.at<float>(row,col);
    }
    const auto write=[&](int x,int y,double z,double conf,double slope,double rough,bool sv,bool rv) {
        if(x<0||y<0||x>=size.width||y>=size.height||conf<=0||z>=zbuffer.at<double>(y,x))return;
        zbuffer.at<double>(y,x)=z;
        out.planes[0].at<float>(y,x)=float(z);out.masks[0].at<unsigned char>(y,x)=255;
        out.planes[5].at<float>(y,x)=float(conf);
        out.planes[3].at<float>(y,x)=sv?float(slope):0;out.masks[3].at<unsigned char>(y,x)=sv?255:0;
        out.planes[4].at<float>(y,x)=rv?float(rough):0;out.masks[4].at<unsigned char>(y,x)=rv?255:0;
    };
    const auto triangle=[&](const Vertex& a,const Vertex& b,const Vertex& c) {
        if(a.z<=0||b.z<=0||c.z<=0||a.confidence<=0||b.confidence<=0||c.confidence<=0)return;
        const auto edge=[](cv::Point2d a,cv::Point2d b,cv::Point2d p){return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);};
        const double area=edge(a.pixel,b.pixel,c.pixel);if(std::abs(area)<1e-9)return;
        const double minx=std::max(0.0,std::ceil(std::min({a.pixel.x,b.pixel.x,c.pixel.x})));
        const double maxx=std::min(double(size.width-1),std::floor(std::max({a.pixel.x,b.pixel.x,c.pixel.x})));
        const double miny=std::max(0.0,std::ceil(std::min({a.pixel.y,b.pixel.y,c.pixel.y})));
        const double maxy=std::min(double(size.height-1),std::floor(std::max({a.pixel.y,b.pixel.y,c.pixel.y})));
        if(minx>maxx||miny>maxy)return;
        for(int y=int(miny);y<=int(maxy);++y)for(int x=int(minx);x<=int(maxx);++x){
            const cv::Point2d p(x,y);double wa=edge(b.pixel,c.pixel,p)/area,wb=edge(c.pixel,a.pixel,p)/area,wc=1-wa-wb;
            if(std::min({wa,wb,wc}) < -1e-8)continue;
            // Perspective-correct interpolation and nearest-depth occlusion.
            wa/=a.z;wb/=b.z;wc/=c.z;const double invz=wa+wb+wc;
            if(invz<=0)continue;const double z=1/invz;wa*=z;wb*=z;wc*=z;
            write(x,y,z,wa*a.confidence+wb*b.confidence+wc*c.confidence,
                wa*a.slope+wb*b.slope+wc*c.slope,wa*a.roughness+wb*b.roughness+wc*c.roughness,
                a.slope_valid&&b.slope_valid&&c.slope_valid,a.roughness_valid&&b.roughness_valid&&c.roughness_valid);
        }
    };
    for(int y=0;y+1<grid.height;++y)for(int x=0;x+1<grid.width;++x){
        const auto a=grid.index(y,x),b=grid.index(y,x+1),c=grid.index(y+1,x),d=grid.index(y+1,x+1);
        triangle(vertices[a],vertices[b],vertices[c]);triangle(vertices[b],vertices[d],vertices[c]);
    }
    // Isolated supported cells remain sparse samples; do not splat across holes.
    for(const auto& v:vertices)if(v.z>0&&v.pixel.x>=-0.5&&v.pixel.x<size.width-0.5&&v.pixel.y>=-0.5&&v.pixel.y<size.height-0.5)
        write(int(std::floor(v.pixel.x+0.5)),int(std::floor(v.pixel.y+0.5)),v.z,v.confidence,v.slope,v.roughness,v.slope_valid,v.roughness_valid);
    if(gradients) depthGradients(out,k);
    return out;
}
} // namespace metric_mapping::detail
#endif
