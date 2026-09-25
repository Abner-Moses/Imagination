#include "test_support.hpp"
#include "metric_mapping/geometry.hpp"
#include "metric_mapping/point_cloud.hpp"

namespace metric_mapping::tests {
namespace {
void testSyntheticPlanarDepth()
{
    const CameraIntrinsics camera{3, 2, 2.0, 2.0, 1.0, 0.5};
    const cv::Mat rgb(2, 3, CV_8UC3, cv::Scalar(5, 10, 15));
    const cv::Mat depth(2, 3, CV_32F, cv::Scalar(2.0F));
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());

    require(result.world_points.size() == 6,
            "Planar frame must emit every valid pixel");
    for (const ColoredPoint& point : result.world_points) {
        requireNear(point.z, 2.0, 1e-6,
                    "Synthetic constant-depth plane must remain at Z=2 m");
    }
}

void testPixelBackProjection()
{
    const CameraIntrinsics camera{10, 10, 2.0, 4.0, 1.0, 2.0};
    const cv::Vec3d point = backProjectPixel(3.0, 4.0, 2.0, camera);
    requireNear(point[0], 2.0, 1e-12, "Back-projected X");
    requireNear(point[1], 1.0, 1e-12, "Back-projected Y");
    requireNear(point[2], 2.0, 1e-12, "Back-projected Z");
}

void testWorldTransformAndConvention()
{
    cv::Matx44d camera_to_world = cv::Matx44d::eye();
    camera_to_world(0, 0) = 0.0;
    camera_to_world(0, 1) = -1.0;
    camera_to_world(1, 0) = 1.0;
    camera_to_world(1, 1) = 0.0;
    camera_to_world(0, 3) = 1.0;
    camera_to_world(1, 3) = 2.0;
    camera_to_world(2, 3) = 3.0;

    const cv::Vec3d world =
        transformPoint(camera_to_world, cv::Vec3d(1.0, 0.0, 0.0));
    requireNear(world[0], 1.0, 1e-12, "World transform X");
    requireNear(world[1], 3.0, 1e-12, "World transform Y");
    requireNear(world[2], 3.0, 1e-12, "World transform Z");

    const cv::Matx44d world_to_camera =
        invertRigidTransform(camera_to_world);
    const cv::Matx44d recovered = cameraToWorldTransform(
        world_to_camera, PoseConvention::WorldToCamera);
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            requireNear(recovered(row, column),
                        camera_to_world(row, column), 1e-12,
                        "T_CW inversion must recover T_WC");
        }
    }
}

void testMultipleFrameOverlap()
{
    const CameraIntrinsics camera{3, 1, 1.0, 1.0, 1.0, 0.0};
    const cv::Mat rgb(1, 3, CV_8UC3, cv::Scalar(30, 20, 10));
    cv::Mat depth_first(1, 3, CV_32F, cv::Scalar(0.0F));
    cv::Mat depth_second(1, 3, CV_32F, cv::Scalar(0.0F));
    depth_first.at<float>(0, 1) = 2.0F;
    depth_second.at<float>(0, 0) = 2.0F;

    const FrameResult first = backProjectFrame(
        rgb, depth_first, camera, metricDepthConfig(), cv::Matx44d::eye());
    cv::Matx44d second_pose = cv::Matx44d::eye();
    second_pose(0, 3) = 2.0;
    const FrameResult second = backProjectFrame(
        rgb, depth_second, camera, metricDepthConfig(), second_pose);

    require(first.world_points.size() == 1 &&
                second.world_points.size() == 1,
            "Each synthetic frame must emit one point");
    requireNear(first.world_points[0].x, second.world_points[0].x, 1e-6,
                "Two views must overlap in world X");
    requireNear(first.world_points[0].y, second.world_points[0].y, 1e-6,
                "Two views must overlap in world Y");
    requireNear(first.world_points[0].z, second.world_points[0].z, 1e-6,
                "Two views must overlap in world Z");

    VoxelGridAccumulator fusion(0.1, 10);
    fusion.add(first.world_points);
    fusion.add(second.world_points);
    require(fusion.points().size() == 1,
            "Overlapping world observations must fuse into one voxel");
}

void testRgbPreservation()
{
    const CameraIntrinsics camera{1, 1, 1.0, 1.0, 0.0, 0.0};
    const cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(30, 20, 10));
    const cv::Mat depth(1, 1, CV_32F, cv::Scalar(1.0F));
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());
    require(result.world_points[0].r == 10 &&
                result.world_points[0].g == 20 &&
                result.world_points[0].b == 30,
            "PLY RGB must preserve the corresponding BGR source pixel");
}

void testMetricScale()
{
    const CameraIntrinsics camera{2, 1, 1.0, 1.0, 0.0, 0.0};
    const cv::Mat rgb(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat depth(1, 2, CV_16U, cv::Scalar(1000));
    DepthConfig millimeter_depth = metricDepthConfig();
    millimeter_depth.unit = DepthUnit::Millimeters;
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, millimeter_depth, cv::Matx44d::eye());
    const ColoredPoint& first = result.world_points[0];
    const ColoredPoint& second = result.world_points[1];
    const double distance = std::sqrt(
        std::pow(second.x - first.x, 2) +
        std::pow(second.y - first.y, 2) +
        std::pow(second.z - first.z, 2));
    requireNear(distance, 1.0, 1e-6,
                "1000 mm synthetic object must measure 1 metre");
}

void testInvalidDepthRejection()
{
    const CameraIntrinsics camera{2, 2, 1.0, 1.0, 0.5, 0.5};
    const cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(2, 2, CV_32F);
    depth.at<float>(0, 0) = 0.0F;
    depth.at<float>(0, 1) = std::numeric_limits<float>::quiet_NaN();
    depth.at<float>(1, 0) = std::numeric_limits<float>::infinity();
    depth.at<float>(1, 1) = 1.0F;
    const FrameResult result = backProjectFrame(
        rgb, depth, camera, metricDepthConfig(), cv::Matx44d::eye());
    require(result.diagnostics.valid_depth_pixels == 1 &&
                result.diagnostics.invalid_depth_pixels == 3 &&
                result.world_points.size() == 1,
            "NaN, infinity, zero, and invalid depth must be rejected");
}

void testStrideBounds()
{
    const CameraIntrinsics camera{2, 2, 1, 1, 0.5, 0.5};
    const cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(0));
    const cv::Mat depth(2, 2, CV_32F, cv::Scalar(1));
    auto config = metricDepthConfig();
    for (int stride : {65536, std::numeric_limits<int>::max()}) {
        config.pixel_stride = stride;
        const auto result = backProjectFrame(rgb, depth, camera, config, cv::Matx44d::eye());
        require(result.world_points.size() == 1, "Large positive strides must emit the first pixel");
    }
    for (int stride : {0, -1}) {
        config.pixel_stride = stride;
        requireThrows([&] { backProjectFrame(rgb, depth, camera, config, cv::Matx44d::eye()); },
                      "Nonpositive strides must be rejected");
    }
}
} // namespace

std::vector<TestCase> rgbdTests()
{
    return {
        {"synthetic planar depth", testSyntheticPlanarDepth},
        {"pixel back-projection", testPixelBackProjection},
        {"world transform and T_CW inversion", testWorldTransformAndConvention},
        {"multiple-frame overlap", testMultipleFrameOverlap},
        {"RGB preservation", testRgbPreservation},
        {"metric scale", testMetricScale},
        {"invalid depth rejection", testInvalidDepthRejection},
        {"pixel stride bounds", testStrideBounds}
    };
}
} // namespace metric_mapping::tests
