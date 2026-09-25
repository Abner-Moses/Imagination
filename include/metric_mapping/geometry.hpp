#pragma once

#include "metric_mapping/types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>

namespace metric_mapping {

void validateCamera(const CameraIntrinsics& camera);
void validateRigidTransform(const cv::Matx44d& transform,
                            const std::string& name);
cv::Matx44d invertRigidTransform(const cv::Matx44d& transform);
cv::Matx44d cameraToWorldTransform(const cv::Matx44d& supplied_pose,
                                   PoseConvention convention);
cv::Vec3d backProjectPixel(double u,
                           double v,
                           double depth_m,
                           const CameraIntrinsics& camera);
cv::Vec3d transformPoint(const cv::Matx44d& transform,
                         const cv::Vec3d& point);
FrameResult backProjectFrame(const cv::Mat& rgb,
                             const cv::Mat& depth,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world);
FrameResult reconstructFrame(const std::filesystem::path& rgb_path,
                             const std::filesystem::path& depth_path,
                             const CameraIntrinsics& camera,
                             const DepthConfig& depth_config,
                             const cv::Matx44d& camera_to_world);

}  // namespace metric_mapping
