#pragma once

#include "metric_mapping/types.hpp"

#include <filesystem>
#include <vector>

namespace metric_mapping {

cv::Mat fitImage(const cv::Mat& image, cv::Size box);

void writeDebugVisualization(
    const std::filesystem::path& path,
    const std::vector<ColoredPoint>& points,
    const CloudBounds& bounds,
    const TerrainGrid& grid,
    const std::vector<cv::Vec3d>& camera_positions_world_m,
    const std::optional<cv::Vec3d>& uav_position_world_m);

}  // namespace metric_mapping
