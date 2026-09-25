#pragma once
#include "metric_mapping/two_view.hpp"

namespace metric_mapping::detail {
void writeTwoViewMetadata(const std::filesystem::path& path,
                          const CameraIntrinsics& camera, double baseline_m,
                          const TwoViewResult& result, const TerrainGrid& grid,
                          const GridStatistics& statistics, std::size_t completed_point_count);
void writeDemoMetadata(const std::filesystem::path& output_directory,
                       const std::filesystem::path& image_1_path,
                       const std::filesystem::path& image_2_path,
                       std::size_t alignment_matches, std::size_t raw_points,
                       std::size_t completed_points, const GridStatistics& statistics);
} // namespace metric_mapping::detail
