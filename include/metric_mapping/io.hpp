#pragma once

#include "metric_mapping/types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace metric_mapping {

std::string escapeJson(const std::string& value);

void writePly(const std::filesystem::path& path,
              const std::vector<ColoredPoint>& points,
              const std::string& coordinate_comment =
                  "coordinates are metres in map_z_up world frame");
void writeDemCsv(const std::filesystem::path& path,
                 const TerrainGrid& grid);
void writeTerrainImages(const std::filesystem::path& output_directory,
                        const TerrainGrid& grid);
void writeLandingAnalysis(
    const std::filesystem::path& output_directory,
    const TerrainGrid& grid,
    const LandingAnalysis& analysis,
    const LandingAnalysisConfig& config,
    const cv::Vec3d& uav_position_world_m,
    bool demo_only);
void writeMetadata(const std::filesystem::path& path,
                   const ReconstructionConfig& config,
                   std::size_t raw_point_count,
                   const std::vector<ColoredPoint>& filtered_points,
                   const CloudBounds& bounds,
                   const TerrainGrid& grid,
                   const GridStatistics& grid_statistics,
                   const std::vector<cv::Vec3d>& camera_positions_world_m);

}  // namespace metric_mapping
