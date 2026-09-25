#pragma once

#include "metric_mapping/types.hpp"

#include <vector>

namespace metric_mapping {

TerrainGrid createTerrainGrid(const std::vector<ColoredPoint>& points,
                              double resolution_m,
                              std::size_t max_cells);
void interpolateIdw(TerrainGrid& grid, const IdwConfig& config);
GridStatistics computeGridStatistics(const TerrainGrid& grid);
std::vector<ColoredPoint> terrainGridPoints(const TerrainGrid& grid);
LandingAnalysis analyzeLandingSites(
    const TerrainGrid& grid,
    const cv::Vec3d& uav_position_world_m,
    const LandingAnalysisConfig& config = {});

}  // namespace metric_mapping
