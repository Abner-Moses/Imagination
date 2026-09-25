#pragma once
#include "metric_mapping/types.hpp"

namespace metric_mapping::detail {
// Build smoothed elevation, slope, roughness, hazards, and hazard clearance.
void measureTerrain(const TerrainGrid& grid, const LandingAnalysisConfig& config,
                    LandingAnalysis& analysis);
// Combine terrain safety and travel distance, then select a feasible site.
void selectLandingSite(const TerrainGrid& grid, const cv::Vec3d& uav_position,
                       const LandingAnalysisConfig& config, LandingAnalysis& analysis);
} // namespace metric_mapping::detail
