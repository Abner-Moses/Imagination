#include "metric_mapping/terrain.hpp"
#include "metric_mapping/ultrasonic.hpp"
#include "analysis_internal.hpp"
#include "../terrain/grid_internal.hpp"
#include <cmath>
#include <stdexcept>

namespace metric_mapping {

LandingAnalysis analyzeLandingSites(const TerrainGrid& grid,
                                    const cv::Vec3d& uav_position_world_m,
                                    const LandingAnalysisConfig& config,
                                    const std::optional<UltrasonicMeasurement>& ultrasonic)
{
    detail::validateGrid(grid);
    if (config.diffusion_iterations < 0 ||
        !std::isfinite(config.diffusion_conductance_m) ||
        config.diffusion_conductance_m <= 0.0 ||
        !std::isfinite(config.diffusion_time_step) ||
        config.diffusion_time_step <= 0.0 ||
        config.diffusion_time_step > 0.25 ||
        !std::isfinite(config.uav_footprint_diagonal_m) ||
        config.uav_footprint_diagonal_m <= 0.0 ||
        !std::isfinite(config.maximum_slope_degrees) ||
        config.maximum_slope_degrees <= 0.0 ||
        !std::isfinite(config.maximum_roughness_m) ||
        config.maximum_roughness_m <= 0.0 ||
        !std::isfinite(config.minimum_hazard_distance_m) ||
        config.minimum_hazard_distance_m <= 0.0 ||
        !std::isfinite(config.minimum_global_safety_index) ||
        config.minimum_global_safety_index < 0.0 ||
        config.minimum_global_safety_index > 100.0 ||
        !std::isfinite(uav_position_world_m[0]) ||
        !std::isfinite(uav_position_world_m[1]) ||
        !std::isfinite(uav_position_world_m[2])) {
        throw std::runtime_error("Invalid landing-analysis configuration");
    }

    LandingAnalysis analysis;
    if (ultrasonic)
        analysis.ultrasonic = checkGroundRange(grid, uav_position_world_m, *ultrasonic);
    detail::measureTerrain(grid, config, analysis);
    detail::selectLandingSite(grid, uav_position_world_m, config, analysis);
    return analysis;
}

} // namespace metric_mapping
