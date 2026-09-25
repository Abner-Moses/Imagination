#pragma once
#include "metric_mapping/types.hpp"

namespace metric_mapping {
void validateUltrasonicMeasurement(const UltrasonicMeasurement& measurement);
GroundRangeCheck checkGroundRange(const TerrainGrid& grid,
                                 const cv::Vec3d& uav_position_world_m,
                                 const UltrasonicMeasurement& measurement);
} // namespace metric_mapping
