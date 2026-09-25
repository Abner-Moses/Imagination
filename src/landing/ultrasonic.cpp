#include "metric_mapping/ultrasonic.hpp"
#include "../terrain/grid_internal.hpp"
#include <cmath>
#include <stdexcept>

namespace metric_mapping {
void validateUltrasonicMeasurement(const UltrasonicMeasurement& measurement)
{
    if (!std::isfinite(measurement.distance_m) || measurement.distance_m <= 0.0 ||
        !std::isfinite(measurement.maximum_error_m) || measurement.maximum_error_m < 0.0)
        throw std::runtime_error("Ultrasonic distance must be positive and tolerance nonnegative (meters)");
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(measurement.sensor_offset_world_m[axis]))
            throw std::runtime_error("Ultrasonic sensor offset must be finite");
}

GroundRangeCheck checkGroundRange(const TerrainGrid& grid,
                                 const cv::Vec3d& uav_position_world_m,
                                 const UltrasonicMeasurement& measurement)
{
    detail::validateGrid(grid);
    validateUltrasonicMeasurement(measurement);
    GroundRangeCheck result{measurement, std::nullopt, false};
    const cv::Vec3d origin = uav_position_world_m + measurement.sensor_offset_world_m;
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(origin[axis]))
            throw std::runtime_error("Ultrasonic sensor position must be finite");

    // Sample the nearest grid location. Only measured cells can
    // corroborate a reading; an IDW-filled hole is not independent evidence.
    const double column = std::round((origin[0] - grid.minimum_x_m) / grid.resolution_m);
    const double row = std::round((grid.maximum_y_m - origin[1]) / grid.resolution_m);
    if (!std::isfinite(column) || !std::isfinite(row) ||
        column < 0 || column >= grid.width || row < 0 || row >= grid.height)
        return result;
    const auto cell = grid.index(static_cast<int>(row), static_cast<int>(column));
    if (grid.validity[cell] != 255 || !std::isfinite(grid.elevation_m[cell]))
        return result;
    const double distance = origin[2] - grid.elevation_m[cell];
    if (!std::isfinite(distance) || distance <= 0.0)
        return result;
    result.mapped_distance_m = distance;
    result.consistent = std::abs(distance - measurement.distance_m) <= measurement.maximum_error_m;
    return result;
}
} // namespace metric_mapping
