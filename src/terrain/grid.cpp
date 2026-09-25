#include "metric_mapping/terrain.hpp"
#include "metric_mapping/point_cloud.hpp"
#include "grid_internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metric_mapping {
namespace {
double gridCoordinate(double value, double origin, double resolution)
{
    const double coordinate = std::abs(value - origin) / resolution;
    // Recover lattice boundaries rounded when XYZ was stored as float. Limit
    // snapping to 0.001 cell so ordinary points keep floor-based binning.
    const double tolerance = std::min(0.001,
        2.0 * std::numeric_limits<float>::epsilon() *
        std::max({std::abs(value), std::abs(origin), resolution}) / resolution);
    const double nearest = std::round(coordinate);
    return std::floor(std::abs(coordinate - nearest) <= tolerance
                          ? nearest : coordinate);
}

} // namespace

namespace detail {
void validateGrid(const TerrainGrid& grid)
{
    const std::size_t expected =
        static_cast<std::size_t>(grid.width) * grid.height;
    if (grid.width <= 0 || grid.height <= 0 ||
        !std::isfinite(grid.resolution_m) || grid.resolution_m <= 0.0 ||
        grid.elevation_m.size() != expected ||
        grid.validity.size() != expected ||
        grid.color_bgr.size() != expected) {
        throw std::runtime_error("Terrain grid is internally inconsistent");
    }
}

} // namespace detail

TerrainGrid createTerrainGrid(const std::vector<ColoredPoint>& points,
                              double resolution_m,
                              std::size_t max_cells)
{
    if (points.empty()) {
        throw std::runtime_error("Cannot create a terrain grid from no points");
    }
    if (!std::isfinite(resolution_m) || resolution_m <= 0.0 ||
        max_cells == 0) {
        throw std::runtime_error(
            "Map resolution and maximum cell count must be positive");
    }

    const CloudBounds bounds = computeBounds(points);
    const double columns = gridCoordinate(
        bounds.maximum[0], bounds.minimum[0], resolution_m) + 1.0;
    const double rows = gridCoordinate(
        bounds.minimum[1], bounds.maximum[1], resolution_m) + 1.0;
    if (!std::isfinite(columns) || !std::isfinite(rows) ||
        columns > std::numeric_limits<int>::max() ||
        rows > std::numeric_limits<int>::max())
        throw std::runtime_error("Terrain grid dimensions exceed integer limits");
    const std::size_t width = static_cast<std::size_t>(columns);
    const std::size_t height = static_cast<std::size_t>(rows);

    if (width == 0 || height == 0 || width > max_cells ||
        height > max_cells || width > max_cells / height ||
        width * height > max_cells ||
        width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Terrain grid exceeds map.max_cells; check metric pose scale, "
            "cloud bounds, and map resolution");
    }

    TerrainGrid grid;
    grid.width = static_cast<int>(width);
    grid.height = static_cast<int>(height);
    grid.resolution_m = resolution_m;
    grid.minimum_x_m = bounds.minimum[0];
    grid.maximum_y_m = bounds.maximum[1];
    const std::size_t cell_count = width * height;
    grid.elevation_m.assign(
        cell_count, std::numeric_limits<float>::quiet_NaN());
    grid.color_bgr.assign(cell_count, cv::Vec3b(0, 0, 0));
    grid.validity.assign(cell_count, 0);

    // Implementation choice (not specified by Iratni & Diaf): when several
    // points fall in one XY cell, retain the highest +Z surface point and its
    // observed color. This produces a vertically viewed surface grid.
    for (const ColoredPoint& point : points) {
        int column = static_cast<int>(gridCoordinate(
            point.x, grid.minimum_x_m, grid.resolution_m));
        int row = static_cast<int>(gridCoordinate(
            point.y, grid.maximum_y_m, grid.resolution_m));
        column = std::clamp(column, 0, grid.width - 1);
        row = std::clamp(row, 0, grid.height - 1);
        const std::size_t cell = grid.index(row, column);

        if (grid.validity[cell] == 0 || point.z > grid.elevation_m[cell]) {
            grid.elevation_m[cell] = point.z;
            grid.color_bgr[cell] = cv::Vec3b(point.b, point.g, point.r);
            grid.validity[cell] = 255;
        }
    }

    return grid;
}

void interpolateIdw(TerrainGrid& grid, const IdwConfig& config)
{
    if (!config.enabled)
        return;
    detail::validateGrid(grid);
    if (!std::isfinite(config.search_radius_m) || config.search_radius_m <= 0 ||
        !std::isfinite(config.maximum_interpolation_distance_m) ||
        config.maximum_interpolation_distance_m <= 0 ||
        !std::isfinite(config.power) || config.power <= 0 ||
        config.minimum_neighbors < 1)
        throw std::runtime_error("Invalid IDW configuration");
    const int radius_cells = static_cast<int>(std::min(
        double(std::max(grid.width, grid.height)), std::ceil(
            std::min(config.search_radius_m,
                     config.maximum_interpolation_distance_m) / grid.resolution_m)));

    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t target = grid.index(row, column);
            if (grid.validity[target] != 0)
                continue;

            double weight_sum = 0.0;
            double elevation_sum = 0.0;
            double blue_sum = 0.0;
            double green_sum = 0.0;
            double red_sum = 0.0;
            int neighbor_count = 0;

            const int minimum_row = std::max(0, row - radius_cells);
            const int maximum_row =
                std::min(grid.height - 1, row + radius_cells);
            const int minimum_column = std::max(0, column - radius_cells);
            const int maximum_column =
                std::min(grid.width - 1, column + radius_cells);

            for (int neighbor_row = minimum_row;
                 neighbor_row <= maximum_row;
                 ++neighbor_row) {
                for (int neighbor_column = minimum_column;
                     neighbor_column <= maximum_column;
                     ++neighbor_column) {
                    const std::size_t neighbor =
                        grid.index(neighbor_row, neighbor_column);
                    // Measured cells never change; new IDW cells cannot
                    // participate, so source copies are unnecessary.
                    if (grid.validity[neighbor] != 255)
                        continue;

                    const double delta_row = neighbor_row - row;
                    const double delta_column = neighbor_column - column;
                    const double distance_m = grid.resolution_m *
                        std::sqrt(delta_row * delta_row +
                                  delta_column * delta_column);
                    if (distance_m <= 0.0 ||
                        distance_m > config.search_radius_m ||
                        distance_m >
                            config.maximum_interpolation_distance_m) {
                        continue;
                    }

                    const double weight =
                        1.0 / std::pow(distance_m, config.power);
                    weight_sum += weight;
                    elevation_sum += weight * grid.elevation_m[neighbor];
                    blue_sum += weight * grid.color_bgr[neighbor][0];
                    green_sum += weight * grid.color_bgr[neighbor][1];
                    red_sum += weight * grid.color_bgr[neighbor][2];
                    ++neighbor_count;
                }
            }

            if (neighbor_count >= config.minimum_neighbors &&
                weight_sum > 0.0) {
                grid.elevation_m[target] = static_cast<float>(
                    elevation_sum / weight_sum);
                grid.color_bgr[target] = cv::Vec3b(
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(blue_sum / weight_sum), 0L, 255L)),
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(green_sum / weight_sum), 0L, 255L)),
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(red_sum / weight_sum), 0L, 255L)));
                grid.validity[target] = 127;
            }
        }
    }
}

GridStatistics computeGridStatistics(const TerrainGrid& grid)
{
    GridStatistics statistics;
    for (std::uint8_t validity : grid.validity) {
        if (validity == 255) {
            ++statistics.measured_cells;
        } else if (validity == 127) {
            ++statistics.interpolated_cells;
        } else {
            ++statistics.unknown_cells;
        }
    }
    return statistics;
}

std::vector<ColoredPoint> terrainGridPoints(const TerrainGrid& grid)
{
    detail::validateGrid(grid);

    std::vector<ColoredPoint> points;
    const GridStatistics statistics = computeGridStatistics(grid);
    points.reserve(statistics.measured_cells +
                   statistics.interpolated_cells);
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            if (grid.validity[cell] == 0 ||
                !std::isfinite(grid.elevation_m[cell])) {
                continue;
            }
            const cv::Vec3b bgr = grid.color_bgr[cell];
            points.push_back({
                static_cast<float>(grid.minimum_x_m +
                                   column * grid.resolution_m),
                static_cast<float>(grid.maximum_y_m -
                                   row * grid.resolution_m),
                grid.elevation_m[cell], bgr[2], bgr[1], bgr[0]});
        }
    }
    return points;
}

} // namespace metric_mapping
