#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace metric_mapping::detail {
enum class FuzzyStage { Safety, Distance, Global };

// Cache only scores requested by this analysis. Scores are percentages.
class FuzzyScores {
public:
    float get(FuzzyStage stage, int first, int second);
private:
    std::array<std::vector<float>, 3> tables_;
    static float evaluate(FuzzyStage stage, float first_unit, float second_unit);
};
int quantize(double value, double maximum);
} // namespace metric_mapping::detail

// --- Analysis Internal ---

namespace metric_mapping::detail {
// Build smoothed elevation, slope, roughness, hazards, and hazard clearance.
void measureTerrain(const TerrainGrid& grid, const LandingAnalysisConfig& config,
                    LandingAnalysis& analysis);
// Combine terrain safety and travel distance, then select a feasible site.
void selectLandingSite(const TerrainGrid& grid, const cv::Vec3d& uav_position,
                       const LandingAnalysisConfig& config, LandingAnalysis& analysis);
} // namespace metric_mapping::detail

// --- Grid ---

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
        config.minimum_neighbors < 1 || config.maximum_neighbors < 0 ||
        (config.maximum_neighbors && config.maximum_neighbors < config.minimum_neighbors))
        throw std::runtime_error("Invalid IDW configuration");
    const int radius_cells = static_cast<int>(std::min(
        double(std::max(grid.width, grid.height)), std::ceil(
            std::min(config.search_radius_m,
                     config.maximum_interpolation_distance_m) / grid.resolution_m)));

    std::vector<std::pair<double, std::size_t>> nearby;
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
            nearby.clear();

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
                    if (config.maximum_neighbors) {
                        nearby.emplace_back(distance_m, neighbor);
                        continue;
                    }
                    weight_sum += weight;
                    elevation_sum += weight * grid.elevation_m[neighbor];
                    blue_sum += weight * grid.color_bgr[neighbor][0];
                    green_sum += weight * grid.color_bgr[neighbor][1];
                    red_sum += weight * grid.color_bgr[neighbor][2];
                    ++neighbor_count;
                }
            }

            if (config.maximum_neighbors) {
                const auto count = std::min(nearby.size(), std::size_t(config.maximum_neighbors));
                std::partial_sort(nearby.begin(), nearby.begin() + count, nearby.end());
                for (std::size_t i = 0; i < count; ++i) {
                    const auto index = nearby[i].second;
                    const double weight = 1.0 / std::pow(nearby[i].first, config.power);
                    weight_sum += weight; elevation_sum += weight * grid.elevation_m[index];
                    blue_sum += weight * grid.color_bgr[index][0];
                    green_sum += weight * grid.color_bgr[index][1];
                    red_sum += weight * grid.color_bgr[index][2];
                }
                neighbor_count = int(count);
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

// --- Ultrasonic ---

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

// --- Fuzzy ---

namespace metric_mapping::detail {
namespace {
constexpr int fuzzy_lut_size = 256;

float leftShoulder(float value, float full_until, float zero_at)
{
    if (value <= full_until)
        return 1.0F;
    if (value >= zero_at)
        return 0.0F;
    return (zero_at - value) / (zero_at - full_until);
}

float rightShoulder(float value, float zero_until, float full_at)
{
    if (value <= zero_until)
        return 0.0F;
    if (value >= full_at)
        return 1.0F;
    return (value - zero_until) / (full_at - zero_until);
}

float triangle(float value, float left, float peak, float right)
{
    if (value <= left || value >= right)
        return 0.0F;
    if (value == peak)
        return 1.0F;
    return value < peak ? (value - left) / (peak - left)
                        : (right - value) / (right - peak);
}

std::array<float, 3> terrainMembership(float ratio)
{
    // Fig. 16: good falls to zero at half the allowed value, normal peaks
    // there, and bad reaches full membership at the allowed limit.
    return {leftShoulder(ratio, 0.0F, 0.5F),
            triangle(ratio, 0.0F, 0.5F, 1.0F),
            rightShoulder(ratio, 0.0F, 1.0F)};
}

std::array<float, 5> safetyOutputMembership(float percent)
{
    // Fig. 17. Labels: unsafe, very risky, risky, little risky, safe.
    return {leftShoulder(percent, 5.0F, 10.0F),
            triangle(percent, 5.0F, 15.0F, 25.0F),
            triangle(percent, 15.0F, 30.0F, 45.0F),
            triangle(percent, 30.0F, 50.0F, 70.0F),
            rightShoulder(percent, 50.0F, 75.0F)};
}

std::array<float, 5> distanceOutputMembership(float percent)
{
    // Figs. 21 and 24. Labels: poor, bad, normal, good, very good.
    return {leftShoulder(percent, 20.0F, 40.0F),
            triangle(percent, 0.0F, 20.0F, 40.0F),
            triangle(percent, 20.0F, 40.0F, 70.0F),
            triangle(percent, 40.0F, 60.0F, 100.0F),
            triangle(percent, 60.0F, 80.0F, 100.0F)};
}

enum class OutputPartition {
    Safety,
    Distance
};

float outputMembership(OutputPartition partition, int label, int percent)
{
    struct Samples {
        std::array<std::array<float, 101>, 5> safety{};
        std::array<std::array<float, 101>, 5> distance{};
    };
    static const Samples samples = [] {
        Samples result;
        for (int value = 0; value <= 100; ++value) {
            const auto safety =
                safetyOutputMembership(static_cast<float>(value));
            const auto distance =
                distanceOutputMembership(static_cast<float>(value));
            for (int output = 0; output < 5; ++output) {
                result.safety[static_cast<std::size_t>(output)]
                             [static_cast<std::size_t>(value)] =
                    safety[static_cast<std::size_t>(output)];
                result.distance[static_cast<std::size_t>(output)]
                               [static_cast<std::size_t>(value)] =
                    distance[static_cast<std::size_t>(output)];
            }
        }
        return result;
    }();
    const auto& selected = partition == OutputPartition::Safety
                               ? samples.safety
                               : samples.distance;
    return selected[static_cast<std::size_t>(label)]
                   [static_cast<std::size_t>(percent)];
}

template <std::size_t Rows, std::size_t Columns>
float fuzzyCentroid(
    const std::array<float, Rows>& rows,
    const std::array<float, Columns>& columns,
    const std::array<std::array<int, Columns>, Rows>& rules,
    OutputPartition output_partition)
{
    std::array<float, 5> activation{};
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Columns; ++column) {
            const float firing = std::min(rows[row], columns[column]);
            const int label = rules[row][column];
            activation[static_cast<std::size_t>(label)] = std::max(
                activation[static_cast<std::size_t>(label)], firing);
        }
    }

    double weighted_sum = 0.0;
    double membership_sum = 0.0;
    for (int percent = 0; percent <= 100; ++percent) {
        float aggregated = 0.0F;
        for (int label = 0; label < 5; ++label) {
            aggregated = std::max(
                aggregated,
                std::min(activation[static_cast<std::size_t>(label)],
                         outputMembership(output_partition, label,
                                          percent)));
        }
        weighted_sum += percent * aggregated;
        membership_sum += aggregated;
    }
    return membership_sum > 0.0
               ? static_cast<float>(weighted_sum / membership_sum)
               : 0.0F;
}

} // namespace

float FuzzyScores::get(FuzzyStage stage, int first, int second)
{
    auto& table = tables_[static_cast<std::size_t>(stage)];
    if (table.empty())
        table.assign(fuzzy_lut_size * fuzzy_lut_size, -1.0F);
    float& score = table[static_cast<std::size_t>(first) * fuzzy_lut_size + second];
    if (score < 0)
        score = evaluate(stage, float(first) / (fuzzy_lut_size - 1),
                         float(second) / (fuzzy_lut_size - 1));
    return score;
}

float FuzzyScores::evaluate(FuzzyStage stage, float first_unit, float second_unit)
{
    // Table IV. Rows: good/normal/bad roughness. Columns:
    // good/normal/bad slope. Output labels run unsafe..safe (0..4).
    constexpr std::array<std::array<int, 3>, 3> safety_rules{{
        {{4, 2, 0}},
        {{3, 1, 0}},
        {{0, 0, 0}},
    }};
    // Table V, reordered to rows bad/normal/good nearest-hazard distance
    // and columns near/normal/far spatial distance. Outputs are poor..very
    // good (0..4).
    constexpr std::array<std::array<int, 3>, 3> distance_rules{{
        {{1, 0, 0}},
        {{2, 2, 1}},
        {{3, 4, 3}},
    }};
    // Table VI, reordered to rows poor..very-good distance index and columns
    // unsafe..safe terrain index. Outputs are poor..very-good (0..4).
    constexpr std::array<std::array<int, 5>, 5> global_rules{{
        {{0, 0, 0, 0, 1}},
        {{0, 0, 1, 1, 1}},
        {{0, 1, 1, 2, 3}},
        {{0, 1, 2, 2, 3}},
        {{0, 1, 2, 3, 4}},
    }};

    if (stage == FuzzyStage::Safety)
        return fuzzyCentroid(terrainMembership(2.0F * first_unit),
                             terrainMembership(2.0F * second_unit),
                             safety_rules, OutputPartition::Safety);
    if (stage == FuzzyStage::Global)
        return fuzzyCentroid(distanceOutputMembership(100.0F * first_unit),
                             safetyOutputMembership(100.0F * second_unit),
                             global_rules, OutputPartition::Distance);
    const float hazard_ratio = 4.0F * first_unit;
    const std::array<float, 3> hazard_distance{
        leftShoulder(hazard_ratio, 0.0F, 2.5F),
        triangle(hazard_ratio, 1.0F, 2.5F, 4.0F),
        rightShoulder(hazard_ratio, 2.5F, 4.0F)};
    const std::array<float, 3> spatial_distance{
        leftShoulder(second_unit, 0.0F, 0.5F),
        triangle(second_unit, 0.0F, 0.5F, 1.0F),
        rightShoulder(second_unit, 0.5F, 1.0F)};
    return fuzzyCentroid(hazard_distance, spatial_distance, distance_rules,
                         OutputPartition::Distance);
}

int quantize(double value, double maximum)
{
    const double normalized = std::clamp(value / maximum, 0.0, 1.0);
    return static_cast<int>(std::lround(
        normalized * (fuzzy_lut_size - 1)));
}

} // namespace metric_mapping::detail

// --- Surface ---

namespace metric_mapping::detail {
namespace {
constexpr std::array<int, 4> row_offset{-1, 1, 0, 0};
constexpr std::array<int, 4> column_offset{0, 0, -1, 1};

cv::Mat smoothElevation(const cv::Mat& dem, const cv::Mat& valid,
                        const TerrainGrid& grid, const LandingAnalysisConfig& config)
{
    cv::Mat smoothed = dem.clone();
    cv::Mat next = smoothed.clone();
    const float conductance =
        static_cast<float>(config.diffusion_conductance_m);
    const float time_step =
        static_cast<float>(config.diffusion_time_step);
    for (int iteration = 0; iteration < config.diffusion_iterations;
         ++iteration) {
        smoothed.copyTo(next);
        for (int row = 0; row < grid.height; ++row) {
            for (int column = 0; column < grid.width; ++column) {
                if (valid.at<std::uint8_t>(row, column) == 0)
                    continue;
                const float center =
                    smoothed.at<float>(row, column);
                float update = 0.0F;
                for (std::size_t direction = 0;
                     direction < row_offset.size(); ++direction) {
                    const int neighbor_row = row + row_offset[direction];
                    const int neighbor_column =
                        column + column_offset[direction];
                    if (neighbor_row < 0 || neighbor_row >= grid.height ||
                        neighbor_column < 0 ||
                        neighbor_column >= grid.width ||
                        valid.at<std::uint8_t>(neighbor_row,
                                               neighbor_column) == 0) {
                        continue;
                    }
                    const float delta =
                        smoothed.at<float>(
                            neighbor_row, neighbor_column) - center;
                    const float ratio = delta / conductance;
                    update += std::exp(-(ratio * ratio)) * delta;
                }
                next.at<float>(row, column) = center + time_step * update;
            }
        }
        std::swap(smoothed, next);
    }

    return smoothed;
}

cv::Mat estimateSlope(const cv::Mat& dem, const cv::Mat& smoothed,
                      const cv::Mat& valid, const TerrainGrid& grid)
{
    cv::Mat gradient_x;
    cv::Mat gradient_y;
    const double sobel_scale = 1.0 / (8.0 * grid.resolution_m);
    cv::Sobel(smoothed, gradient_x, CV_32F, 1, 0, 3,
              sobel_scale, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(smoothed, gradient_y, CV_32F, 0, 1, 3,
              sobel_scale, 0.0, cv::BORDER_REPLICATE);
    cv::Mat slope = cv::Mat(
        grid.height, grid.width, CV_32F,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (valid.at<std::uint8_t>(row, column) == 0)
                continue;
            double gradient = std::hypot(gradient_x.at<float>(row, column),
                                         gradient_y.at<float>(row, column));
            // Centered derivatives cancel on alternating ridges, and diffusion
            // preserves sharp edges. Keep the measured neighbor gradients too.
            for (std::size_t direction = 0; direction < row_offset.size(); ++direction) {
                const int neighbor_row = row + row_offset[direction];
                const int neighbor_column = column + column_offset[direction];
                if (neighbor_row >= 0 && neighbor_row < grid.height && neighbor_column >= 0 && neighbor_column < grid.width &&
                    valid.at<std::uint8_t>(neighbor_row, neighbor_column))
                    gradient = std::max(gradient,
                        std::abs(double(dem.at<float>(neighbor_row, neighbor_column)) -
                                 dem.at<float>(row, column)) / grid.resolution_m);
            }
            slope.at<float>(row, column) =
                static_cast<float>(std::atan(gradient) * 180.0 / CV_PI);
        }
    }

    return slope;
}

// Shared residual statistics used by both analytical features and legacy landing.
cv::Mat residualDeviation(const cv::Mat& dem, const cv::Mat& smoothed,
                          const cv::Mat& valid, int window, cv::Mat& count)
{
    cv::Mat residual = dem - smoothed, squared, weights, sum, sum_squared;
    residual.setTo(0, valid == 0);
    cv::multiply(residual, residual, squared);
    valid.convertTo(weights, CV_32F, 1.0 / 255);
    const cv::Size size(window, window);
    cv::boxFilter(residual, sum, CV_32F, size, {-1,-1}, false, cv::BORDER_CONSTANT);
    cv::boxFilter(squared, sum_squared, CV_32F, size, {-1,-1}, false, cv::BORDER_CONSTANT);
    cv::boxFilter(weights, count, CV_32F, size, {-1,-1}, false, cv::BORDER_CONSTANT);
    cv::Mat result(dem.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    for (int y = 0; y < dem.rows; ++y) for (int x = 0; x < dem.cols; ++x) {
        const float n = count.at<float>(y,x), total = sum.at<float>(y,x);
        if (valid.at<std::uint8_t>(y,x) && n > 1)
            result.at<float>(y,x) = std::sqrt(std::max(0.0F,
                sum_squared.at<float>(y,x) - total * total / n) / (n - 1));
    }
    return result;
}

void classifyFootprints(const cv::Mat& dem, const cv::Mat& valid,
                        const TerrainGrid& grid, const LandingAnalysisConfig& config,
                        LandingAnalysis& analysis)
{
    int footprint_cells = std::max(3, int(std::ceil(config.uav_footprint_diagonal_m / grid.resolution_m)));
    if (footprint_cells % 2 == 0) ++footprint_cells;
    cv::Mat count;
    analysis.roughness_m = residualDeviation(dem, analysis.smoothed_dem_m, valid, footprint_cells, count);
    analysis.hazard_mask = cv::Mat(grid.height, grid.width, CV_8U,
                                   cv::Scalar(255));
    cv::Mat safe_mask(grid.height, grid.width, CV_8U, cv::Scalar(0));
    const float full_footprint_count =
        static_cast<float>(footprint_cells * footprint_cells);
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (valid.at<std::uint8_t>(row, column) == 0) {
                ++analysis.hazard_cells;
                continue;
            }
            const float samples = count.at<float>(row, column);
            const float slope =
                analysis.slope_degrees.at<float>(row, column);
            const float roughness =
                analysis.roughness_m.at<float>(row, column);
            const bool footprint_is_valid =
                samples >= full_footprint_count - 0.5F;
            const bool is_hazard =
                !footprint_is_valid || !std::isfinite(slope) ||
                !std::isfinite(roughness) ||
                slope > config.maximum_slope_degrees ||
                roughness > config.maximum_roughness_m;
            if (is_hazard) {
                ++analysis.hazard_cells;
            } else {
                analysis.hazard_mask.at<std::uint8_t>(row, column) = 0;
                safe_mask.at<std::uint8_t>(row, column) = 255;
                ++analysis.admissible_cells;
            }
        }
    }

    cv::distanceTransform(safe_mask,
                          analysis.nearest_hazard_distance_m,
                          2, cv::DIST_MASK_PRECISE);  // Euclidean distance.
    analysis.nearest_hazard_distance_m *= grid.resolution_m;
}
} // namespace

void measureTerrain(const TerrainGrid& grid, const LandingAnalysisConfig& config,
                    LandingAnalysis& analysis)
{
    cv::Mat dem(grid.height, grid.width, CV_32F, cv::Scalar(0));
    cv::Mat valid(grid.height, grid.width, CV_8U, cv::Scalar(0));
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            if (grid.validity[cell] != 0 &&
                std::isfinite(grid.elevation_m[cell])) {
                dem.at<float>(row, column) = grid.elevation_m[cell];
                valid.at<std::uint8_t>(row, column) = 255;
            }
        }
    }

    analysis.smoothed_dem_m = smoothElevation(dem, valid, grid, config);
    analysis.slope_degrees = estimateSlope(dem, analysis.smoothed_dem_m, valid, grid);
    classifyFootprints(dem, valid, grid, config, analysis);
}
} // namespace metric_mapping::detail

// --- Selection ---

namespace metric_mapping::detail {

void selectLandingSite(const TerrainGrid& grid, const cv::Vec3d& uav_position_world_m,
                       const LandingAnalysisConfig& config, LandingAnalysis& analysis)
{
    analysis.spatial_distance_m = cv::Mat(
        grid.height, grid.width, CV_32F,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    analysis.safety_index = cv::Mat(grid.height, grid.width, CV_32F,
                                    cv::Scalar(0));
    analysis.distance_index = cv::Mat(grid.height, grid.width, CV_32F,
                                      cv::Scalar(0));
    analysis.global_safety_index = cv::Mat(grid.height, grid.width, CV_32F,
                                           cv::Scalar(0));

    double minimum_spatial_distance =
        std::numeric_limits<double>::infinity();
    double maximum_spatial_distance = 0.0;
    double spatial_distance_sum = 0.0;
    std::size_t spatial_distance_count = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            if (grid.validity[cell] == 0 ||
                !std::isfinite(grid.elevation_m[cell])) {
                continue;
            }
            const cv::Vec3d point(
                grid.minimum_x_m + column * grid.resolution_m,
                grid.maximum_y_m - row * grid.resolution_m,
                grid.elevation_m[cell]);
            const double distance = cv::norm(uav_position_world_m - point);
            analysis.spatial_distance_m.at<float>(row, column) =
                static_cast<float>(distance);
            minimum_spatial_distance =
                std::min(minimum_spatial_distance, distance);
            maximum_spatial_distance =
                std::max(maximum_spatial_distance, distance);
            spatial_distance_sum += distance;
            ++spatial_distance_count;
        }
    }
    if (spatial_distance_count == 0)
        return;
    const double average_spatial_distance =
        spatial_distance_sum / spatial_distance_count;

    FuzzyScores fuzzy;
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            if (grid.validity[cell] == 0)
                continue;
            const double slope =
                analysis.slope_degrees.at<float>(row, column);
            const double roughness =
                analysis.roughness_m.at<float>(row, column);
            if (!std::isfinite(slope) || !std::isfinite(roughness))
                continue;

            const int roughness_index = quantize(
                roughness / config.maximum_roughness_m, 2.0);
            const int slope_index = quantize(
                slope / config.maximum_slope_degrees, 2.0);
            const float safety = fuzzy.get(FuzzyStage::Safety, roughness_index,
                                     slope_index);
            analysis.safety_index.at<float>(row, column) = safety;

            const double spatial =
                analysis.spatial_distance_m.at<float>(row, column);
            double spatial_unit = 0.5;
            if (spatial <= average_spatial_distance) {
                const double span = std::max(
                    average_spatial_distance - minimum_spatial_distance,
                    1e-9);
                spatial_unit = 0.5 *
                    (spatial - minimum_spatial_distance) / span;
            } else {
                const double span = std::max(
                    maximum_spatial_distance - average_spatial_distance,
                    1e-9);
                spatial_unit = 0.5 + 0.5 *
                    (spatial - average_spatial_distance) / span;
            }
            const double hazard_distance =
                analysis.nearest_hazard_distance_m.at<float>(row, column);
            const int hazard_index = quantize(
                hazard_distance /
                    config.minimum_hazard_distance_m,
                4.0);
            const int spatial_index = quantize(spatial_unit, 1.0);
            const float distance = fuzzy.get(FuzzyStage::Distance, hazard_index,
                                       spatial_index);
            analysis.distance_index.at<float>(row, column) = distance;

            const int distance_index = quantize(distance, 100.0);
            const int safety_index = quantize(safety, 100.0);
            const float global = fuzzy.get(FuzzyStage::Global, distance_index,
                                     safety_index);
            analysis.global_safety_index.at<float>(row, column) = global;

            const bool satisfies_hard_constraints =
                analysis.hazard_mask.at<std::uint8_t>(row, column) == 0 &&
                slope <= config.maximum_slope_degrees &&
                roughness <= config.maximum_roughness_m &&
                hazard_distance >= config.minimum_hazard_distance_m;
            if (satisfies_hard_constraints) {
                ++analysis.clearance_cells;
                analysis.maximum_global_safety_index = std::max(
                    analysis.maximum_global_safety_index,
                    static_cast<double>(global));
            }
            const bool range_consistent = !analysis.ultrasonic || analysis.ultrasonic->consistent;
            const bool candidate = range_consistent && satisfies_hard_constraints &&
                global >= config.minimum_global_safety_index;
            if (!candidate)
                continue;
            ++analysis.candidate_cells;

            const LandingSite& best = analysis.best_site;
            const bool better = !best.found ||
                global > best.global_safety_index + 1e-6 ||
                (std::abs(global - best.global_safety_index) <= 1e-6 &&
                 (safety > best.safety_index + 1e-6 ||
                  (std::abs(safety - best.safety_index) <= 1e-6 &&
                   distance > best.distance_index)));
            if (!better)
                continue;

            analysis.best_site.found = true;
            analysis.best_site.row = row;
            analysis.best_site.column = column;
            analysis.best_site.world_m = cv::Vec3d(
                grid.minimum_x_m + column * grid.resolution_m,
                grid.maximum_y_m - row * grid.resolution_m,
                grid.elevation_m[cell]);
            analysis.best_site.slope_degrees = slope;
            analysis.best_site.roughness_m = roughness;
            analysis.best_site.nearest_hazard_distance_m = hazard_distance;
            analysis.best_site.spatial_distance_m = spatial;
            analysis.best_site.safety_index = safety;
            analysis.best_site.distance_index = distance;
            analysis.best_site.global_safety_index = global;
        }
    }

}
} // namespace metric_mapping::detail

// --- Analysis ---

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

namespace metric_mapping {
SurfaceFeatures measureSurface(const TerrainGrid& grid, const SurfaceSettings& config)
{
    detail::validateGrid(grid);
    if (config.diffusion_iterations < 0 || config.diffusion_iterations > 100 ||
        !std::isfinite(config.diffusion_time_step) || config.diffusion_time_step <= 0 || config.diffusion_time_step > 0.25 ||
        !std::isfinite(config.diffusion_conductance_m) || config.diffusion_conductance_m <= 0 ||
        config.roughness_window < 3 || config.roughness_window > 31 || config.roughness_window % 2 == 0)
        throw std::runtime_error("Invalid analytical surface settings");
    SurfaceFeatures out;
    out.elevation_m = cv::Mat(grid.height, grid.width, CV_32F, cv::Scalar(0));
    out.valid = cv::Mat(grid.height, grid.width, CV_8U, cv::Scalar(0));
    for (int y=0; y<grid.height; ++y) for (int x=0; x<grid.width; ++x) {
        const auto i=grid.index(y,x);
        if (grid.validity[i] && std::isfinite(grid.elevation_m[i])) {
            out.elevation_m.at<float>(y,x)=grid.elevation_m[i]; out.valid.at<unsigned char>(y,x)=255;
        }
    }
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); };
    if (config.gradients || config.slope) {
        const auto start=Clock::now();
        cv::Sobel(out.elevation_m,out.gradient_x,CV_32F,1,0,3,1/(8*grid.resolution_m));
        // Terrain rows point toward world -Y.
        cv::Sobel(out.elevation_m,out.gradient_y,CV_32F,0,1,3,-1/(8*grid.resolution_m));
        cv::erode(out.valid,out.gradient_valid,cv::Mat::ones(3,3,CV_8U),{-1,-1},1,cv::BORDER_CONSTANT,0);
        out.gradient_x.setTo(0,out.gradient_valid==0); out.gradient_y.setTo(0,out.gradient_valid==0);
        out.gradients_ms=elapsed(start);
        if (config.slope) {
            const auto t=Clock::now();
            cv::magnitude(out.gradient_x,out.gradient_y,out.slope_rad);
            for (int y=0;y<grid.height;++y) for(int x=0;x<grid.width;++x)
                out.slope_rad.at<float>(y,x)=std::atan(out.slope_rad.at<float>(y,x));
            out.slope_ms=elapsed(t);
        }
    }
    if (config.roughness) {
        LandingAnalysisConfig diffusion;
        diffusion.diffusion_iterations=config.diffusion_iterations;
        diffusion.diffusion_conductance_m=config.diffusion_conductance_m;
        diffusion.diffusion_time_step=config.diffusion_time_step;
        auto t=Clock::now();
        out.smoothed_m=detail::smoothElevation(out.elevation_m,out.valid,grid,diffusion);
        out.diffusion_ms=elapsed(t); t=Clock::now();
        cv::Mat count;
        out.roughness_m=detail::residualDeviation(out.elevation_m,out.smoothed_m,out.valid,config.roughness_window,count);
        // Require a full neighborhood in the analytical bank; no unsupported extrapolation.
        cv::compare(count,config.roughness_window*config.roughness_window-0.5,out.roughness_valid,cv::CMP_GE);
        cv::bitwise_and(out.roughness_valid,out.valid,out.roughness_valid);
        out.roughness_m.setTo(0,out.roughness_valid==0);
        out.roughness_ms=elapsed(t);
    }
    return out;
}
} // namespace metric_mapping
