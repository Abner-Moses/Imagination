#include "analysis_internal.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

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

void classifyFootprints(const cv::Mat& dem, const cv::Mat& valid,
                        const TerrainGrid& grid, const LandingAnalysisConfig& config,
                        LandingAnalysis& analysis)
{
    cv::Mat residual = dem - analysis.smoothed_dem_m;
    cv::Mat residual_squared;
    cv::multiply(residual, residual, residual_squared);
    cv::Mat valid_float;
    valid.convertTo(valid_float, CV_32F, 1.0 / 255.0);
    residual.setTo(0.0F, valid == 0);
    residual_squared.setTo(0.0F, valid == 0);

    int footprint_cells = static_cast<int>(std::ceil(
        config.uav_footprint_diagonal_m / grid.resolution_m));
    footprint_cells = std::max(3, footprint_cells);
    if (footprint_cells % 2 == 0)
        ++footprint_cells;
    const cv::Size footprint(footprint_cells, footprint_cells);
    cv::Mat sum;
    cv::Mat sum_squared;
    cv::Mat count;
    cv::boxFilter(residual, sum, CV_32F, footprint, cv::Point(-1, -1),
                  false, cv::BORDER_CONSTANT);
    cv::boxFilter(residual_squared, sum_squared, CV_32F, footprint,
                  cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
    cv::boxFilter(valid_float, count, CV_32F, footprint,
                  cv::Point(-1, -1), false, cv::BORDER_CONSTANT);

    analysis.roughness_m = cv::Mat(
        grid.height, grid.width, CV_32F,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
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
            if (samples > 1.0F) {
                const float residual_sum = sum.at<float>(row, column);
                const float numerator = std::max(
                    0.0F, sum_squared.at<float>(row, column) -
                              residual_sum * residual_sum / samples);
                analysis.roughness_m.at<float>(row, column) =
                    std::sqrt(numerator / (samples - 1.0F));
            }
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
