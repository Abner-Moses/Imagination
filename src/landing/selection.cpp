#include "analysis_internal.hpp"
#include "fuzzy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

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
            const bool candidate = satisfies_hard_constraints &&
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
