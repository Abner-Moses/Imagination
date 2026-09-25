#include "metric_mapping/io.hpp"
#include "output_internal.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace metric_mapping {
using namespace detail;

void writeLandingAnalysis(
    const std::filesystem::path& output_directory,
    const TerrainGrid& grid,
    const LandingAnalysis& analysis,
    const LandingAnalysisConfig& config,
    const cv::Vec3d& uav_position_world_m,
    bool demo_only)
{
    std::filesystem::create_directories(output_directory);
    const cv::Size expected(grid.width, grid.height);
    const std::vector<const cv::Mat*> maps{
        &analysis.smoothed_dem_m,
        &analysis.slope_degrees,
        &analysis.roughness_m,
        &analysis.hazard_mask,
        &analysis.nearest_hazard_distance_m,
        &analysis.spatial_distance_m,
        &analysis.safety_index,
        &analysis.distance_index,
        &analysis.global_safety_index};
    for (const cv::Mat* map : maps) {
        if (map->size() != expected)
            throw std::runtime_error(
                "Landing-analysis map dimensions do not match the terrain");
    }

    const cv::Mat valid = terrainValidMask(grid);
    const cv::Mat smoothed_dem = colorizeDem(
        analysis.smoothed_dem_m, valid);
    const cv::Mat slope = colorizeFloat(
        analysis.slope_degrees, valid, 0.0,
        2.0 * config.maximum_slope_degrees);
    const cv::Mat roughness = colorizeFloat(
        analysis.roughness_m, valid, 0.0,
        2.0 * config.maximum_roughness_m);
    const cv::Mat hazard_distance = colorizeFloat(
        analysis.nearest_hazard_distance_m, valid, 0.0,
        4.0 * config.minimum_hazard_distance_m);
    const cv::Mat safety = colorizeFloat(
        analysis.safety_index, valid, 0.0, 100.0);
    const cv::Mat distance = colorizeFloat(
        analysis.distance_index, valid, 0.0, 100.0);
    const cv::Mat global = colorizeFloat(
        analysis.global_safety_index, valid, 0.0, 100.0);

    writeImage(output_directory / "smoothed_dem.png", smoothed_dem);
    writeImage(output_directory / "slope.png", slope);
    writeImage(output_directory / "roughness.png", roughness);
    writeImage(output_directory / "hazard_mask.png",
               analysis.hazard_mask);
    writeImage(output_directory / "nearest_hazard_distance.png",
               hazard_distance);
    writeImage(output_directory / "safety_index.png", safety);
    writeImage(output_directory / "distance_index.png", distance);
    writeImage(output_directory / "global_safety_index.png", global);

    cv::Mat overlay(grid.height, grid.width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            cv::Vec3b color = grid.color_bgr[cell];
            if (analysis.hazard_mask.at<std::uint8_t>(row, column) != 0) {
                color = cv::Vec3b(
                    static_cast<std::uint8_t>(0.35 * color[0]),
                    static_cast<std::uint8_t>(0.35 * color[1]),
                    static_cast<std::uint8_t>(0.35 * color[2] + 165.0));
            }
            overlay.at<cv::Vec3b>(row, column) = color;
        }
    }
    if (analysis.best_site.found) {
        const cv::Point best(analysis.best_site.column,
                             analysis.best_site.row);
        const int footprint_radius = std::max(
            3, static_cast<int>(std::lround(
                   0.5 * config.uav_footprint_diagonal_m /
                   grid.resolution_m)));
        cv::circle(overlay, best, footprint_radius,
                   cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::drawMarker(overlay, best, cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
    } else {
        cv::putText(overlay, "NO ADMISSIBLE LANDING SITE",
                    cv::Point(12, std::max(24, grid.height / 2)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    if (demo_only) {
        cv::putText(overlay, "SYNTHETIC DEMO - NOT FOR FLIGHT",
                    cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    writeImage(output_directory / "best_landing_site.png", overlay);

    const std::array<cv::Mat, 8> panels{
        labeledPanel(overlay, "Best site (red = hazard)"),
        labeledPanel(smoothed_dem, "Anisotropic-smoothed DEM"),
        labeledPanel(slope, "Local slope (0 to 2x limit)"),
        labeledPanel(roughness, "Local roughness (0 to 2x limit)"),
        labeledPanel(hazard_distance, "Nearest-hazard distance"),
        labeledPanel(safety, "Fuzzy terrain safety index"),
        labeledPanel(distance, "Fuzzy distance index"),
        labeledPanel(global, "Global safety index")};
    cv::Mat overview(680, 1680, CV_8UC3, cv::Scalar(24, 24, 24));
    for (int index = 0; index < 8; ++index) {
        panels[static_cast<std::size_t>(index)].copyTo(
            overview(cv::Rect((index % 4) * 420,
                              (index / 4) * 340, 420, 340)));
    }
    writeImage(output_directory / "landing_analysis_overview.png",
               overview);

    detail::writeLandingJson(output_directory, analysis, config, uav_position_world_m, demo_only);
}
} // namespace metric_mapping
