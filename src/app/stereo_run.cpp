#include "stereo_app.hpp"
#include "../output/stereo_metadata.hpp"
#include "metric_mapping/io.hpp"
#include "metric_mapping/point_cloud.hpp"
#include "metric_mapping/terrain.hpp"
#include "metric_mapping/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace metric_mapping::app {

void printLandingSummary(
    const metric_mapping::LandingAnalysis& analysis)
{
    std::cout << "Terrain hazard cells: " << analysis.hazard_cells << '\n';
    std::cout << "Landing candidate cells: "
              << analysis.candidate_cells << '\n';
    if (!analysis.best_site.found) {
        std::cout << "Best landing site: none satisfies every constraint\n";
        return;
    }
    const metric_mapping::LandingSite& site = analysis.best_site;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Best landing XYZ: " << site.world_m[0] << ' '
              << site.world_m[1] << ' ' << site.world_m[2] << '\n';
    std::cout << "Best landing slope/roughness: "
              << site.slope_degrees << " deg / "
              << site.roughness_m << " m\n";
    std::cout << "Best landing global safety index: "
              << site.global_safety_index << "%\n";
}

void runStereo(const std::filesystem::path& image_1_path,
               const std::filesystem::path& image_2_path,
               const CameraIntrinsics& camera, double baseline_m)
{
    const TwoViewResult result =
        reconstructTwoView(image_1_path, image_2_path,
                                           camera, baseline_m);
    TerrainGrid grid =
        createTerrainGrid(result.points, 0.1, 5000000U);
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 0.3;
    idw.minimum_neighbors = 3;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 0.3;
    interpolateIdw(grid, idw);
    const GridStatistics statistics =
        computeGridStatistics(grid);
    const std::vector<ColoredPoint> completed_points =
        terrainGridPoints(grid);
    const CloudBounds bounds =
        computeBounds(result.points);
    const cv::Vec3d uav_position =
        result.camera_positions_world_m.empty()
            ? cv::Vec3d(0.0, 0.0, 0.0)
            : result.camera_positions_world_m.front();
    const LandingAnalysisConfig landing_config;
    const LandingAnalysis landing =
        analyzeLandingSites(
            grid, uav_position, landing_config);

    const std::filesystem::path cloud_path = "pointcloud.ply";
    const std::filesystem::path raw_cloud_path = "pointcloud_raw.ply";
    const std::filesystem::path matches_path = "matches.jpg";
    const std::filesystem::path disparity_path = "disparity.png";
    writePly(
        raw_cloud_path, result.points,
        "metres in local nadir z-up frame; camera 1 is the origin");
    writePly(
        cloud_path, completed_points,
        "metres in local nadir z-up frame; 0.1 m terrain grid with "
        "bounded IDW completion");
    writeDemCsv("dem.csv", grid);
    writeTerrainImages(".", grid);
    writeDebugVisualization(
        "debug_overview.png", result.points, bounds, grid,
        result.camera_positions_world_m, cv::Vec3d(0.0, 0.0, 0.0));
    writeLandingAnalysis(
        ".", grid, landing, landing_config, uav_position, false);
    detail::writeTwoViewMetadata("metadata.json", camera, baseline_m, result,
                         grid, statistics, completed_points.size());
    if (!cv::imwrite(matches_path.string(), result.matches_image) ||
        !cv::imwrite(disparity_path.string(),
                     result.disparity_preview)) {
        throw std::runtime_error(
            "Could not write matches/disparity images");
    }

    std::cout << "Features image 1: "
              << result.diagnostics.features_image_1 << '\n';
    std::cout << "Features image 2: "
              << result.diagnostics.features_image_2 << '\n';
    std::cout << "Good matches: "
              << result.diagnostics.good_matches << '\n';
    std::cout << "RANSAC inliers: "
              << result.diagnostics.ransac_inliers << '\n';
    std::cout << "Dense stereo points: " << result.points.size()
              << '\n';
    std::cout << "Measured terrain cells: "
              << statistics.measured_cells << '\n';
    std::cout << "IDW interpolated cells: "
              << statistics.interpolated_cells << '\n';
    std::cout << "Final 3D points: "
              << completed_points.size() << '\n';
    printLandingSummary(landing);
    std::cout << "Output: pointcloud.ply\n";
    std::cout << "DEM: dem.csv\n";
    std::cout << "Orthomosaic: orthomosaic.png\n";
    std::cout << "Landing analysis: landing_analysis_overview.png\n";
    std::cout << "WARNING: Metric accuracy requires a measured baseline, "
                 "calibration at the input resolution, and a nadir-facing camera.\n";
}
} // namespace metric_mapping::app
