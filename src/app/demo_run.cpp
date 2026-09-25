#include "stereo_app.hpp"
#include "../demo/synthetic_scene.hpp"
#include "metric_mapping/io.hpp"
#include "metric_mapping/point_cloud.hpp"
#include "metric_mapping/terrain.hpp"
#include "metric_mapping/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include "../output/stereo_metadata.hpp"
#include <iostream>
#include <stdexcept>

namespace metric_mapping::app {
void runSyntheticDemo(const std::filesystem::path& image_1_path,
                      const std::filesystem::path& image_2_path)
{
    const auto scene = makeSyntheticScene(image_1_path, image_2_path);
    const auto& synthetic_points = scene.points;
    const auto& matches_image = scene.matches_image;
    TerrainGrid grid = createTerrainGrid(synthetic_points, 0.05, 1000000U);
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 0.3;
    idw.minimum_neighbors = 3;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 0.3;
    interpolateIdw(grid, idw);
    const std::vector<ColoredPoint> completed = terrainGridPoints(grid);
    const GridStatistics statistics = computeGridStatistics(grid);
    const CloudBounds bounds = computeBounds(synthetic_points);
    const cv::Vec3d demo_uav_position(
        0.5 * (bounds.minimum[0] + bounds.maximum[0]),
        0.5 * (bounds.minimum[1] + bounds.maximum[1]),
        bounds.maximum[2] + 10.0);
    const LandingAnalysisConfig landing_config;
    const LandingAnalysis landing = analyzeLandingSites(
        grid, demo_uav_position, landing_config);

    const std::filesystem::path output_directory = "demo_output";
    std::filesystem::create_directories(output_directory);
    writePly(output_directory / "pointcloud_raw.ply", synthetic_points,
             "SYNTHETIC DEMO geometry; not reconstructed from the images");
    writePly(output_directory / "pointcloud.ply", completed,
             "SYNTHETIC DEMO one point per image pixel; not reconstructed "
             "from the images");
    writeDemCsv(output_directory / "dem.csv", grid);
    writeTerrainImages(output_directory, grid);
    writeDebugVisualization(output_directory / "debug_overview.png",
                            synthetic_points, bounds, grid, {}, std::nullopt);
    writeLandingAnalysis(output_directory, grid, landing, landing_config,
                         demo_uav_position, true);
    if (!cv::imwrite((output_directory / "matches.jpg").string(),
                     matches_image)) {
        throw std::runtime_error("Could not write demo matches image");
    }

    detail::writeDemoMetadata(output_directory, image_1_path, image_2_path,
                              scene.alignment_matches, synthetic_points.size(),
                              completed.size(), statistics);

    std::cout << "DEMO ONLY: Geometry is synthetic and was not reconstructed "
                 "from the images.\n";
    std::cout << "Aligned feature matches: " << scene.alignment_matches
              << '\n';
    std::cout << "Synthetic raw points: " << synthetic_points.size()
              << '\n';
    std::cout << "IDW interpolated cells: "
              << statistics.interpolated_cells << '\n';
    std::cout << "Final demo points: " << completed.size() << '\n';
    printLandingSummary(landing);
    std::cout << "Output directory: demo_output\n";
}

} // namespace metric_mapping::app
