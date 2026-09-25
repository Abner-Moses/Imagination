#include "metric_mapping/config.hpp"
#include "metric_mapping/geometry.hpp"
#include "metric_mapping/io.hpp"
#include "metric_mapping/point_cloud.hpp"
#include "metric_mapping/terrain.hpp"
#include "metric_mapping/visualization.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "rgbd_app.hpp"

namespace metric_mapping::app {
void runRgbd(const std::filesystem::path& config_path)
{
    const ReconstructionConfig config = loadConfig(config_path);
    std::filesystem::create_directories(config.output_directory);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coordinate frame: map_z_up (metres)\n";
    std::cout << "Camera: " << config.camera.width << 'x'
              << config.camera.height << " fx=" << config.camera.fx
              << " fy=" << config.camera.fy
              << " cx=" << config.camera.cx
              << " cy=" << config.camera.cy << '\n';
    std::cout << "Depth unit: " << depthUnitName(config.depth.unit)
              << " -> metres\n";
    std::cout << "Pose input: "
              << poseConventionName(config.pose_convention) << '\n';

    VoxelGridAccumulator fusion(config.fusion.voxel_size_m,
                                config.fusion.max_voxels);
    std::vector<ColoredPoint> raw_points;
    raw_points.reserve(std::min<std::size_t>(
        config.fusion.max_raw_points, 1000000U));
    std::vector<cv::Vec3d> camera_positions_world_m;
    camera_positions_world_m.reserve(config.frames.size());

    for (std::size_t index = 0; index < config.frames.size(); ++index) {
        const FrameSpec& frame = config.frames[index];
        const cv::Matx44d camera_to_world = cameraToWorldTransform(
            frame.supplied_pose, config.pose_convention);
        const FrameResult result = reconstructFrame(
            frame.rgb_path, frame.depth_path, config.camera,
            config.depth, camera_to_world);

        if (result.world_points.size() >
            config.fusion.max_raw_points - raw_points.size()) {
            throw std::runtime_error(
                "Raw point export exceeded fusion.max_raw_points; use a "
                "larger pixel_stride or explicitly raise the safety "
                "limit");
        }

        raw_points.insert(raw_points.end(), result.world_points.begin(),
                          result.world_points.end());
        fusion.add(result.world_points);
        const cv::Vec3d camera_position(camera_to_world(0, 3),
                                        camera_to_world(1, 3),
                                        camera_to_world(2, 3));
        camera_positions_world_m.push_back(camera_position);

        const FrameDiagnostics& diagnostics = result.diagnostics;
        std::cout << "Frame " << index << ": "
                  << frame.rgb_path.filename().string() << '\n';
        std::cout << "  Image: " << diagnostics.width << 'x'
                  << diagnostics.height << '\n';
        std::cout << "  Depth min/median/max: "
                  << diagnostics.minimum_depth_m << " / "
                  << diagnostics.median_depth_m << " / "
                  << diagnostics.maximum_depth_m << " m\n";
        std::cout << "  Valid/invalid depth pixels: "
                  << diagnostics.valid_depth_pixels << " / "
                  << diagnostics.invalid_depth_pixels << '\n';
        std::cout << "  Emitted points: "
                  << diagnostics.emitted_points << '\n';
        std::cout << "  T_WC translation: [" << camera_position[0]
                  << ", " << camera_position[1] << ", "
                  << camera_position[2] << "] m\n";
        std::cout << "  T_WC rotation rows: ["
                  << camera_to_world(0, 0) << ", "
                  << camera_to_world(0, 1) << ", "
                  << camera_to_world(0, 2) << "] ["
                  << camera_to_world(1, 0) << ", "
                  << camera_to_world(1, 1) << ", "
                  << camera_to_world(1, 2) << "] ["
                  << camera_to_world(2, 0) << ", "
                  << camera_to_world(2, 1) << ", "
                  << camera_to_world(2, 2) << "]\n";
    }

    const std::vector<ColoredPoint> filtered_points = fusion.points();
    if (filtered_points.empty()) {
        throw std::runtime_error(
            "No world points remain after voxel fusion");
    }
    const CloudBounds bounds = computeBounds(filtered_points);
    TerrainGrid grid = createTerrainGrid(
        filtered_points, config.map.resolution_m, config.map.max_cells);
    interpolateIdw(grid, config.map.idw);
    const GridStatistics grid_statistics = computeGridStatistics(grid);

    writePly(config.output_directory / "cloud_raw.ply", raw_points);
    writePly(config.output_directory / "cloud_filtered.ply",
             filtered_points);
    writeDemCsv(config.output_directory / "dem.csv", grid);
    writeTerrainImages(config.output_directory, grid);
    if (config.uav_position_world_m) {
        const LandingAnalysisConfig landing_config;
        const LandingAnalysis landing = analyzeLandingSites(
            grid, *config.uav_position_world_m, landing_config, config.ultrasonic);
        if (landing.ultrasonic)
            std::cout << "Ultrasonic ground check: " << landing.ultrasonic->status() << '\n';
        writeLandingAnalysis(config.output_directory, grid, landing,
                             landing_config,
                             *config.uav_position_world_m, false);
        std::cout << "Landing candidate cells: "
                  << landing.candidate_cells << '\n';
        if (landing.best_site.found) {
            std::cout << "Best landing XYZ: ["
                      << landing.best_site.world_m[0] << ", "
                      << landing.best_site.world_m[1] << ", "
                      << landing.best_site.world_m[2] << "] m\n";
            std::cout << "Best landing global safety index: "
                      << landing.best_site.global_safety_index
                      << "%\n";
        } else {
            std::cout << "Best landing site: none satisfies every "
                         "constraint\n";
        }
    }
    writeDebugVisualization(
        config.output_directory / "debug_overview.png", filtered_points,
        bounds, grid, camera_positions_world_m,
        config.uav_position_world_m);
    writeMetadata(config.output_directory / "metadata.json", config,
                  raw_points.size(), filtered_points, bounds, grid,
                  grid_statistics, camera_positions_world_m);

    const std::size_t total_cells = grid.validity.size();
    const auto percent = [total_cells](std::size_t count) {
        return total_cells == 0
                   ? 0.0
                   : 100.0 * static_cast<double>(count) /
                         static_cast<double>(total_cells);
    };

    std::cout << "Point cloud raw/filtered: " << raw_points.size()
              << " / " << filtered_points.size() << '\n';
    std::cout << "Point bounds X: [" << bounds.minimum[0] << ", "
              << bounds.maximum[0] << "] m\n";
    std::cout << "Point bounds Y: [" << bounds.minimum[1] << ", "
              << bounds.maximum[1] << "] m\n";
    std::cout << "Point bounds Z: [" << bounds.minimum[2] << ", "
              << bounds.maximum[2] << "] m\n";
    std::cout << "Map: " << grid.width << 'x' << grid.height
              << " at " << grid.resolution_m << " m/cell\n";
    std::cout << "Map measured/interpolated/unknown: "
              << percent(grid_statistics.measured_cells) << "% / "
              << percent(grid_statistics.interpolated_cells) << "% / "
              << percent(grid_statistics.unknown_cells) << "%\n";
    if (!config.uav_position_world_m) {
        std::cout << "UAV position: not supplied; paper-style landing "
                     "analysis skipped\n";
    }
    std::cout << "Output directory: "
              << config.output_directory.string() << '\n';
}
} // namespace metric_mapping::app
