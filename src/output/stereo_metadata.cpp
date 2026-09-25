#include "stereo_metadata.hpp"
#include "metric_mapping/io.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace metric_mapping::detail {
void writeTwoViewMetadata(
    const std::filesystem::path& path,
    const metric_mapping::CameraIntrinsics& camera,
    double baseline_m,
    const metric_mapping::TwoViewResult& result,
    const metric_mapping::TerrainGrid& grid,
    const metric_mapping::GridStatistics& statistics,
    std::size_t completed_point_count)
{
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Could not write " + path.string());
    output << std::fixed << std::setprecision(8);
    output << "{\n";
    output << "  \"method\": \"calibrated two-view dense stereo\",\n";
    output << "  \"coordinate_frame\": "
              "\"local_nadir_z_up, camera_1_origin\",\n";
    output << "  \"units\": \"metres\",\n";
    output << "  \"required_assumption\": "
              "\"both cameras are nadir-facing at equal altitude with no "
              "pitch/roll change, and baseline_m is measured\",\n";
    output << "  \"camera\": {\"width\": " << camera.width
           << ", \"height\": " << camera.height
           << ", \"fx\": " << camera.fx
           << ", \"fy\": " << camera.fy
           << ", \"cx\": " << camera.cx
           << ", \"cy\": " << camera.cy << "},\n";
    output << "  \"baseline_m\": " << baseline_m << ",\n";
    output << "  \"features_image_1\": "
           << result.diagnostics.features_image_1 << ",\n";
    output << "  \"features_image_2\": "
           << result.diagnostics.features_image_2 << ",\n";
    output << "  \"good_matches\": "
           << result.diagnostics.good_matches << ",\n";
    output << "  \"geometric_inliers\": "
           << result.diagnostics.ransac_inliers << ",\n";
    output << "  \"parallel_alignment_inliers\": "
           << result.diagnostics.alignment_inliers << ",\n";
    output << "  \"median_rectified_epipolar_error_px\": "
           << result.diagnostics.median_rectified_epipolar_error_px
           << ",\n";
    output << "  \"raw_dense_points\": " << result.points.size()
           << ",\n";
    output << "  \"completed_grid_points\": "
           << completed_point_count << ",\n";
    output << "  \"map_resolution_m\": " << grid.resolution_m
           << ",\n";
    output << "  \"map_width\": " << grid.width << ",\n";
    output << "  \"map_height\": " << grid.height << ",\n";
    output << "  \"map_minimum_x_m\": " << grid.minimum_x_m << ",\n";
    output << "  \"map_maximum_y_m\": " << grid.maximum_y_m << ",\n";
    output << "  \"measured_cells\": " << statistics.measured_cells
           << ",\n";
    output << "  \"idw_interpolated_cells\": "
           << statistics.interpolated_cells << ",\n";
    output << "  \"unknown_cells\": " << statistics.unknown_cells
           << "\n";
    output << "}\n";
    output.close();
    if (!output)
        throw std::runtime_error("Could not finish writing " +
                                 path.string());
}

void writeDemoMetadata(const std::filesystem::path& output_directory,
                       const std::filesystem::path& image_1_path,
                       const std::filesystem::path& image_2_path,
                       std::size_t alignment_matches, std::size_t raw_points,
                       std::size_t completed_points, const GridStatistics& statistics)
{
    std::ofstream metadata(output_directory / "metadata.json");
    if (!metadata)
        throw std::runtime_error("Could not write demo metadata");
    metadata << "{\n"
             << "  \"demo_only\": true,\n"
             << "  \"not_a_reconstruction\": true,\n"
             << "  \"geometry_source\": "
                "\"deterministic synthetic height from image appearance\",\n"
             << "  \"input_image_1\": \"" << escapeJson(image_1_path.string())
             << "\",\n"
             << "  \"input_image_2\": \"" << escapeJson(image_2_path.string())
             << "\",\n"
             << "  \"homography_inlier_matches\": "
             << alignment_matches << ",\n"
             << "  \"raw_points\": " << raw_points
             << ",\n"
             << "  \"completed_points\": " << completed_points << ",\n"
             << "  \"measured_cells\": " << statistics.measured_cells
             << ",\n"
             << "  \"idw_interpolated_cells\": "
             << statistics.interpolated_cells << ",\n"
             << "  \"landing_analysis\": \"landing_site.json\"\n"
             << "}\n";
    metadata.close();
    if (!metadata)
        throw std::runtime_error("Could not finish demo metadata");

}
} // namespace metric_mapping::detail
