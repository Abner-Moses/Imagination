#include "metric_mapping/io.hpp"
#include "metric_mapping/config.hpp"
#include "output_internal.hpp"
#include <iomanip>

namespace metric_mapping {
using detail::requireWritable;
namespace {
double percentage(std::size_t count, std::size_t total)
{
    return total == 0
               ? 0.0
               : 100.0 * static_cast<double>(count) /
                     static_cast<double>(total);
}

} // namespace

void writeMetadata(const std::filesystem::path& path,
                   const ReconstructionConfig& config,
                   std::size_t raw_point_count,
                   const std::vector<ColoredPoint>& filtered_points,
                   const CloudBounds& bounds,
                   const TerrainGrid& grid,
                   const GridStatistics& grid_statistics,
                   const std::vector<cv::Vec3d>& camera_positions_world_m)
{
    std::ofstream output(path);
    requireWritable(output, path);
    output << std::fixed << std::setprecision(8);

    const std::size_t total_cells = grid.validity.size();
    const double maximum_x = grid.minimum_x_m +
        (grid.width - 1) * grid.resolution_m;
    const double minimum_y = grid.maximum_y_m -
        (grid.height - 1) * grid.resolution_m;

    output << "{\n";
    output << "  \"coordinate_frame\": {\n";
    output << "    \"name\": \"map_z_up\",\n";
    output << "    \"units\": \"metres\",\n";
    output << "    \"x_axis\": \"horizontal\",\n";
    output << "    \"y_axis\": \"horizontal\",\n";
    output << "    \"z_axis\": \"up/elevation\",\n";
    output << "    \"camera_axes_before_T_WC\": "
              "\"OpenCV: +X right, +Y down, +Z forward\",\n";
    output << "    \"grid_indexing\": "
              "\"column increases +X; row increases -Y\"\n";
    output << "  },\n";
    output << "  \"camera_intrinsics\": {\n";
    output << "    \"width\": " << config.camera.width << ",\n";
    output << "    \"height\": " << config.camera.height << ",\n";
    output << "    \"fx_px\": " << config.camera.fx << ",\n";
    output << "    \"fy_px\": " << config.camera.fy << ",\n";
    output << "    \"cx_px\": " << config.camera.cx << ",\n";
    output << "    \"cy_px\": " << config.camera.cy << "\n";
    output << "  },\n";
    output << "  \"depth\": {\n";
    output << "    \"input_unit\": \""
           << depthUnitName(config.depth.unit) << "\",\n";
    output << "    \"output_unit\": \"metres\",\n";
    output << "    \"minimum_m\": " << config.depth.min_depth_m
           << ",\n";
    output << "    \"maximum_m\": " << config.depth.max_depth_m
           << ",\n";
    output << "    \"pixel_stride\": " << config.depth.pixel_stride
           << "\n";
    output << "  },\n";
    output << "  \"pose_convention_supplied\": \""
           << poseConventionName(config.pose_convention) << "\",\n";
    output << "  \"internal_pose_convention\": "
              "\"T_WC maps camera coordinates to map_z_up world "
              "coordinates\",\n";
    output << "  \"number_of_input_frames\": " << config.frames.size()
           << ",\n";
    output << "  \"raw_point_count\": " << raw_point_count << ",\n";
    output << "  \"filtered_point_count\": " << filtered_points.size()
           << ",\n";
    output << "  \"voxel_size_m\": " << config.fusion.voxel_size_m
           << ",\n";
    output << "  \"point_cloud_bounds_m\": {\n";
    output << "    \"minimum\": [" << bounds.minimum[0] << ", "
           << bounds.minimum[1] << ", " << bounds.minimum[2] << "],\n";
    output << "    \"maximum\": [" << bounds.maximum[0] << ", "
           << bounds.maximum[1] << ", " << bounds.maximum[2] << "]\n";
    output << "  },\n";
    output << "  \"map\": {\n";
    output << "    \"resolution_m\": " << grid.resolution_m << ",\n";
    output << "    \"width_cells\": " << grid.width << ",\n";
    output << "    \"height_cells\": " << grid.height << ",\n";
    output << "    \"minimum_x_m\": " << grid.minimum_x_m << ",\n";
    output << "    \"maximum_x_m\": " << maximum_x << ",\n";
    output << "    \"minimum_y_m\": " << minimum_y << ",\n";
    output << "    \"maximum_y_m\": " << grid.maximum_y_m << ",\n";
    output << "    \"measured_percent\": "
           << percentage(grid_statistics.measured_cells, total_cells)
           << ",\n";
    output << "    \"interpolated_percent\": "
           << percentage(grid_statistics.interpolated_cells, total_cells)
           << ",\n";
    output << "    \"unknown_percent\": "
           << percentage(grid_statistics.unknown_cells, total_cells) << "\n";
    output << "  },\n";
    output << "  \"interpolation\": {\n";
    output << "    \"enabled\": "
           << (config.map.idw.enabled ? "true" : "false") << ",\n";
    output << "    \"method\": \"inverse_distance_weighting\",\n";
    output << "    \"search_radius_m\": "
           << config.map.idw.search_radius_m << ",\n";
    output << "    \"minimum_neighbors\": "
           << config.map.idw.minimum_neighbors << ",\n";
    output << "    \"power\": " << config.map.idw.power << ",\n";
    output << "    \"maximum_interpolation_distance_m\": "
           << config.map.idw.maximum_interpolation_distance_m << "\n";
    output << "  },\n";

    output << "  \"camera_trajectory_world_m\": [";
    for (std::size_t index = 0; index < camera_positions_world_m.size();
         ++index) {
        if (index != 0)
            output << ',';
        const cv::Vec3d& position = camera_positions_world_m[index];
        output << "\n    [" << position[0] << ", " << position[1] << ", "
               << position[2] << ']';
    }
    if (!camera_positions_world_m.empty())
        output << '\n';
    output << "  ],\n";

    output << "  \"uav_position_world_m\": ";
    if (config.uav_position_world_m) {
        const cv::Vec3d& position = *config.uav_position_world_m;
        output << '[' << position[0] << ", " << position[1] << ", "
               << position[2] << ']';
    } else {
        output << "null";
    }
    output << ",\n";

    output << "  \"input_frames\": [";
    for (std::size_t index = 0; index < config.frames.size(); ++index) {
        if (index != 0)
            output << ',';
        output << "\n    {\"rgb\": \""
               << escapeJson(config.frames[index].rgb_path.string())
               << "\", \"depth\": \""
               << escapeJson(config.frames[index].depth_path.string())
               << "\"}";
    }
    if (!config.frames.empty())
        output << '\n';
    output << "  ],\n";
    output << "  \"implementation_choices_not_specified_by_paper\": [\n";
    output << "    \"Metric RGB-D pinhole back-projection is the front-end "
              "input contract\",\n";
    output << "    \"Voxel fusion averages XYZ and RGB observations per "
              "metric voxel\",\n";
    output << "    \"Each terrain cell uses the highest measured +Z point "
              "and its color\",\n";
    output << "    \"IDW only uses directly measured neighbors and never "
              "propagates interpolated cells\"\n";
    output << "  ]\n";
    output << "}\n";
    output.close();
    requireWritable(output, path);
}

} // namespace metric_mapping
