#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// --- Raster ---

namespace metric_mapping::detail {
cv::Mat terrainValidMask(const TerrainGrid& grid)
{
    cv::Mat mask(grid.height, grid.width, CV_8U, cv::Scalar(0));
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (grid.validity[grid.index(row, column)] != 0)
                mask.at<std::uint8_t>(row, column) = 255;
        }
    }
    return mask;
}

cv::Mat colorizeFloat(const cv::Mat& values,
                      const cv::Mat& valid_mask,
                      double minimum,
                      double maximum)
{
    cv::Mat scalar(values.size(), CV_8U, cv::Scalar(0));
    const double span = std::max(maximum - minimum, 1e-12);
    for (int row = 0; row < values.rows; ++row) {
        for (int column = 0; column < values.cols; ++column) {
            if (valid_mask.at<std::uint8_t>(row, column) == 0)
                continue;
            const float value = values.at<float>(row, column);
            if (!std::isfinite(value))
                continue;
            scalar.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(std::clamp(
                    std::lround(255.0 * (value - minimum) / span),
                    0L, 255L));
        }
    }
    cv::Mat color;
    cv::applyColorMap(scalar, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);
    return color;
}

cv::Mat colorizeDem(const cv::Mat& dem, const cv::Mat& valid_mask)
{
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < dem.rows; ++row) {
        for (int column = 0; column < dem.cols; ++column) {
            if (valid_mask.at<std::uint8_t>(row, column) == 0)
                continue;
            const float value = dem.at<float>(row, column);
            if (std::isfinite(value)) {
                minimum = std::min(minimum, static_cast<double>(value));
                maximum = std::max(maximum, static_cast<double>(value));
            }
        }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum))
        return cv::Mat(dem.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    return colorizeFloat(dem, valid_mask, minimum, maximum);
}

void writeImage(const std::filesystem::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Could not write " + path.string());
}

cv::Mat labeledPanel(const cv::Mat& image, const std::string& title)
{
    constexpr int width = 420;
    constexpr int height = 340;
    cv::Mat panel(height, width, CV_8UC3, cv::Scalar(24, 24, 24));
    const cv::Mat resized = fitImage(image, cv::Size(width, height - 34));
    const int x = (width - resized.cols) / 2;
    const int y = 30 + (height - 30 - resized.rows) / 2;
    resized.copyTo(panel(cv::Rect(x, y, resized.cols, resized.rows)));
    cv::putText(panel, title, cv::Point(10, 21),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
    return panel;
}

} // namespace metric_mapping::detail

// --- Cloud Files ---

namespace metric_mapping {
namespace detail {
void requireWritable(const std::ofstream& stream,
                     const std::filesystem::path& path)
{
    if (!stream) {
        throw std::runtime_error("Could not write output: " + path.string());
    }
}

} // namespace detail
using detail::requireWritable;

std::string escapeJson(const std::string& value)
{
    std::ostringstream escaped;
    for (unsigned char character : value) {
        if (character == '"' || character == '\\')
            escaped << '\\' << character;
        else if (character < 0x20)
            escaped << "\\u" << std::hex << std::setw(4)
                    << std::setfill('0') << static_cast<int>(character);
        else
            escaped << character;
    }
    return escaped.str();
}

void writePly(const std::filesystem::path& path,
              const std::vector<ColoredPoint>& points,
              const std::string& coordinate_comment)
{
    std::ofstream output(path);
    requireWritable(output, path);

    output << "ply\n";
    output << "format ascii 1.0\n";
    output << "comment " << coordinate_comment << '\n';
    output << "element vertex " << points.size() << '\n';
    output << "property float x\n";
    output << "property float y\n";
    output << "property float z\n";
    output << "property uchar red\n";
    output << "property uchar green\n";
    output << "property uchar blue\n";
    output << "end_header\n";
    output << std::fixed << std::setprecision(7);

    for (const ColoredPoint& point : points) {
        output << point.x << ' ' << point.y << ' ' << point.z << ' '
               << static_cast<int>(point.r) << ' '
               << static_cast<int>(point.g) << ' '
               << static_cast<int>(point.b) << '\n';
    }
    output.close();
    requireWritable(output, path);
}

void writeDemCsv(const std::filesystem::path& path,
                 const TerrainGrid& grid)
{
    std::ofstream output(path);
    requireWritable(output, path);
    output << std::fixed << std::setprecision(7);

    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (column != 0)
                output << ',';
            const std::size_t cell = grid.index(row, column);
            if (grid.validity[cell] == 0 ||
                !std::isfinite(grid.elevation_m[cell])) {
                output << "nan";
            } else {
                output << grid.elevation_m[cell];
            }
        }
        output << '\n';
    }
    output.close();
    requireWritable(output, path);
}

void writeTerrainImages(const std::filesystem::path& output_directory,
                        const TerrainGrid& grid)
{
    cv::Mat orthomosaic(grid.height, grid.width, CV_8UC3,
                        cv::Scalar(0, 0, 0));
    cv::Mat validity(grid.height, grid.width, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            orthomosaic.at<cv::Vec3b>(row, column) = grid.color_bgr[cell];
            validity.at<std::uint8_t>(row, column) = grid.validity[cell];
        }
    }

    const std::filesystem::path orthomosaic_path =
        output_directory / "orthomosaic.png";
    const std::filesystem::path validity_path =
        output_directory / "validity_mask.png";
    if (!cv::imwrite(orthomosaic_path.string(), orthomosaic) ||
        !cv::imwrite(validity_path.string(), validity)) {
        throw std::runtime_error("Could not write terrain raster images");
    }
}

} // namespace metric_mapping

// --- Rgbd Metadata ---

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

// --- Landing Json ---

namespace metric_mapping::detail {
void writeLandingJson(const std::filesystem::path& output_directory,
                      const LandingAnalysis& analysis, const LandingAnalysisConfig& config,
                      const cv::Vec3d& uav_position_world_m, bool demo_only)
{
    const std::filesystem::path json_path =
        output_directory / "landing_site.json";
    std::ofstream json(json_path);
    requireWritable(json, json_path);
    json << std::fixed << std::setprecision(8);
    json << "{\n";
    json << "  \"demo_only\": " << (demo_only ? "true" : "false")
         << ",\n";
    json << "  \"warning\": \""
         << (demo_only
                 ? "Synthetic appearance-derived geometry; not valid for flight"
                 : "Verify calibration, scale, thresholds, and flight regulations before use")
         << "\",\n";
    json << "  \"water_detection\": "
            "\"disabled; dry-scene assumption (paper uses a neural model)\",\n";
    json << "  \"uav_position_world_m\": ["
         << uav_position_world_m[0] << ", "
         << uav_position_world_m[1] << ", "
         << uav_position_world_m[2] << "],\n";
    json << "  \"ultrasonic\": ";
    if (!analysis.ultrasonic) {
        json << "null,\n";
    } else {
        const auto& check = *analysis.ultrasonic;
        const auto& reading = check.measurement;
        json << "{\n    \"status\": \"" << check.status() << "\",\n"
             << "    \"direction\": \"world_down\",\n"
             << "    \"distance_m\": " << reading.distance_m << ",\n"
             << "    \"maximum_error_m\": " << reading.maximum_error_m << ",\n"
             << "    \"sensor_offset_world_m\": [" << reading.sensor_offset_world_m[0]
             << ", " << reading.sensor_offset_world_m[1] << ", "
             << reading.sensor_offset_world_m[2] << "],\n"
             << "    \"mapped_distance_m\": ";
        if (check.mapped_distance_m) json << *check.mapped_distance_m;
        else json << "null";
        json << "\n  },\n";
    }
    json << "  \"constraints\": {\n";
    json << "    \"maximum_slope_degrees\": "
         << config.maximum_slope_degrees << ",\n";
    json << "    \"maximum_roughness_m\": "
         << config.maximum_roughness_m << ",\n";
    json << "    \"minimum_hazard_distance_m\": "
         << config.minimum_hazard_distance_m << ",\n";
    json << "    \"uav_footprint_diagonal_m\": "
         << config.uav_footprint_diagonal_m << ",\n";
    json << "    \"minimum_global_safety_index\": "
         << config.minimum_global_safety_index << "\n";
    json << "  },\n";
    json << "  \"hazard_cells\": " << analysis.hazard_cells << ",\n";
    json << "  \"admissible_cells\": "
         << analysis.admissible_cells << ",\n";
    json << "  \"clearance_cells\": "
         << analysis.clearance_cells << ",\n";
    json << "  \"candidate_cells\": "
         << analysis.candidate_cells << ",\n";
    json << "  \"maximum_global_safety_index\": "
         << analysis.maximum_global_safety_index << ",\n";
    json << "  \"landing_site_found\": "
         << (analysis.best_site.found ? "true" : "false") << ",\n";
    json << "  \"best_landing_site\": ";
    if (!analysis.best_site.found) {
        json << "null\n";
    } else {
        const LandingSite& site = analysis.best_site;
        json << "{\n";
        json << "    \"grid_row\": " << site.row << ",\n";
        json << "    \"grid_column\": " << site.column << ",\n";
        json << "    \"world_m\": [" << site.world_m[0] << ", "
             << site.world_m[1] << ", " << site.world_m[2] << "],\n";
        json << "    \"slope_degrees\": " << site.slope_degrees
             << ",\n";
        json << "    \"roughness_m\": " << site.roughness_m << ",\n";
        json << "    \"nearest_hazard_distance_m\": "
             << site.nearest_hazard_distance_m << ",\n";
        json << "    \"spatial_distance_m\": "
             << site.spatial_distance_m << ",\n";
        json << "    \"safety_index\": " << site.safety_index << ",\n";
        json << "    \"distance_index\": " << site.distance_index
             << ",\n";
        json << "    \"global_safety_index\": "
             << site.global_safety_index << "\n";
        json << "  }\n";
    }
    json << "}\n";
    json.close();
    requireWritable(json, json_path);
}

} // namespace metric_mapping::detail

// --- Landing Images ---

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

// --- Overview ---

namespace metric_mapping {
namespace {

constexpr int panel_width = 600;
constexpr int panel_height = 500;

cv::Mat letterbox(const cv::Mat& source, const std::string& title)
{
    cv::Mat panel(panel_height, panel_width, CV_8UC3,
                  cv::Scalar(24, 24, 24));
    if (!source.empty()) {
        const cv::Mat resized = fitImage(
            source, cv::Size(panel_width, panel_height - 35));
        const int x = (panel_width - resized.cols) / 2;
        const int y = 30 + (panel_height - 30 - resized.rows) / 2;
        resized.copyTo(panel(cv::Rect(x, y, resized.cols, resized.rows)));
    }
    cv::putText(panel, title, cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    return panel;
}

bool insideGrid(const TerrainGrid& grid, const cv::Point& point)
{
    return point.x >= 0 && point.x < grid.width && point.y >= 0 &&
           point.y < grid.height;
}

cv::Point worldToGrid(const TerrainGrid& grid, const cv::Vec3d& point)
{
    return cv::Point(
        static_cast<int>(std::lround(
            (point[0] - grid.minimum_x_m) / grid.resolution_m)),
        static_cast<int>(std::lround(
            (grid.maximum_y_m - point[1]) / grid.resolution_m)));
}

void drawAxes(cv::Mat& panel, bool top_down)
{
    const cv::Point origin(55, panel.rows - 45);
    cv::arrowedLine(panel, origin, origin + cv::Point(75, 0),
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(panel, "+X", origin + cv::Point(80, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::arrowedLine(panel, origin, origin + cv::Point(0, -75),
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(panel, top_down ? "+Y" : "+Z",
                origin + cv::Point(-15, -82), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

}  // namespace

cv::Mat fitImage(const cv::Mat& image, cv::Size box)
{
    if (image.empty() || box.width < 1 || box.height < 1)
        throw std::runtime_error("Cannot resize an empty image or panel");
    const double scale = std::min(double(box.width) / image.cols,
                                  double(box.height) / image.rows);
    const cv::Size size(std::max(1, cvRound(image.cols * scale)),
                        std::max(1, cvRound(image.rows * scale)));
    cv::Mat resized;
    cv::resize(image, resized, size, 0, 0, cv::INTER_NEAREST);
    return resized;
}

void writeDebugVisualization(
    const std::filesystem::path& path,
    const std::vector<ColoredPoint>& points,
    const CloudBounds& bounds,
    const TerrainGrid& grid,
    const std::vector<cv::Vec3d>& camera_positions_world_m,
    const std::optional<cv::Vec3d>& uav_position_world_m)
{
    if (grid.width <= 0 || grid.height <= 0) {
        throw std::runtime_error("Cannot visualize an empty terrain grid");
    }

    cv::Mat orthomosaic(grid.height, grid.width, CV_8UC3,
                        cv::Scalar(0, 0, 0));
    cv::Mat validity(grid.height, grid.width, CV_8UC3,
                     cv::Scalar(0, 0, 0));
    cv::Mat dem_scalar(grid.height, grid.width, CV_8UC1, cv::Scalar(0));

    double minimum_z = std::numeric_limits<double>::infinity();
    double maximum_z = -std::numeric_limits<double>::infinity();
    for (std::size_t cell = 0; cell < grid.validity.size(); ++cell) {
        if (grid.validity[cell] != 0 &&
            std::isfinite(grid.elevation_m[cell])) {
            minimum_z = std::min(minimum_z,
                                 static_cast<double>(grid.elevation_m[cell]));
            maximum_z = std::max(maximum_z,
                                 static_cast<double>(grid.elevation_m[cell]));
        }
    }
    const double z_span = std::max(maximum_z - minimum_z, 1e-9);

    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const std::size_t cell = grid.index(row, column);
            orthomosaic.at<cv::Vec3b>(row, column) = grid.color_bgr[cell];
            if (grid.validity[cell] == 255) {
                validity.at<cv::Vec3b>(row, column) =
                    cv::Vec3b(255, 255, 255);
            } else if (grid.validity[cell] == 127) {
                validity.at<cv::Vec3b>(row, column) =
                    cv::Vec3b(0, 220, 255);
            }
            if (grid.validity[cell] != 0) {
                dem_scalar.at<std::uint8_t>(row, column) =
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(255.0 *
                                    (grid.elevation_m[cell] - minimum_z) /
                                    z_span),
                        0L, 255L));
            }
        }
    }

    cv::Mat dem_color;
    cv::applyColorMap(dem_scalar, dem_color, cv::COLORMAP_TURBO);
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            if (grid.validity[grid.index(row, column)] == 0) {
                dem_color.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    cv::Mat top_panel = letterbox(
        orthomosaic, "Orthomosaic: cyan trajectory, magenta UAV");
    cv::Mat dem_panel = letterbox(dem_color, "DEM preview (+Z elevation)");
    cv::Mat validity_panel = letterbox(
        validity, "Validity: white measured, yellow IDW, black unknown");

    const double top_scale = std::min(
        static_cast<double>(panel_width) / grid.width,
        static_cast<double>(panel_height - 35) / grid.height);
    const int top_x_offset =
        (panel_width - static_cast<int>(std::lround(grid.width * top_scale))) /
        2;
    const int top_y_offset = 30 +
        (panel_height - 30 -
         static_cast<int>(std::lround(grid.height * top_scale))) /
            2;
    const auto topPixel = [&](const cv::Vec3d& position) {
        const cv::Point pixel = worldToGrid(grid, position);
        return cv::Point(
            top_x_offset + static_cast<int>(std::lround(pixel.x * top_scale)),
            top_y_offset + static_cast<int>(std::lround(pixel.y * top_scale)));
    };

    for (std::size_t index = 1; index < camera_positions_world_m.size();
         ++index) {
        const cv::Point previous_grid =
            worldToGrid(grid, camera_positions_world_m[index - 1]);
        const cv::Point current_grid =
            worldToGrid(grid, camera_positions_world_m[index]);
        if (insideGrid(grid, previous_grid) &&
            insideGrid(grid, current_grid)) {
            cv::line(top_panel,
                     topPixel(camera_positions_world_m[index - 1]),
                     topPixel(camera_positions_world_m[index]),
                     cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        }
    }
    for (const cv::Vec3d& position : camera_positions_world_m) {
        if (insideGrid(grid, worldToGrid(grid, position))) {
            cv::circle(top_panel, topPixel(position), 4,
                       cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
        }
    }
    if (uav_position_world_m &&
        insideGrid(grid, worldToGrid(grid, *uav_position_world_m))) {
        cv::drawMarker(top_panel, topPixel(*uav_position_world_m),
                       cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 14, 2,
                       cv::LINE_AA);
    }
    drawAxes(top_panel, true);

    cv::Mat side(panel_height, panel_width, CV_8UC3,
                 cv::Scalar(24, 24, 24));
    cv::putText(side, "Colored cloud side view (display transform only)",
                cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    double display_minimum_x = bounds.minimum[0];
    double display_maximum_x = bounds.maximum[0];
    double display_minimum_z = bounds.minimum[2];
    double display_maximum_z = bounds.maximum[2];
    for (const cv::Vec3d& position : camera_positions_world_m) {
        display_minimum_x = std::min(display_minimum_x, position[0]);
        display_maximum_x = std::max(display_maximum_x, position[0]);
        display_minimum_z = std::min(display_minimum_z, position[2]);
        display_maximum_z = std::max(display_maximum_z, position[2]);
    }
    if (uav_position_world_m) {
        display_minimum_x =
            std::min(display_minimum_x, (*uav_position_world_m)[0]);
        display_maximum_x =
            std::max(display_maximum_x, (*uav_position_world_m)[0]);
        display_minimum_z =
            std::min(display_minimum_z, (*uav_position_world_m)[2]);
        display_maximum_z =
            std::max(display_maximum_z, (*uav_position_world_m)[2]);
    }
    const double x_span =
        std::max(display_maximum_x - display_minimum_x, 1e-9);
    const double cloud_z_span =
        std::max(display_maximum_z - display_minimum_z, 1e-9);
    const auto sidePixel = [&](double x, double z) {
        return cv::Point(
            30 + static_cast<int>(std::lround(
                     (panel_width - 60) * (x - display_minimum_x) / x_span)),
            panel_height - 30 - static_cast<int>(std::lround(
                (panel_height - 70) * (z - display_minimum_z) /
                cloud_z_span)));
    };

    const std::size_t display_stride =
        std::max<std::size_t>(1, points.size() / 200000U);
    for (std::size_t index = 0; index < points.size();
         index += display_stride) {
        const ColoredPoint& point = points[index];
        side.at<cv::Vec3b>(sidePixel(point.x, point.z)) =
            cv::Vec3b(point.b, point.g, point.r);
    }
    for (const cv::Vec3d& position : camera_positions_world_m) {
        cv::circle(side, sidePixel(position[0], position[2]), 4,
                   cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
    }
    if (uav_position_world_m) {
        cv::drawMarker(side,
                       sidePixel((*uav_position_world_m)[0],
                                 (*uav_position_world_m)[2]),
                       cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 12, 2,
                       cv::LINE_AA);
    }
    drawAxes(side, false);

    cv::Mat overview(panel_height * 2, panel_width * 2, CV_8UC3);
    top_panel.copyTo(
        overview(cv::Rect(0, 0, panel_width, panel_height)));
    dem_panel.copyTo(
        overview(cv::Rect(panel_width, 0, panel_width, panel_height)));
    validity_panel.copyTo(
        overview(cv::Rect(0, panel_height, panel_width, panel_height)));
    side.copyTo(overview(cv::Rect(panel_width, panel_height,
                                  panel_width, panel_height)));

    const std::filesystem::path dem_preview_path =
        path.parent_path() / "dem_preview.png";
    if (!cv::imwrite(dem_preview_path.string(), dem_color) ||
        !cv::imwrite(path.string(), overview)) {
        throw std::runtime_error("Could not write debug visualization");
    }
}

}  // namespace metric_mapping

#ifdef HAVE_TWO_VIEW
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

#endif

#ifdef HAVE_VISUAL_FEATURES

namespace metric_mapping {
namespace {
cv::Mat scalarPreview(const cv::Mat& plane)
{
    if (plane.empty()) return {};
    double maximum = 0; cv::minMaxLoc(plane, nullptr, &maximum);
    cv::Mat gray, color;
    plane.convertTo(gray, CV_8U, maximum > 0 ? 255 / maximum : 0);
    cv::applyColorMap(gray, color, cv::COLORMAP_INFERNO);
    return color;
}
cv::Mat anglePreview(const cv::Mat& angle, const cv::Mat& valid, double period,
                     const cv::Mat& coherence = cv::Mat())
{
    if (angle.empty()) return {};
    cv::Mat hsv(angle.size(), CV_8UC3, cv::Scalar(0, 0, 0)), bgr;
    for (int y = 0; y < angle.rows; ++y) for (int x = 0; x < angle.cols; ++x) {
        if (!valid.at<std::uint8_t>(y,x)) continue;
        const float strength = coherence.empty() ? 1 : coherence.at<float>(y,x);
        hsv.at<cv::Vec3b>(y,x) = {std::uint8_t(std::clamp(int(179 * angle.at<float>(y,x) / period), 0, 179)),
                                 220, std::uint8_t(std::clamp(int(255 * strength), 0, 255))};
    }
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR); return bgr;
}
cv::Mat gradients(const VisualFeatureFrame& f)
{
    if (f.gradient_x.empty()) return {};
    cv::Mat halves[2]; int i = 0;
    for (const auto* plane : {&f.gradient_x, &f.gradient_y}) {
        double low, high; cv::minMaxLoc(*plane, &low, &high);
        const double scale = 127 / std::max({std::abs(low), std::abs(high), 1e-9});
        cv::Mat gray; plane->convertTo(gray, CV_8U, scale, 128);
        cv::applyColorMap(gray, halves[i++], cv::COLORMAP_JET);
    }
    cv::Mat result; cv::hconcat(halves, 2, result); return result;
}
} // namespace

cv::Mat renderVisualFeatures(const cv::Mat& original, const VisualFeatureFrame& f, const std::string& motion_note)
{
    if (original.type() != CV_8UC3 || original.empty() || f.bgr.empty()) throw std::runtime_error("Missing diagnostic image");
    constexpr int width = 400, height = 350, top = 96;
    cv::Mat canvas(top + 3 * height + 52, 4 * width, CV_8UC3, cv::Scalar(20, 24, 30));
    cv::putText(canvas, "IMAGINATION | Camera analytic pre-extraction", {24, 36}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {235,235,235}, 2, cv::LINE_AA);
    cv::putText(canvas, "GPS-denied UAV research | analytic inputs -> future convolution-assisted Transformer", {24, 64}, cv::FONT_HERSHEY_SIMPLEX, 0.52, {170,195,210}, 1, cv::LINE_AA);
    const auto panel = [&](int index, const std::string& title, const std::string& legend, const cv::Mat& image) {
        const int x = (index % 4) * width, y = top + (index / 4) * height;
        cv::rectangle(canvas, {x+5,y+5,width-10,height-10}, {45,50,58}, 1);
        cv::putText(canvas, title, {x+16,y+29}, cv::FONT_HERSHEY_SIMPLEX, 0.52, {240,240,240}, 1, cv::LINE_AA);
        cv::putText(canvas, legend, {x+16,y+height-19}, cv::FONT_HERSHEY_SIMPLEX, 0.36, {175,185,200}, 1, cv::LINE_AA);
        if (image.empty()) {
            cv::putText(canvas, "Disabled / unavailable", {x+55,y+165}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {150,170,190}, 1, cv::LINE_AA);
            return;
        }
        const double scale = std::min(double(width-32)/image.cols, double(height-82)/image.rows);
        cv::Mat resized, color;
        cv::resize(image, resized, {std::max(1, int(image.cols*scale)), std::max(1, int(image.rows*scale))}, 0, 0, cv::INTER_NEAREST);
        if (resized.channels() == 1) cv::cvtColor(resized, color, cv::COLOR_GRAY2BGR); else color = resized;
        color.copyTo(canvas(cv::Rect(x+(width-color.cols)/2, y+43+(height-82-color.rows)/2, color.cols, color.rows)));
    };
    panel(0, "01  Original RGB frame", "Input displayed in color; no invented calibration", original);
    panel(1, "02  Brightness gradients: Gx | Gy", "Blue negative / red positive; green zero", gradients(f));
    panel(2, "03  Edge magnitude", "sqrt(Gx^2 + Gy^2); display auto contrast", scalarPreview(f.edge_magnitude));
    panel(3, "04  Edge orientation", "Hue = normal angle [0,2pi); black = invalid", anglePreview(f.edge_orientation_rad, f.edge_orientation_valid, 2*CV_PI));
    cv::Mat image;
    if (f.computed[std::size_t(VisualFeature::Corners)]) {
        image = f.bgr.clone(); for (const auto& p : f.corners) cv::circle(image, p.pt, 2, {0,255,255}, 1);
    }
    panel(4, "05  Corners: " + std::to_string(f.corners.size()), "Minimum tensor eigenvalue; grid + spacing limit", image);
    image.release();
    if (f.computed[std::size_t(VisualFeature::Keypoints)]) {
        image = f.bgr.clone(); for (const auto& p : f.keypoints) cv::drawMarker(image, p.pt, {70,255,80}, cv::MARKER_CROSS, 5, 1);
    }
    panel(5, "06  Local FAST keypoints: " + std::to_string(f.keypoints.size()), "No descriptors; grid + spacing limit", image);
    panel(6, "07  Color transitions", "Opponent RG/BY gradients; display auto contrast", scalarPreview(f.color_transition_magnitude));
    panel(7, "08  Texture orientation", "Hue = tangent [0,pi); value = coherence", anglePreview(f.texture_orientation_rad, f.texture_orientation_valid, CV_PI, f.texture_coherence));
    panel(8, "09  Local binary texture (LBP8)", "Clockwise bits from NW; code, not intensity", f.local_binary_texture);
    image.release();
    if (f.computed[std::size_t(VisualFeature::Contours)]) {
        image = f.bgr.clone(); cv::drawContours(image, f.contours, -1, {60,255,255}, 1);
    }
    panel(9, "10  Contours: " + std::to_string(f.contours.size()), "Canny traces; image evidence, not objects", image);
    panel(10, "11  Shape boundaries", "Simplified retained traces; not semantic masks", f.shape_boundaries);
    image.release();
    if (f.motion && f.motion->valid) {
        image = f.bgr.clone(); for (const auto& t : f.tracks) if (t.age > 1)
            cv::arrowedLine(image, t.previous_px, t.current_px, {0,255,0}, 1);
    }
    panel(11, "12  Motion / optical flow", f.motion ? f.motion_status : "No verified sequential/calibrated input", image);
    const std::string note = motion_note.empty() ? "Maps use working-image pixels; visual contrast scaling does not change programmatic data." : motion_note;
    cv::putText(canvas, note, {24,canvas.rows-23}, cv::FONT_HERSHEY_SIMPLEX, 0.48, {175,195,210}, 1, cv::LINE_AA);
    return canvas;
}
} // namespace metric_mapping

#endif

#ifdef HAVE_VISUAL_FEATURES
namespace metric_mapping {
cv::Mat renderAnalyticalFeatures(const VisualFeatureFrame& f)
{
    const auto& tensor=f.feature_tensor;
    const int columns=6,width=224,height=250,top=72;
    const int panels=int(tensor.channel_names.size())+2,rows=(panels+columns-1)/columns;
    cv::Mat canvas(top+rows*height,columns*width,CV_8UC3,cv::Scalar(22,24,28));
    cv::putText(canvas,"IMAGINATION | Analytical spatial tensor",{18,28},cv::FONT_HERSHEY_SIMPLEX,0.7,{240,240,240},1,cv::LINE_AA);
    cv::putText(canvas,"Current-view maps | fixed display ranges | magenta: unavailable | no semantic decisions",{18,53},cv::FONT_HERSHEY_SIMPLEX,0.43,{185,200,210},1,cv::LINE_AA);
    const auto panel=[&](int index,const std::string& title,const cv::Mat& image){
        const int x=(index%columns)*width,y=top+(index/columns)*height;
        cv::putText(canvas,title,{x+8,y+19},cv::FONT_HERSHEY_SIMPLEX,0.40,{240,240,240},1,cv::LINE_AA);
        cv::Mat resized;cv::resize(image,resized,{208,208},0,0,cv::INTER_NEAREST);resized.copyTo(canvas(cv::Rect(x+8,y+30,208,208)));
    };
    panel(0,"Current prepared RGB",f.bgr);
    cv::Mat tracked=f.bgr.clone();for(const auto& t:f.tracks)if(t.age>1)cv::arrowedLine(tracked,t.previous_px,t.current_px,{0,255,0},1);
    for(std::size_t i=0;i<f.pose_correspondences.size();++i)if(f.relative_pose.inliers[i])
        cv::circle(tracked,f.pose_correspondences[i].current_px,2,{0,255,255},1);
    panel(1,"LK green / pose inliers yellow",tracked);
    for(std::size_t i=0;i<tensor.channel_names.size();++i){
        const auto& name=tensor.channel_names[i];auto p=tensor.plane(i),mask=tensor.mask(i);
        const bool signed_value=name=="Gx"||name=="Gy"||name=="HarrisResponse"||name=="OpticalFlowU"||name=="OpticalFlowV"||name=="DepthGradientX"||name=="DepthGradientY";
        cv::Mat gray,color;p.convertTo(gray,CV_8U,signed_value?127.5:255,signed_value?127.5:0);
        cv::applyColorMap(gray,color,signed_value?cv::COLORMAP_JET:cv::COLORMAP_VIRIDIS);
        color.setTo(cv::Scalar(160,0,160),mask==0);panel(int(i)+2,name,color);
    }
    return canvas;
}
} // namespace metric_mapping
#endif
