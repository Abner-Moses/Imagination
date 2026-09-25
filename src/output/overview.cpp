#include "metric_mapping/visualization.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
