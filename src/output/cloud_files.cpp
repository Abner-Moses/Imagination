#include "metric_mapping/io.hpp"
#include "output_internal.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
