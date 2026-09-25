#include "output_internal.hpp"
#include "metric_mapping/visualization.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
