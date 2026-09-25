#pragma once
#include "metric_mapping/types.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace metric_mapping::tests {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path(std::filesystem::temp_directory_path() /
               ("mapping_regression_" + std::to_string(
                   std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(path);
    }
    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::filesystem::path path;
};

inline DepthConfig metricDepthConfig()
{
    DepthConfig config;
    config.unit = DepthUnit::Meters;
    config.min_depth_m = 0.1;
    config.max_depth_m = 100.0;
    config.pixel_stride = 1;
    config.invalid_values = {0.0};
    return config;
}

inline void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

inline void requireNear(double actual,
                 double expected,
                 double tolerance,
                 const std::string& message)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", received " + std::to_string(actual));
    }
}

inline TerrainGrid makeFlatGrid(int width, int height, double resolution_m)
{
    TerrainGrid grid;
    grid.width = width;
    grid.height = height;
    grid.resolution_m = resolution_m;
    grid.minimum_x_m = -0.5 * (width - 1) * resolution_m;
    grid.maximum_y_m = 0.5 * (height - 1) * resolution_m;
    const std::size_t cells =
        static_cast<std::size_t>(width) * height;
    grid.elevation_m.assign(cells, 0.0F);
    grid.color_bgr.assign(cells, cv::Vec3b(60, 90, 120));
    grid.validity.assign(cells, 255);
    return grid;
}

inline void requireThrows(const std::function<void()>& action, const std::string& message)
{
    try { action(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

using TestCase = std::pair<std::string, std::function<void()>>;
std::vector<TestCase> rgbdTests();
std::vector<TestCase> terrainTests();
std::vector<TestCase> configTests();
std::vector<TestCase> outputTests();
#ifdef HAVE_TWO_VIEW
std::vector<TestCase> stereoTests();
#endif
} // namespace metric_mapping::tests
