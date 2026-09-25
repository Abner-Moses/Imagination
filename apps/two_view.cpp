#include "../src/app/stereo_app.hpp"
#include "metric_mapping/config.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
double parseNumber(const char* text, const std::string& name)
{
    std::size_t consumed = 0;
    const std::string value(text);
    double number = 0.0;
    try {
        number = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(name + " must be a number, not '" + value +
                                 "'");
    }
    if (consumed != value.size() || !std::isfinite(number))
        throw std::runtime_error(name + " must be a finite number");
    return number;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--demo") {
        try {
            metric_mapping::app::runSyntheticDemo(argv[2], argv[3]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 8 && argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " image1 image2 fx fy cx cy baseline_m [--ultrasonic sensor.yaml]\n"
                  << "   or: " << argv[0]
                  << " --demo image1 image2\n";
        return 1;
    }

    try {
        std::optional<metric_mapping::UltrasonicMeasurement> ultrasonic;
        if (argc == 10) {
            if (std::string(argv[8]) != "--ultrasonic")
                throw std::runtime_error("Expected --ultrasonic sensor.yaml");
            ultrasonic = metric_mapping::loadUltrasonicConfig(argv[9]);
        }
        const std::filesystem::path image_1_path(argv[1]);
        const std::filesystem::path image_2_path(argv[2]);
        const cv::Mat probe = cv::imread(image_1_path.string(),
                                         cv::IMREAD_COLOR);
        if (probe.empty())
            throw std::runtime_error("Could not load " +
                                     image_1_path.string());

        metric_mapping::CameraIntrinsics camera;
        camera.width = probe.cols;
        camera.height = probe.rows;
        camera.fx = parseNumber(argv[3], "fx");
        camera.fy = parseNumber(argv[4], "fy");
        camera.cx = parseNumber(argv[5], "cx");
        camera.cy = parseNumber(argv[6], "cy");
        const double baseline_m = parseNumber(argv[7], "baseline_m");

        metric_mapping::app::runStereo(image_1_path, image_2_path, camera, baseline_m, ultrasonic);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
