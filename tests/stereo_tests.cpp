#include "test_support.hpp"
#include "metric_mapping/two_view.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace metric_mapping::tests {
namespace {
void testDenseTwoViewStereo(int foreground_disparity,
                            double yaw_degrees = 0.0, double fy = 200.0)
{
    TemporaryDirectory temporary;
    const auto& directory = temporary.path;

    constexpr int width = 320;
    constexpr int height = 240;
    cv::Mat first(height, width, CV_8UC3);
    cv::RNG random(123456U);
    random.fill(first, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(first, first, cv::Size(3, 3), 0.5);

    cv::Mat second(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const bool foreground = column >= 105 && column < 215 &&
                                    row >= 70 && row < 170;
            const int disparity = foreground ? foreground_disparity : 20;
            const int second_column = column - disparity;
            if (second_column >= 0 && second_column < width) {
                second.at<cv::Vec3b>(row, second_column) =
                    first.at<cv::Vec3b>(row, column);
            }
        }
    }

    const CameraIntrinsics camera{width, height, 200.0, fy, 159.5, 119.5};
    if (yaw_degrees != 0.0) {
        const double angle = yaw_degrees * CV_PI / 180.0;
        const cv::Matx33d rotation(std::cos(angle), -std::sin(angle), 0,
                                   std::sin(angle), std::cos(angle), 0, 0, 0, 1);
        const cv::Matx33d intrinsic(camera.fx, 0, camera.cx,
                                    0, camera.fy, camera.cy, 0, 0, 1);
        cv::Mat rotated;
        cv::warpPerspective(second, rotated, intrinsic * rotation * intrinsic.inv(), second.size());
        second = rotated;
    }

    const std::filesystem::path first_path = directory / "first.png";
    const std::filesystem::path second_path = directory / "second.png";
    require(cv::imwrite(first_path.string(), first) &&
                cv::imwrite(second_path.string(), second),
            "Synthetic stereo images must be writable");

    const TwoViewResult result = reconstructTwoView(
        first_path, second_path, camera, 1.0);
    require(result.points.size() > 500,
            "Known stereo pair must create a dense cloud");

    std::vector<double> depths;
    depths.reserve(result.points.size());
    for (const ColoredPoint& point : result.points)
        depths.push_back(-point.z);
    std::sort(depths.begin(), depths.end());
    const double median_depth = depths[depths.size() / 2];
    requireNear(median_depth, 10.0, 1.0,
                "Known 20 px disparity must reconstruct near 10 m depth");
    std::size_t foreground_points = 0;
    std::size_t correct_foreground_points = 0;
    for (const auto& point : result.points) {
        const double z = -point.z;
        const double u = point.x * camera.fx / z + camera.cx;
        const double v = -point.y * camera.fy / z + camera.cy;
        if (u > 125 && u < 160 && v > 90 && v < 150) {
            ++foreground_points;
            if (std::abs(z - 200.0 / foreground_disparity) < 0.5)
                ++correct_foreground_points;
        }
    }
    require(foreground_points > 1000 &&
            correct_foreground_points > 0.95 * foreground_points,
            "Raised foreground must retain its true depth, not background depth");
}
} // namespace
std::vector<TestCase> stereoTests()
{
    return {{"dense calibrated two-view stereo", [] {
        for (int disparity : {25, 35, 40, 60})
            testDenseTwoViewStereo(disparity);
        testDenseTwoViewStereo(35, 3.0, 150.0);
    }}};
}
} // namespace metric_mapping::tests
