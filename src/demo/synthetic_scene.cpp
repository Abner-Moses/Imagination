#include "synthetic_scene.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace metric_mapping::app {
SyntheticScene makeSyntheticScene(const std::filesystem::path& image_1_path,
                                  const std::filesystem::path& image_2_path)
{
    using namespace metric_mapping;
    const cv::Mat image_1 = cv::imread(image_1_path.string(),
                                       cv::IMREAD_COLOR);
    const cv::Mat image_2 = cv::imread(image_2_path.string(),
                                       cv::IMREAD_COLOR);
    if (image_1.empty() || image_2.empty())
        throw std::runtime_error("Could not load both demo images");
    if (image_1.size() != image_2.size())
        throw std::runtime_error("Demo images must have the same size");

    cv::Mat gray_1;
    cv::Mat gray_2;
    cv::cvtColor(image_1, gray_1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_2, gray_2, cv::COLOR_BGR2GRAY);
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    orb->detectAndCompute(gray_1, cv::noArray(), keypoints_1,
                          descriptors_1);
    orb->detectAndCompute(gray_2, cv::noArray(), keypoints_2,
                          descriptors_2);
    if (descriptors_1.empty() || descriptors_2.empty())
        throw std::runtime_error("Demo alignment found no descriptors");

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> neighbors;
    matcher.knnMatch(descriptors_1, descriptors_2, neighbors, 2);
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pixels_1;
    std::vector<cv::Point2f> pixels_2;
    for (const auto& pair : neighbors) {
        if (pair.size() == 2 && pair[0].distance < 0.75F * pair[1].distance) {
            good_matches.push_back(pair[0]);
            pixels_1.push_back(keypoints_1[pair[0].queryIdx].pt);
            pixels_2.push_back(keypoints_2[pair[0].trainIdx].pt);
        }
    }
    if (good_matches.size() < 8U)
        throw std::runtime_error("Too few demo image matches");

    cv::Mat homography_mask;
    const cv::Mat image_2_to_image_1 = cv::findHomography(
        pixels_2, pixels_1, cv::RANSAC, 3.0, homography_mask);
    if (image_2_to_image_1.empty())
        throw std::runtime_error("Could not align the two demo images");

    cv::Mat aligned_2;
    cv::warpPerspective(image_2, aligned_2, image_2_to_image_1,
                        image_1.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);
    cv::Mat source_mask(image_1.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat aligned_mask;
    cv::warpPerspective(source_mask, aligned_mask, image_2_to_image_1,
                        image_1.size(), cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT);
    cv::Mat average;
    cv::addWeighted(image_1, 0.5, aligned_2, 0.5, 0.0, average);
    cv::Mat blended = image_1.clone();
    average.copyTo(blended, aligned_mask);

    std::vector<cv::DMatch> inlier_matches;
    for (std::size_t index = 0; index < good_matches.size(); ++index) {
        if (homography_mask.at<std::uint8_t>(static_cast<int>(index)) != 0)
            inlier_matches.push_back(good_matches[index]);
    }
    cv::Mat matches_image;
    cv::drawMatches(image_1, keypoints_1, image_2, keypoints_2,
                    inlier_matches, matches_image, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::Mat gray_float;
    cv::cvtColor(blended, gray_float, cv::COLOR_BGR2GRAY);
    gray_float.convertTo(gray_float, CV_32F, 1.0 / 255.0);
    cv::Mat broad;
    cv::Mat local;
    cv::GaussianBlur(gray_float, broad, cv::Size(0, 0), 14.0);
    cv::GaussianBlur(gray_float, local, cv::Size(0, 0), 2.0);
    const float broad_mean = static_cast<float>(cv::mean(broad)[0]);

    constexpr int pixel_stride = 1;
    constexpr double xy_resolution_m = 0.05;
    std::vector<ColoredPoint> synthetic_points;
    synthetic_points.reserve(image_1.total() /
                             (pixel_stride * pixel_stride));
    for (int row = 0; row < image_1.rows; row += pixel_stride) {
        for (int column = 0; column < image_1.cols;
             column += pixel_stride) {
            const float low_frequency = broad.at<float>(row, column) -
                                        broad_mean;
            const float local_contrast = std::abs(
                gray_float.at<float>(row, column) -
                local.at<float>(row, column));
            const float synthetic_height = std::clamp(
                0.35F + 0.8F * low_frequency +
                    2.5F * local_contrast,
                0.0F, 1.5F);
            const cv::Vec3b bgr = blended.at<cv::Vec3b>(row, column);
            synthetic_points.push_back({
                static_cast<float>((column - 0.5 * image_1.cols) *
                                   xy_resolution_m),
                static_cast<float>((0.5 * image_1.rows - row) *
                                   xy_resolution_m),
                synthetic_height, bgr[2], bgr[1], bgr[0]});
        }
    }

    return {std::move(synthetic_points), matches_image, inlier_matches.size()};
}
} // namespace metric_mapping::app
