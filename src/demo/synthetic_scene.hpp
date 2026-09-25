#pragma once
#include "metric_mapping/types.hpp"

namespace metric_mapping::app {
// Appearance-derived geometry for exercising exports, never metric stereo.
struct SyntheticScene {
    std::vector<ColoredPoint> points;
    cv::Mat matches_image;
    std::size_t alignment_matches = 0;
};
SyntheticScene makeSyntheticScene(const std::filesystem::path& image_1_path,
                                  const std::filesystem::path& image_2_path);
} // namespace metric_mapping::app
