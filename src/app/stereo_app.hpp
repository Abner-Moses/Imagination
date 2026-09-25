#pragma once
#include "metric_mapping/two_view.hpp"

namespace metric_mapping::app {
void runStereo(const std::filesystem::path& image_1_path,
               const std::filesystem::path& image_2_path,
               const CameraIntrinsics& camera, double baseline_m);
void runSyntheticDemo(const std::filesystem::path& image_1_path,
                      const std::filesystem::path& image_2_path);
void printLandingSummary(const LandingAnalysis& analysis);
} // namespace metric_mapping::app
