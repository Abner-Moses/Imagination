#pragma once
#include "metric_mapping/types.hpp"
#include <fstream>

namespace metric_mapping::detail {
void requireWritable(const std::ofstream& stream, const std::filesystem::path& path);
cv::Mat terrainValidMask(const TerrainGrid& grid);
cv::Mat colorizeFloat(const cv::Mat& values, const cv::Mat& valid_mask,
                      double minimum, double maximum);
cv::Mat colorizeDem(const cv::Mat& dem, const cv::Mat& valid_mask);
void writeImage(const std::filesystem::path& path, const cv::Mat& image);
cv::Mat labeledPanel(const cv::Mat& image, const std::string& title);
void writeLandingJson(const std::filesystem::path& output_directory,
                      const LandingAnalysis& analysis, const LandingAnalysisConfig& config,
                      const cv::Vec3d& uav_position_world_m, bool demo_only);
} // namespace metric_mapping::detail
