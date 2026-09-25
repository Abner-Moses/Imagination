#pragma once

#include "metric_mapping/types.hpp"

#include <filesystem>
#include <string>

namespace metric_mapping {

ReconstructionConfig loadConfig(const std::filesystem::path& config_path);
std::string depthUnitName(DepthUnit unit);
std::string poseConventionName(PoseConvention convention);

}  // namespace metric_mapping
