#include "test_support.hpp"
#include "metric_mapping/config.hpp"
#include <opencv2/imgcodecs.hpp>
#include <fstream>

namespace metric_mapping::tests {
namespace {
void testStrictConfigParsing()
{
    TemporaryDirectory directory;
    cv::imwrite((directory.path / "image.png").string(),
                cv::Mat(1, 1, CV_16U, cv::Scalar(1000)));
    const std::string valid = R"(%YAML:1.0
---
world_frame: "map_z_up"
images_are_rectified: 1
depth_registered_to_rgb: 1
pose_convention: "camera_to_world"
output_directory: "output"
camera: {width: 1, height: 1, fx: 2.0, fy: 2.0, cx: 0.0, cy: 0.0}
depth: {unit: "millimeters", min_depth_m: 0.1, max_depth_m: 100.0, pixel_stride: 1, invalid_values: [0, 65535]}
fusion: {voxel_size_m: 0.1, max_raw_points: 100, max_voxels: 100}
map:
  resolution_m: 0.1
  max_cells: 100
  idw: {enabled: 1, search_radius_m: 0.3, minimum_neighbors: 3, power: 2.0, maximum_interpolation_distance_m: 0.3}
uav_position_world_m: [0.0, 0.0, 10.0]
frames:
  - rgb: "image.png"
    depth: "image.png"
    pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
)";
    const auto path = directory.path / "config.yaml";
    const auto read = [&](const std::string& yaml) {
        { std::ofstream output(path); output << yaml; }
        return loadConfig(path);
    };
    requireNear(read(valid).camera.fx, 2.0, 0.0, "Valid configuration must load");
    for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
             {"fx: 2.0", "fx: \"not calibrated\""},
             {"fx: 2.0", "fx: .Inf"},
             {"images_are_rectified: 1", "images_are_rectified: 0.6"},
             {"width: 1", "width: 1.4"},
             {"width: 1", "width: 1.0e100"},
             {"pixel_stride: 1", "pixel_stride: 0"},
             {"max_raw_points: 100", "max_raw_points: 1.0e100"},
             {"max_raw_points: 100", "max_raw_points: 18446744073709551616.0"},
             {"max_cells: 100", "max_cells: 2.5"},
             {"[0, 65535]", "[0, \"invalid\"]"},
             {"[0.0, 0.0, 10.0]", "[0.0, \"invalid\", 10.0]"},
             {"pose: [1, 0", "pose: [\"invalid\", 0"}}) {
        std::string yaml = valid;
        yaml.replace(yaml.find(replacement.first), replacement.first.size(), replacement.second);
        requireThrows([&] { read(yaml); }, "Must reject " + replacement.second);
    }
}
} // namespace

std::vector<TestCase> configTests()
{
    return {
        {"strict configuration parsing", testStrictConfigParsing}
    };
}
} // namespace metric_mapping::tests
