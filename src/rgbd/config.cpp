#include "metric_mapping/config.hpp"

#include "metric_mapping/geometry.hpp"
#include "metric_mapping/ultrasonic.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace metric_mapping {
namespace {

cv::FileNode requireNode(const cv::FileNode& parent,
                         const std::string& name,
                         const std::string& context)
{
    const cv::FileNode node = parent[name];
    if (node.empty()) {
        throw std::runtime_error("Missing configuration value: " + context +
                                 "." + name);
    }
    return node;
}

double readNumber(const cv::FileNode& node, const std::string& context)
{
    if (!node.isInt() && !node.isReal())
        throw std::runtime_error("Configuration value must be numeric: " + context);
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) {
        throw std::runtime_error("Configuration value must be finite: " +
                                 context);
    }
    return value;
}

double readFiniteDouble(const cv::FileNode& parent,
                        const std::string& name,
                        const std::string& context)
{
    return readNumber(requireNode(parent, name, context), context + "." + name);
}

int readInt(const cv::FileNode& parent,
            const std::string& name,
            const std::string& context)
{
    const double value = readFiniteDouble(parent, name, context);
    if (value != std::trunc(value) ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max())
        throw std::runtime_error("Configuration value must be an in-range integer: " +
                                 context + "." + name);
    return static_cast<int>(value);
}

std::size_t readSize(const cv::FileNode& parent,
                     const std::string& name,
                     const std::string& context)
{
    const double value = readFiniteDouble(parent, name, context);
    // Use the exclusive power-of-two limit: SIZE_MAX may round up as a double.
    if (value < 1.0 || value != std::trunc(value) ||
        value >= std::ldexp(1.0, std::numeric_limits<std::size_t>::digits)) {
        throw std::runtime_error("Configuration value must be an in-range positive integer: " +
                                 context + "." + name);
    }
    return static_cast<std::size_t>(value);
}

std::string readString(const cv::FileNode& parent,
                       const std::string& name,
                       const std::string& context)
{
    const cv::FileNode node = requireNode(parent, name, context);
    if (!node.isString()) {
        throw std::runtime_error("Configuration value must be a string: " +
                                 context + "." + name);
    }
    return static_cast<std::string>(node);
}

bool readExplicitBool(const cv::FileNode& parent,
                      const std::string& name,
                      const std::string& context)
{
    const int value = readInt(parent, name, context);
    if (value != 0 && value != 1) {
        throw std::runtime_error("Configuration boolean must be 0 or 1: " +
                                 context + "." + name);
    }
    return value == 1;
}

cv::Matx44d readPose(const cv::FileNode& node, const std::string& context)
{
    if (!node.isSeq() || node.size() != 16) {
        throw std::runtime_error(context +
                                 " must contain exactly 16 row-major values");
    }

    cv::Matx44d transform = cv::Matx44d::zeros();
    int index = 0;
    for (const cv::FileNode& value_node : node) {
        const double value = readNumber(value_node, context);
        transform(index / 4, index % 4) = value;
        ++index;
    }
    return transform;
}

cv::Vec3d readVec3(const cv::FileNode& node, const std::string& context)
{
    if (!node.isSeq() || node.size() != 3) {
        throw std::runtime_error(context + " must contain exactly 3 values");
    }

    cv::Vec3d value;
    int index = 0;
    for (const cv::FileNode& component_node : node) {
        const double component = readNumber(component_node, context);
        value[index++] = component;
    }
    return value;
}

UltrasonicMeasurement readUltrasonic(const cv::FileNode& node)
{
    if (!node.isMap())
        throw std::runtime_error("ultrasonic must be a mapping");
    UltrasonicMeasurement measurement;
    measurement.distance_m = readFiniteDouble(node, "distance_m", "ultrasonic");
    measurement.maximum_error_m = readFiniteDouble(node, "maximum_error_m", "ultrasonic");
    measurement.sensor_offset_world_m = readVec3(
        requireNode(node, "sensor_offset_world_m", "ultrasonic"),
        "ultrasonic.sensor_offset_world_m");
    if (readString(node, "direction", "ultrasonic") != "world_down")
        throw std::runtime_error("Ultrasonic direction must be world_down (vertical -Z)");
    validateUltrasonicMeasurement(measurement);
    return measurement;
}

std::filesystem::path resolvePath(const std::filesystem::path& base,
                                  const std::string& configured_path)
{
    const std::filesystem::path path(configured_path);
    return (path.is_absolute() ? path : base / path).lexically_normal();
}

}  // namespace

std::string depthUnitName(DepthUnit unit)
{
    return unit == DepthUnit::Meters ? "meters" : "millimeters";
}

std::string poseConventionName(PoseConvention convention)
{
    return convention == PoseConvention::CameraToWorld
               ? "camera_to_world"
               : "world_to_camera";
}

UltrasonicMeasurement loadUltrasonicConfig(const std::filesystem::path& config_path)
{
    cv::FileStorage storage(config_path.string(), cv::FileStorage::READ);
    if (!storage.isOpened())
        throw std::runtime_error("Could not load ultrasonic configuration: " + config_path.string());
    return readUltrasonic(requireNode(storage.root(), "ultrasonic", "config"));
}

ReconstructionConfig loadConfig(const std::filesystem::path& config_path)
{
    const std::filesystem::path absolute_config =
        std::filesystem::absolute(config_path).lexically_normal();
    cv::FileStorage storage(absolute_config.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("Could not open configuration: " +
                                 absolute_config.string());
    }

    ReconstructionConfig config;
    config.config_path = absolute_config;
    const std::filesystem::path base = absolute_config.parent_path();
    const cv::FileNode root = storage.root();

    config.world_frame = readString(root, "world_frame", "root");
    if (config.world_frame != "map_z_up") {
        throw std::runtime_error(
            "world_frame must be exactly 'map_z_up'; poses must map OpenCV "
            "camera coordinates (+X right, +Y down, +Z forward) into a "
            "gravity-aligned world frame with +Z up");
    }

    config.images_are_rectified =
        readExplicitBool(root, "images_are_rectified", "root");
    config.depth_registered_to_rgb =
        readExplicitBool(root, "depth_registered_to_rgb", "root");
    if (!config.images_are_rectified || !config.depth_registered_to_rgb) {
        throw std::runtime_error(
            "This pinhole front end requires rectified RGB images and depth "
            "registered to the RGB pixels");
    }

    const std::string pose_convention =
        readString(root, "pose_convention", "root");
    if (pose_convention == "camera_to_world") {
        config.pose_convention = PoseConvention::CameraToWorld;
    } else if (pose_convention == "world_to_camera") {
        config.pose_convention = PoseConvention::WorldToCamera;
    } else {
        throw std::runtime_error(
            "pose_convention must be 'camera_to_world' or 'world_to_camera'");
    }

    const cv::FileNode camera = requireNode(root, "camera", "root");
    config.camera.width = readInt(camera, "width", "camera");
    config.camera.height = readInt(camera, "height", "camera");
    config.camera.fx = readFiniteDouble(camera, "fx", "camera");
    config.camera.fy = readFiniteDouble(camera, "fy", "camera");
    config.camera.cx = readFiniteDouble(camera, "cx", "camera");
    config.camera.cy = readFiniteDouble(camera, "cy", "camera");
    validateCamera(config.camera);

    const cv::FileNode depth = requireNode(root, "depth", "root");
    const std::string depth_unit = readString(depth, "unit", "depth");
    if (depth_unit == "meters") {
        config.depth.unit = DepthUnit::Meters;
    } else if (depth_unit == "millimeters") {
        config.depth.unit = DepthUnit::Millimeters;
    } else {
        throw std::runtime_error(
            "depth.unit must be exactly 'meters' or 'millimeters'");
    }
    config.depth.min_depth_m =
        readFiniteDouble(depth, "min_depth_m", "depth");
    config.depth.max_depth_m =
        readFiniteDouble(depth, "max_depth_m", "depth");
    config.depth.pixel_stride = readInt(depth, "pixel_stride", "depth");

    const cv::FileNode invalid_values =
        requireNode(depth, "invalid_values", "depth");
    if (!invalid_values.isSeq() || invalid_values.empty()) {
        throw std::runtime_error(
            "depth.invalid_values must explicitly list sensor-invalid raw "
            "values, for example [0, 65535]");
    }
    for (const cv::FileNode& value_node : invalid_values) {
        config.depth.invalid_values.push_back(
            readNumber(value_node, "depth.invalid_values"));
    }

    if (!(config.depth.min_depth_m > 0.0) ||
        !(config.depth.max_depth_m > config.depth.min_depth_m) ||
        config.depth.pixel_stride < 1) {
        throw std::runtime_error(
            "Depth limits must satisfy 0 < min_depth_m < max_depth_m and "
            "pixel_stride must be at least 1");
    }

    const cv::FileNode fusion = requireNode(root, "fusion", "root");
    config.fusion.voxel_size_m =
        readFiniteDouble(fusion, "voxel_size_m", "fusion");
    config.fusion.max_raw_points =
        readSize(fusion, "max_raw_points", "fusion");
    config.fusion.max_voxels = readSize(fusion, "max_voxels", "fusion");
    if (!(config.fusion.voxel_size_m > 0.0)) {
        throw std::runtime_error("fusion.voxel_size_m must be positive");
    }

    const cv::FileNode map = requireNode(root, "map", "root");
    config.map.resolution_m =
        readFiniteDouble(map, "resolution_m", "map");
    config.map.max_cells = readSize(map, "max_cells", "map");
    if (!(config.map.resolution_m > 0.0)) {
        throw std::runtime_error("map.resolution_m must be positive");
    }

    const cv::FileNode idw = requireNode(map, "idw", "map");
    config.map.idw.enabled = readExplicitBool(idw, "enabled", "map.idw");
    config.map.idw.search_radius_m =
        readFiniteDouble(idw, "search_radius_m", "map.idw");
    config.map.idw.minimum_neighbors =
        readInt(idw, "minimum_neighbors", "map.idw");
    config.map.idw.power = readFiniteDouble(idw, "power", "map.idw");
    config.map.idw.maximum_interpolation_distance_m = readFiniteDouble(
        idw, "maximum_interpolation_distance_m", "map.idw");
    if (config.map.idw.enabled &&
        (!(config.map.idw.search_radius_m > 0.0) ||
         config.map.idw.minimum_neighbors < 1 ||
         !(config.map.idw.power > 0.0) ||
         !(config.map.idw.maximum_interpolation_distance_m > 0.0))) {
        throw std::runtime_error(
            "Enabled IDW requires positive radii and power, and at least one "
            "neighbor");
    }

    config.output_directory = resolvePath(
        base, readString(root, "output_directory", "root"));

    const cv::FileNode uav_position = root["uav_position_world_m"];
    if (!uav_position.empty()) {
        config.uav_position_world_m =
            readVec3(uav_position, "uav_position_world_m");
    }
    if (!root["ultrasonic"].empty()) {
        config.ultrasonic = readUltrasonic(root["ultrasonic"]);
        if (!config.uav_position_world_m)
            throw std::runtime_error("ultrasonic requires uav_position_world_m");
    }

    const cv::FileNode frames = requireNode(root, "frames", "root");
    if (!frames.isSeq() || frames.empty()) {
        throw std::runtime_error(
            "frames must contain at least one RGB/depth/pose entry; no "
            "camera pose or metric depth will be invented");
    }

    int frame_index = 0;
    for (const cv::FileNode& frame_node : frames) {
        const std::string context =
            "frames[" + std::to_string(frame_index) + "]";
        FrameSpec frame;
        frame.rgb_path = resolvePath(
            base, readString(frame_node, "rgb", context));
        frame.depth_path = resolvePath(
            base, readString(frame_node, "depth", context));
        frame.supplied_pose = readPose(
            requireNode(frame_node, "pose", context), context + ".pose");

        if (!std::filesystem::is_regular_file(frame.rgb_path)) {
            throw std::runtime_error("RGB image not found: " +
                                     frame.rgb_path.string());
        }
        if (!std::filesystem::is_regular_file(frame.depth_path)) {
            throw std::runtime_error("Depth image not found: " +
                                     frame.depth_path.string());
        }

        config.frames.push_back(frame);
        ++frame_index;
    }

    return config;
}

}  // namespace metric_mapping
