#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

// --- Config ---

namespace metric_mapping::detail {
#ifdef HAVE_MOTION
std::optional<ImuSample> readImuSnapshot(const std::filesystem::path& path)
{
    if (path.empty()) return std::nullopt;
    std::ifstream file(path);
    ImuSample sample;
    std::string extra;
    if (!(file >> sample.timestamp_s >> sample.roll_rad >> sample.pitch_rad >> sample.yaw_rad) ||
        file >> extra) return std::nullopt;
    return sample; // Frame-time freshness, attitude and geometry checks live in motion.cpp.
}

std::optional<AltitudeSample> readAltitudeSnapshot(const std::filesystem::path& path)
{
    if (path.empty()) return std::nullopt;
    std::ifstream file(path);
    double timestamp, range, offset_z;
    std::string extra;
    if (!(file >> timestamp >> range >> offset_z) || file >> extra) return std::nullopt;
    try { return altitudeFromUltrasonic({range, 0.2, {0, 0, offset_z}}, timestamp); }
    catch (const std::exception&) { return std::nullopt; }
}
#endif

int parseInteger(const std::string& text, int minimum, int maximum)
{
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size() || value < minimum || value > maximum)
        throw std::runtime_error("Expected integer in [" + std::to_string(minimum) +
                                 "," + std::to_string(maximum) + "]: " + text);
    return value;
}

TimingSummary summarizeTimes(std::vector<double>& times)
{
    if (times.empty()) throw std::runtime_error("Timing summary needs at least one sample");
    for (double time : times)
        if (!std::isfinite(time) || time < 0) throw std::runtime_error("Invalid timing sample");
    std::sort(times.begin(), times.end());
    return {std::accumulate(times.begin(), times.end(), 0.0) / times.size(),
            times[times.size() / 2], // Preserve the benchmark's upper-median convention.
            times[std::size_t(std::ceil(0.95 * times.size())) - 1], times.back()};
}
} // namespace metric_mapping::detail

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

#ifdef HAVE_VISUAL_FEATURES
#include <sstream>
namespace metric_mapping {
const char* analyticFeatureName(AnalyticFeature f)
{
    static const char* names[]={"appearance","gradients","hog","harris","canny","contours","chroma","flow",
        "depth","depth_gradients","slope","roughness","geometry_confidence"};
    if(std::size_t(f)>=analytic_feature_count)throw std::runtime_error("Invalid analytical family");
    return names[std::size_t(f)];
}
AnalyticSelection selectAnalyticFeatures(const std::string& names)
{
    if(names=="all")return AnalyticSelection{}.set();
    if(names=="none")return {};
    if(names.empty()||names.back()==',')throw std::runtime_error("Empty analytical family");
    AnalyticSelection selected;std::istringstream stream(names);std::string name;
    while(std::getline(stream,name,',')){
        std::size_t i=0;for(;i<analytic_feature_count;++i)if(name==analyticFeatureName(AnalyticFeature(i)))break;
        if(i==analytic_feature_count)throw std::runtime_error("Unknown analytical family: "+name);
        selected.set(i);
    }return selected;
}
VisualFeatureSettings analyticalFeatureSettings()
{
    VisualFeatureSettings s;s.features.reset();s.analytical.enabled=true;s.profiling=true;
    s.motion.triangulation=true;s.motion.forward_backward=true;s.motion.profiling=true;
    return s;
}
std::optional<CameraIntrinsics> loadVisualCalibration(const std::filesystem::path& path,std::vector<double>& distortion)
{
    if(path.empty())return std::nullopt;
    cv::FileStorage file(path.string(),cv::FileStorage::READ);
    if(!file.isOpened())throw std::runtime_error("Cannot open calibration: "+path.string());
    const auto node=file["camera"];
    CameraIntrinsics k{readInt(node,"width","camera"),readInt(node,"height","camera"),
        readFiniteDouble(node,"fx","camera"),readFiniteDouble(node,"fy","camera"),
        readFiniteDouble(node,"cx","camera"),readFiniteDouble(node,"cy","camera")};
    validateCamera(k);distortion.clear();
    if(!file["distortion"].empty())file["distortion"]>>distortion;
    return k;
}
namespace {
class AnalyticalConfigReader {
public:
    explicit AnalyticalConfigReader(cv::FileNode root) : root_(root) {}

    bool contains(const char* key) const { return !root_[key].empty(); }
    std::string text(const char* key) const { return std::string(root_[key]); }

    void integer(const char* key, int& value) const
    {
        if(contains(key)) value = readInt(root_, key, "analytical");
    }
    void number(const char* key, double& value) const
    {
        if(contains(key)) value = readFiniteDouble(root_, key, "analytical");
    }
    void size(const char* key, std::size_t& value) const
    {
        if(contains(key)) value = readSize(root_, key, "analytical");
    }
    void boolean(const char* key, bool& value) const
    {
        if(!contains(key)) return;
        const int number = readInt(root_, key, "analytical");
        if(number != 0 && number != 1)
            throw std::runtime_error(std::string(key) + " must be 0 or 1");
        value = number != 0;
    }
    void rejectUnknown(const std::vector<std::string>& known) const
    {
        for(const auto& node : root_) {
            if(std::find(known.begin(), known.end(), node.name()) == known.end())
                throw std::runtime_error("Unknown analytical setting: " + node.name());
        }
    }

private:
    cv::FileNode root_;
};
} // namespace

VisualFeatureSettings loadAnalyticalConfig(const std::filesystem::path& path)
{
    auto s = analyticalFeatureSettings();
    cv::FileStorage file(path.string(),cv::FileStorage::READ);
    if(!file.isOpened())
        throw std::runtime_error("Cannot open analytical config: " + path.string());
    const AnalyticalConfigReader config(file.root());
    auto& a = s.analytical;
    // Strict key checking catches misspelled ablations rather than silently running them.
    if(config.contains("features"))
        a.features = selectAnalyticFeatures(config.text("features"));
    if(config.contains("comparison_features"))
        s.features = selectVisualFeatures(config.text("comparison_features"));

    config.integer("working_width", a.working_size.width);
    config.integer("working_height", a.working_size.height);
    config.integer("grid_width", a.grid_size.width);
    config.integer("grid_height", a.grid_size.height);
    config.integer("gaussian_kernel", a.gaussian_kernel);
    config.number("gaussian_sigma", a.gaussian_sigma);
    config.number("harris_k", a.harris_k);
    config.number("harris_scale", a.harris_scale);
    config.number("canny_low", s.canny_low);
    config.number("canny_high", s.canny_high);
    config.number("flow_scale_px", a.flow_scale_px);
    config.number("depth_scale_m", a.depth_scale_m);
    config.number("depth_gradient_scale", a.depth_gradient_scale);
    config.number("roughness_scale_m", a.roughness_scale_m);
    config.integer("flow_levels", a.flow_levels);
    config.integer("flow_window", a.flow_window);
    config.integer("flow_iterations", a.flow_iterations);
    config.integer("flow_poly_n", a.flow_poly_n);
    config.number("flow_pyramid_scale", a.flow_pyramid_scale);
    config.number("flow_poly_sigma", a.flow_poly_sigma);
    config.boolean("enable_geometry", a.enable_geometry);
    config.boolean("batch_dimension", a.batch_dimension);
    config.integer("pose_interval", a.pose_interval);
    config.number("map_radius_m", a.map_radius_m);
    config.number("map_age_s", a.map_age_s);
    config.number("voxel_size_m", a.voxel_size_m);
    config.number("grid_resolution_m", a.grid_resolution_m);
    config.size("maximum_map_points", a.maximum_map_points);
    config.size("maximum_grid_cells", a.maximum_grid_cells);
    config.boolean("idw_enabled", a.idw.enabled);
    config.number("idw_radius_m", a.idw.search_radius_m);
    config.number("idw_maximum_distance_m", a.idw.maximum_interpolation_distance_m);
    config.number("idw_power", a.idw.power);
    config.integer("idw_minimum_neighbors", a.idw.minimum_neighbors);
    config.integer("idw_maximum_neighbors", a.idw.maximum_neighbors);
    config.integer("diffusion_iterations", a.surface.diffusion_iterations);
    config.number("diffusion_time_step", a.surface.diffusion_time_step);
    config.number("diffusion_conductance_m", a.surface.diffusion_conductance_m);
    config.integer("roughness_window", a.surface.roughness_window);
    config.integer("maximum_tracks", s.motion.maximum_tracks);
    config.integer("replenish_below", s.motion.replenish_below);
    config.number("corner_quality", s.motion.corner_quality);
    config.number("corner_distance_px", s.motion.corner_distance_px);
    config.integer("corner_block_size", s.motion.corner_block_size);
    config.boolean("forward_backward", s.motion.forward_backward);
    config.integer("motion_pyramid_levels", s.motion.pyramid_levels);
    config.integer("motion_window", s.motion.window_size);
    config.number("motion_model_error_px", s.motion.model_error_px);
    config.boolean("require_imu", s.motion.require_imu);
    config.integer("triangulation_interval", s.motion.triangulation_interval);
    config.integer("maximum_triangulations_per_frame",
                   s.motion.maximum_triangulations_per_frame);
    config.number("minimum_baseline_m",
                  s.motion.triangulation_limits.minimum_baseline_m);
    config.number("minimum_parallax_rad",
                  s.motion.triangulation_limits.minimum_parallax_rad);
    config.number("maximum_reprojection_error_px",
                  s.motion.triangulation_limits.maximum_reprojection_error_px);
    config.number("maximum_depth_m",
                  s.motion.triangulation_limits.maximum_depth_m);
    config.integer("pose_minimum_inliers", a.relative_pose.minimum_inliers);
    config.integer("pose_maximum_iterations", a.relative_pose.maximum_iterations);
    config.integer("pose_maximum_points", a.relative_pose.maximum_points);
    config.number("pose_threshold_px", a.relative_pose.threshold_px);
    config.number("pose_minimum_parallax_rad", a.relative_pose.minimum_parallax_rad);
    config.number("pose_maximum_reprojection_error_px",
                  a.relative_pose.maximum_reprojection_error_px);
    config.number("pose_maximum_relative_depth", a.relative_pose.maximum_relative_depth);

    const std::vector<std::string> known_keys = {
        "batch_dimension", "canny_high", "canny_low", "comparison_features",
        "corner_block_size", "corner_distance_px", "corner_quality",
        "depth_gradient_scale", "depth_scale_m", "diffusion_conductance_m",
        "diffusion_iterations", "diffusion_time_step", "enable_geometry", "features",
        "flow_iterations", "flow_levels", "flow_poly_n", "flow_poly_sigma",
        "flow_pyramid_scale", "flow_scale_px", "flow_window", "forward_backward",
        "gaussian_kernel", "gaussian_sigma", "grid_height", "grid_resolution_m",
        "grid_width", "harris_k", "harris_scale", "idw_enabled",
        "idw_maximum_distance_m", "idw_maximum_neighbors", "idw_minimum_neighbors",
        "idw_power", "idw_radius_m", "map_age_s", "map_radius_m",
        "maximum_depth_m", "maximum_grid_cells", "maximum_map_points",
        "maximum_reprojection_error_px", "maximum_tracks",
        "maximum_triangulations_per_frame", "minimum_baseline_m",
        "minimum_parallax_rad", "motion_model_error_px", "motion_pyramid_levels",
        "motion_window", "pose_interval", "pose_maximum_iterations",
        "pose_maximum_points", "pose_maximum_relative_depth",
        "pose_maximum_reprojection_error_px", "pose_minimum_inliers",
        "pose_minimum_parallax_rad", "pose_threshold_px", "replenish_below",
        "require_imu", "roughness_scale_m", "roughness_window",
        "triangulation_interval", "voxel_size_m", "working_height", "working_width"
    };
    config.rejectUnknown(known_keys);
    detail::validateAnalyticalSettings(s);
    return s;
}
} // namespace metric_mapping
namespace metric_mapping::detail {
void validateAnalyticalSettings(const VisualFeatureSettings& s)
{
    const auto& a=s.analytical;
    for(double v:s.distortion)if(!std::isfinite(v))throw std::runtime_error("Nonfinite distortion");
    const auto n=s.distortion.size();if(n&&n!=4&&n!=5&&n!=8&&n!=12&&n!=14)throw std::runtime_error("Unsupported distortion vector length");
    if(!a.enabled)return;
    for(double v:{a.gaussian_sigma,a.harris_k,a.harris_scale,a.flow_scale_px,a.depth_scale_m,a.depth_gradient_scale,
        a.roughness_scale_m,a.flow_pyramid_scale,a.flow_poly_sigma,a.map_radius_m,a.map_age_s,a.voxel_size_m,a.grid_resolution_m,
        a.idw.search_radius_m,a.idw.maximum_interpolation_distance_m,a.idw.power,a.surface.diffusion_time_step,a.surface.diffusion_conductance_m})
        if(!std::isfinite(v)||v<=0)throw std::runtime_error("Analytical scales must be positive and finite");
    if(a.working_size.width<32||a.working_size.height<32||a.working_size.width>1024||a.working_size.height>1024||
       a.grid_size.width<2||a.grid_size.height<2||a.grid_size.width>a.working_size.width||a.grid_size.height>a.working_size.height||
       a.gaussian_kernel<1||a.gaussian_kernel>15||a.gaussian_kernel%2==0||a.harris_k>=0.25||
       a.flow_levels<1||a.flow_levels>5||a.flow_window<5||a.flow_window>51||a.flow_window%2==0||a.flow_iterations<1||a.flow_iterations>10||
       (a.flow_poly_n!=5&&a.flow_poly_n!=7)||a.flow_pyramid_scale>=1||a.pose_interval<1||
       a.maximum_map_points<1||a.maximum_map_points>10000||a.maximum_grid_cells<1||a.maximum_grid_cells>65536||
       a.idw.minimum_neighbors<1||a.idw.maximum_neighbors<a.idw.minimum_neighbors||
       a.surface.diffusion_iterations<0||a.surface.diffusion_iterations>100||a.surface.diffusion_time_step>0.25||
       a.surface.roughness_window<3||a.surface.roughness_window>31||a.surface.roughness_window%2==0)
        throw std::runtime_error("Invalid analytical settings");
}
} // namespace metric_mapping::detail
#endif
