#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace metric_mapping {
namespace {
CameraIntrinsics imagePlaneCoordinates(cv::Size size)
{
    // These are coordinate-normalization constants, NOT camera calibration.
    // Equal scales make the existing consensus model an image-space similarity.
    const double scale = std::max(size.width, size.height);
    return {size.width, size.height, scale, scale, (size.width - 1) * 0.5, (size.height - 1) * 0.5};
}
OpticalFlowSettings pixelOnlySettings(OpticalFlowSettings settings)
{
    settings.triangulation = false;
    return settings;
}
}

PixelFlowTracker::PixelFlowTracker(cv::Size input_size, OpticalFlowSettings settings)
    : tracker_(imagePlaneCoordinates(input_size), pixelOnlySettings(settings)) {}

cv::Size PixelFlowTracker::workingSize() const
{
    const auto& coordinates = tracker_.workingCamera();
    return {coordinates.width, coordinates.height};
}

PixelFlowEstimate PixelFlowTracker::processFrame(const cv::Mat& frame, double timestamp)
{
    // Never supply altitude. Only expose image measurements, not inferred pose.
    last_ = tracker_.processFrame(frame, timestamp);
    PixelFlowEstimate result;
    result.valid = last_.valid;
    result.status = last_.status;
    result.dt_s = last_.dt_s;
    result.median_flow_px = last_.median_flow_px;
    if (result.valid && result.dt_s > 0) {
        const double u = result.median_flow_px.x / result.dt_s;
        const double v = result.median_flow_px.y / result.dt_s;
        const double limit = std::numeric_limits<float>::max();
        if (std::isfinite(u) && std::isfinite(v) && std::abs(u) <= limit && std::abs(v) <= limit)
            result.velocity_px_s = cv::Point2f(float(u), float(v));
    }
    result.active_tracks = last_.active_tracks;
    result.tracked_points = last_.tracked_points;
    result.accepted_points = last_.accepted_points;
    result.quality = last_.quality;
    result.processing_ms = last_.timing.total_ms;
    return result;
}
} // namespace metric_mapping
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

// --- Tracker Internal ---

namespace metric_mapping {
struct SparseFlowTracker::Impl {
    struct Anchor {
        bool valid = false, triangulated = false;
        cv::Point2f pixel;
        NadirPose pose;
        std::uint32_t age = 0;
    };
    CameraIntrinsics input_camera, camera;
    OpticalFlowSettings settings;
    cv::Mat gray, resized, mask, corners;
    std::vector<cv::Mat> previous_pyramid, current_pyramid;
    int previous_levels = 0, current_levels = 0;
    std::vector<TrackedFeature> tracks, correspondences;
    std::vector<Anchor> anchors;
    std::vector<cv::Point2f> previous, next, backward;
    std::vector<unsigned char> status, backward_status;
    std::vector<float> errors, backward_errors;
    std::vector<int> cells;
    motion_detail::GeometryScratch scratch;
    std::vector<SparseLandmark> landmarks;
    std::uint64_t frame_index = 0, next_id = 1, segment_id = 0;
    std::uint64_t last_detection = 0;
    bool initialized = false, detected = false, metric_chain = false;
    double previous_time = 0, origin_height = 0;
    std::optional<double> previous_altitude;
    std::optional<ImuSample> previous_imu, current_imu;
    NadirPose pose;
    std::size_t landmark_cursor = 0, triangulation_cursor = 0;

    Impl(const CameraIntrinsics& input, const OpticalFlowSettings& config);
    void preprocess(const cv::Mat& frame);
    void buildCurrentPyramid();
    std::optional<double> prepareImu(std::optional<ImuSample> imu, double timestamp, MotionEstimate& result);
    void estimateTrackedMotion(MotionEstimate& result, std::optional<double> height,
                               std::optional<double> imu_yaw_delta);
    void replenish(MotionEstimate& result);
    void track(MotionEstimate& result);
    void updateMap(MotionEstimate& result, std::optional<double> altitude);
    void clearMetricChain();
};
} // namespace metric_mapping

// --- Settings ---

namespace metric_mapping::motion_detail {
void validateSettings(const OpticalFlowSettings& s)
{
    if (s.corner_block_size < 3 || s.corner_block_size > 15 || s.corner_block_size % 2 == 0)
        throw std::runtime_error("Invalid corner block size");
    for (double v : {s.corner_quality, s.corner_distance_px, s.lk_epsilon,
                     s.maximum_lk_error, s.maximum_forward_backward_error_px,
                     s.model_error_px, s.minimum_inlier_fraction, s.minimum_coverage,
                     s.minimum_quality, s.maximum_yaw_rad, s.maximum_scale_change,
                     s.maximum_frame_gap_s, s.maximum_altitude_age_s, s.altitude_scale_tolerance,
                     s.maximum_imu_age_s, s.maximum_imu_tilt_rad, s.maximum_imu_tilt_change_rad,
                     s.triangulation_limits.minimum_baseline_m,
                     s.triangulation_limits.minimum_parallax_rad,
                     s.triangulation_limits.maximum_reprojection_error_px,
                     s.triangulation_limits.maximum_depth_m})
        if (!std::isfinite(v) || v <= 0)
            throw std::runtime_error("Motion settings must be positive and finite");
    if (s.maximum_width < 64 || s.maximum_height < 64 ||
        s.maximum_width > 4096 || s.maximum_height > 4096 ||
        s.maximum_tracks < 12 || s.maximum_tracks > 1000 ||
        s.minimum_inliers < 6 || s.minimum_inliers > s.replenish_below ||
        s.replenish_below > s.maximum_tracks ||
        s.grid_columns < 2 || s.grid_columns > 16 || s.grid_rows < 2 || s.grid_rows > 16 ||
        s.detection_interval < 1 || s.detection_interval > 1000 ||
        s.corner_quality >= 1 || s.corner_distance_px > 100 ||
        s.window_size < 5 || s.window_size > 31 || s.window_size % 2 == 0 ||
        s.pyramid_levels < 0 || s.pyramid_levels > 2 ||
        s.lk_iterations < 1 || s.lk_iterations > 30 ||
        s.minimum_inlier_fraction > 1 || s.minimum_coverage > 1 || s.minimum_quality > 1 ||
        s.maximum_yaw_rad > 0.5 || s.maximum_scale_change >= 0.5 ||
        s.maximum_imu_age_s > 0.1 || s.maximum_imu_tilt_rad > 10 * CV_PI / 180 ||
        s.maximum_imu_tilt_change_rad > 2 * CV_PI / 180 ||
        s.altitude_scale_tolerance >= 0.5 || s.triangulation_interval < 1 ||
        s.minimum_track_age < 2 || s.maximum_triangulations_per_frame < 1 ||
        s.maximum_triangulations_per_frame > s.maximum_tracks ||
        s.maximum_landmarks < 1 || s.maximum_landmarks > 10000 ||
        s.triangulation_limits.minimum_parallax_rad >= CV_PI / 2)
        throw std::runtime_error("Invalid sparse motion configuration");
}

void GeometryScratch::reserve(int count, int cell_count)
{
    first.reserve(count); second.reserve(count); values.reserve(count); errors.reserve(count);
    mask.reserve(count); trial.reserve(count); cells.resize(cell_count);
}

float median(std::vector<float>& values)
{
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle; // Upper median: deterministic, no extra pass for an even count.
}
} // namespace metric_mapping::motion_detail

// --- Geometry ---

namespace metric_mapping::motion_detail {
namespace {
bool fit(const GeometryScratch& w, Similarity& model, const std::optional<cv::Vec2d>& rotation)
{
    cv::Vec2d p(0, 0), q(0, 0);
    int count = 0;
    for (std::size_t i = 0; i < w.first.size(); ++i) if (w.mask[i]) {
        p += w.first[i]; q += w.second[i]; ++count;
    }
    if (count < 3) return false;
    p /= count; q /= count;
    double denominator = 0, a = 0, b = 0;
    for (std::size_t i = 0; i < w.first.size(); ++i) if (w.mask[i]) {
        const auto x = w.first[i] - p, y = w.second[i] - q;
        denominator += x.dot(x);
        a += x.dot(y); b += x[0] * y[1] - x[1] * y[0];
    }
    if (denominator < 1e-6) return false;
    if (rotation) {
        // With IMU yaw fixed, only scale and translation remain to estimate.
        const double scale = (a * (*rotation)[0] + b * (*rotation)[1]) / denominator;
        if (scale <= 0) return false;
        model.a = scale * (*rotation)[0];
        model.b = scale * (*rotation)[1];
    } else {
        model.a = a / denominator;
        model.b = b / denominator;
    }
    model.t = q - cv::Vec2d(model.a * p[0] - model.b * p[1],
                            model.b * p[0] + model.a * p[1]);
    return true;
}

bool plausible(const Similarity& m, const OpticalFlowSettings& s)
{
    return std::abs(std::hypot(m.a, m.b) - 1) <= s.maximum_scale_change &&
           std::abs(std::atan2(m.b, m.a)) <= s.maximum_yaw_rad;
}

std::size_t classify(const Similarity& m, const CameraIntrinsics& k,
                     GeometryScratch& w, double threshold)
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < w.first.size(); ++i) {
        const auto error = w.second[i] - m.apply(w.first[i]);
        w.errors[i] = float(std::hypot(error[0] * k.fx, error[1] * k.fy));
        w.trial[i] = w.errors[i] <= threshold;
        count += w.trial[i];
    }
    return count;
}
} // namespace

Similarity estimateMotion(const std::vector<TrackedFeature>& tracks,
                          const CameraIntrinsics& camera, const OpticalFlowSettings& s,
                          GeometryScratch& w, MotionEstimate& out, std::optional<double> imu_yaw_delta)
{
    const std::size_t n = tracks.size();
    Similarity model;
    w.mask.assign(n, 0);
    if (n < std::size_t(s.minimum_inliers)) return model;
    std::optional<cv::Vec2d> rotation;
    if (imu_yaw_delta) {
        if (!std::isfinite(*imu_yaw_delta) || std::abs(*imu_yaw_delta) > s.maximum_yaw_rad) return model;
        rotation = cv::Vec2d(std::cos(*imu_yaw_delta), std::sin(*imu_yaw_delta));
    }
    w.first.resize(n); w.second.resize(n); w.errors.resize(n); w.trial.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        w.first[i] = {(tracks[i].previous_px.x - camera.cx) / camera.fx,
                      (tracks[i].previous_px.y - camera.cy) / camera.fy};
        w.second[i] = {(tracks[i].current_px.x - camera.cx) / camera.fx,
                       (tracks[i].current_px.y - camera.cy) / camera.fy};
    }
    // A loose MAD gate discards gross jumps without rejecting valid yaw flow.
    cv::Point2f center;
    for (int axis = 0; axis < 2; ++axis) {
        w.values.clear();
        for (const auto& track : tracks) {
            const auto d = track.current_px - track.previous_px;
            w.values.push_back(axis == 0 ? d.x : d.y);
        }
        (axis == 0 ? center.x : center.y) = median(w.values);
    }
    w.values.clear();
    for (std::size_t i = 0; i < n; ++i) {
        w.errors[i] = float(cv::norm(tracks[i].current_px - tracks[i].previous_px - center));
        w.values.push_back(w.errors[i]);
    }
    const float spread = median(w.values);
    for (std::size_t i = 0; i < n; ++i) w.mask[i] = w.errors[i] <= std::max(3.0F, 4 * spread);
    bool fitted = fit(w, model, rotation);
    std::size_t count = fitted ? classify(model, camera, w, s.model_error_px) : 0;
    // At most 32 two-point hypotheses, only when the cheap fit lacks consensus.
    // This is a bounded RANSAC-style fallback, not arbitrary 6-DoF recovery.
    if (!fitted || !plausible(model, s) || count < 0.8 * n) {
        out.consensus_fallback = true;
        Similarity best = model;
        std::size_t best_count = fitted && plausible(model, s) ? count : 0;
        std::uint32_t random = 0x12345U;
        for (int iteration = 0; iteration < 32; ++iteration) {
            random = random * 1664525U + 1013904223U;
            const auto i = std::size_t(random) % n;
            random = random * 1664525U + 1013904223U;
            const auto j = std::size_t(random) % n;
            const auto p = w.first[j] - w.first[i], q = w.second[j] - w.second[i];
            const double norm = p.dot(p);
            const auto pixel_span = tracks[j].previous_px - tracks[i].previous_px;
            // Conditioning depends on image spread; don't silently impose a
            // field-of-view limit through a fixed normalized-coordinate cutoff.
            if (pixel_span.dot(pixel_span) < 400 || norm < 1e-12) continue;
            Similarity candidate;
            if (rotation) {
                const cv::Vec2d rotated((*rotation)[0] * p[0] - (*rotation)[1] * p[1],
                                       (*rotation)[1] * p[0] + (*rotation)[0] * p[1]);
                const double scale = rotated.dot(q) / norm;
                if (scale <= 0) continue;
                candidate.a = scale * (*rotation)[0];
                candidate.b = scale * (*rotation)[1];
            } else {
                candidate.a = p.dot(q) / norm;
                candidate.b = (p[0] * q[1] - p[1] * q[0]) / norm;
            }
            candidate.t = w.second[i] - candidate.apply(w.first[i]);
            if (!plausible(candidate, s)) continue;
            const auto support = classify(candidate, camera, w, s.model_error_px);
            if (support > best_count) { best = candidate; best_count = support; }
        }
        model = best;
    }
    for (int pass = 0; pass < 2; ++pass) {
        classify(model, camera, w, s.model_error_px);
        w.mask = w.trial;
        if (!fit(w, model, rotation)) { w.mask.assign(n, 0); return model; }
    }
    count = classify(model, camera, w, s.model_error_px);
    w.mask = w.trial;
    out.accepted_points = count;
    // Geometry sees only tracks that survived LK, bounds, residual and optional
    // forward/backward checks. Dividing by every attempted LK track double-counts
    // those earlier rejections and can invalidate a strong geometric consensus.
    out.inlier_fraction = n ? double(count) / n : 0;
    std::fill(w.cells.begin(), w.cells.end(), 0);
    double squared_error = 0;
    for (std::size_t i = 0; i < n; ++i) if (w.mask[i]) {
        squared_error += w.errors[i] * w.errors[i];
        const auto& p = tracks[i].current_px;
        const int col = std::clamp(int(p.x * s.grid_columns / camera.width), 0, s.grid_columns - 1);
        const int row = std::clamp(int(p.y * s.grid_rows / camera.height), 0, s.grid_rows - 1);
        w.cells[row * s.grid_columns + col] = 1;
    }
    out.coverage = double(std::count(w.cells.begin(), w.cells.end(), 1)) / w.cells.size();
    out.residual_px = count ? std::sqrt(squared_error / count) : s.model_error_px;
    out.quality = out.inlier_fraction * out.coverage /
                  (1 + out.residual_px / s.model_error_px);
    if (count < std::size_t(s.minimum_inliers) || !plausible(model, s) ||
        out.inlier_fraction < s.minimum_inlier_fraction || out.coverage < s.minimum_coverage ||
        out.quality < s.minimum_quality) return model;
    out.valid = true; out.status = "ok";
    out.imu_used = imu_yaw_delta.has_value();
    out.yaw_delta_rad = std::atan2(model.b, model.a);
    out.image_scale = std::hypot(model.a, model.b);
    for (int axis = 0; axis < 2; ++axis) {
        w.values.clear();
        for (std::size_t i = 0; i < n; ++i) if (w.mask[i]) {
            const auto d = tracks[i].current_px - tracks[i].previous_px;
            w.values.push_back(axis == 0 ? d.x : d.y);
        }
        (axis == 0 ? out.median_flow_px.x : out.median_flow_px.y) = median(w.values);
    }
    // q2 = s R(theta) q1 + t. Camera displacement/h1 is -R^T t/s,
    // then flip camera-down Y to local world-up Y. This is NOT raw flow.
    const double scale_squared = model.a * model.a + model.b * model.b;
    out.normalized_displacement = {-(model.a * model.t[0] + model.b * model.t[1]) / scale_squared,
                                   (-model.b * model.t[0] + model.a * model.t[1]) / scale_squared};
    const double length = cv::norm(out.normalized_displacement);
    if (length * std::min(camera.fx, camera.fy) > 0.1)
        out.translation_direction = out.normalized_displacement / length;
    return model;
}
} // namespace metric_mapping::motion_detail

// --- Triangulation ---

namespace metric_mapping {
namespace motion_detail {
cv::Matx33d nadirRotation(double yaw)
{
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {c, s, 0, s, -c, 0, 0, 0, -1}; // Rz(yaw) diag(1,-1,-1).
}
} // namespace motion_detail

AltitudeSample altitudeFromUltrasonic(const UltrasonicMeasurement& reading, double timestamp_s)
{
    validateUltrasonicMeasurement(reading);
    // range = sensor_z - ground_z = camera_height + offset_z.
    const double height = reading.distance_m - reading.sensor_offset_world_m[2];
    if (!std::isfinite(timestamp_s) || !std::isfinite(height) || height <= 0)
        throw std::runtime_error("Invalid camera altitude or range timestamp");
    return {height, timestamp_s};
}

std::optional<TriangulatedPoint> triangulateTrack(
    const cv::Point2f& first, const cv::Point2f& second,
    const CameraIntrinsics& camera, const NadirPose& first_pose,
    const NadirPose& second_pose, const TriangulationSettings& settings)
{
    validateCamera(camera);
    for (double value : {settings.minimum_baseline_m, settings.minimum_parallax_rad,
                         settings.maximum_reprojection_error_px, settings.maximum_depth_m})
        if (!std::isfinite(value) || value <= 0)
            throw std::runtime_error("Invalid triangulation limits");
    if (settings.minimum_parallax_rad >= CV_PI / 2)
        throw std::runtime_error("Triangulation minimum parallax must be less than pi/2");
    for (const auto& pose : {first_pose, second_pose}) {
        if (!std::isfinite(pose.yaw_rad)) return std::nullopt;
        for (double value : pose.position_m.val)
            if (!std::isfinite(value)) return std::nullopt;
    }
    for (const auto& pixel : {first, second})
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
            pixel.x < 0 || pixel.x >= camera.width || pixel.y < 0 || pixel.y >= camera.height)
            return std::nullopt;
    const auto r1 = motion_detail::nadirRotation(first_pose.yaw_rad);
    const auto r2 = motion_detail::nadirRotation(second_pose.yaw_rad);
    cv::Vec3d u = r1 * cv::Vec3d((first.x - camera.cx) / camera.fx,
                                 (first.y - camera.cy) / camera.fy, 1);
    cv::Vec3d v = r2 * cv::Vec3d((second.x - camera.cx) / camera.fx,
                                 (second.y - camera.cy) / camera.fy, 1);
    u /= cv::norm(u);
    v /= cv::norm(v);
    const cv::Vec3d baseline = second_pose.position_m - first_pose.position_m;
    if (cv::norm(baseline) < settings.minimum_baseline_m) return std::nullopt;
    const double dot = std::clamp(u.dot(v), -1.0, 1.0);
    const double denominator = 1 - dot * dot;
    // sin^2(parallax) controls conditioning, not raw pixel displacement (yaw).
    const double min_sin = std::sin(settings.minimum_parallax_rad);
    if (dot <= 0 || denominator < min_sin * min_sin) return std::nullopt;
    const double along_u = (u.dot(baseline) - dot * v.dot(baseline)) / denominator;
    const double along_v = (dot * u.dot(baseline) - v.dot(baseline)) / denominator;
    if (along_u <= 0 || along_v <= 0) return std::nullopt;
    const cv::Vec3d point = 0.5 * (first_pose.position_m + along_u * u +
                                  second_pose.position_m + along_v * v);
    double error = 0;
    const auto project = [&](const NadirPose& pose, const cv::Matx33d& rotation,
                             const cv::Point2f& observed) {
        const cv::Vec3d p = rotation.t() * (point - pose.position_m);
        if (!std::isfinite(p[2]) || p[2] <= 0 || p[2] > settings.maximum_depth_m)
            return false;
        const double residual = std::hypot(camera.fx * p[0] / p[2] + camera.cx - observed.x,
                                           camera.fy * p[1] / p[2] + camera.cy - observed.y);
        if (!std::isfinite(residual)) return false;
        error = std::max(error, residual);
        return residual <= settings.maximum_reprojection_error_px;
    };
    if (!project(first_pose, r1, first) || !project(second_pose, r2, second))
        return std::nullopt;
    return TriangulatedPoint{point, std::acos(dot), error};
}
} // namespace metric_mapping

// --- Optical Flow ---

namespace metric_mapping {
using namespace motion_detail;

SparseFlowTracker::Impl::Impl(const CameraIntrinsics& input, const OpticalFlowSettings& config)
    : input_camera(input), camera(input), settings(config)
{
    validateCamera(input); validateSettings(config);
    const auto working_size = detail::fitImageSize({input.width, input.height},
                                                  config.maximum_width, config.maximum_height);
    camera = detail::resizeCamera(input, working_size);
    if (camera.width / config.grid_columns < config.window_size + 4 ||
        camera.height / config.grid_rows < config.window_size + 4)
        throw std::runtime_error("Motion image is too small for configured feature grid/window");
    const int n = config.maximum_tracks;
    tracks.reserve(n); anchors.reserve(n); previous.reserve(n); next.reserve(n);
    status.reserve(n); errors.reserve(n);
    if (config.forward_backward) {
        backward.reserve(n); backward_status.reserve(n); backward_errors.reserve(n);
    }
    previous_pyramid.reserve(2 * (config.pyramid_levels + 1));
    current_pyramid.reserve(2 * (config.pyramid_levels + 1));
    cells.resize(config.grid_rows * config.grid_columns);
    scratch.reserve(n, int(cells.size()));
    if (config.triangulation) landmarks.reserve(config.maximum_landmarks);
}

void SparseFlowTracker::Impl::preprocess(const cv::Mat& frame)
{
    if (frame.size() == cv::Size(camera.width, camera.height)) {
        if (frame.channels() == 1) frame.copyTo(gray);
        else cv::cvtColor(frame, gray, frame.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    } else {
        // Downsample before color conversion; no full-resolution gray buffer.
        if (frame.channels() == 1)
            cv::resize(frame, gray, {camera.width, camera.height}, 0, 0, cv::INTER_AREA);
        else {
            cv::resize(frame, resized, {camera.width, camera.height}, 0, 0, cv::INTER_AREA);
            cv::cvtColor(resized, gray, frame.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
        }
    }
}

SparseFlowTracker::SparseFlowTracker(const CameraIntrinsics& camera, const OpticalFlowSettings& settings)
    : impl_(std::make_unique<Impl>(camera, settings)) {}
SparseFlowTracker::~SparseFlowTracker() = default;
const CameraIntrinsics& SparseFlowTracker::workingCamera() const { return impl_->camera; }
const std::vector<TrackedFeature>& SparseFlowTracker::tracks() const { return impl_->tracks; }
const std::vector<TrackedFeature>& SparseFlowTracker::correspondences() const { return impl_->correspondences; }
const std::vector<SparseLandmark>& SparseFlowTracker::landmarks() const { return impl_->landmarks; }

void SparseFlowTracker::reset()
{
    impl_->correspondences.clear();
    auto& p = *impl_;
    p.previous_imu.reset(); p.current_imu.reset();
    p.initialized = false; p.detected = false; p.frame_index = 0;
    p.tracks.clear(); p.anchors.clear(); p.previous_altitude.reset(); p.clearMetricChain();
}

// Building a pyramid is expensive. Both initialization and tracking use this
// helper, and processFrame keeps the result for the next frame.
void SparseFlowTracker::Impl::buildCurrentPyramid()
{
    current_pyramid.resize(2 * (settings.pyramid_levels + 1));
    current_levels = cv::buildOpticalFlowPyramid(gray, current_pyramid,
        {settings.window_size, settings.window_size}, settings.pyramid_levels, true,
        cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
}

void SparseFlowTracker::Impl::estimateTrackedMotion(MotionEstimate& result, std::optional<double> height,
                                                   std::optional<double> imu_yaw_delta)
{
    estimateMotion(tracks, camera, settings, scratch, result, imu_yaw_delta);
    if (!result.valid) {
        if (imu_yaw_delta) {
            result.imu_status = "imu_visual_no_consensus";
            result.metric_status = "imu_visual_no_consensus";
        }
        tracks.clear();
        anchors.clear();
        return;
    }

    // Keep track identity and triangulation anchors together when rejecting points.
    std::size_t kept = 0;
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        if (!scratch.mask[i]) continue;
        tracks[kept] = tracks[i];
        anchors[kept] = anchors[i];
        ++kept;
    }
    tracks.resize(kept);
    anchors.resize(kept);

    // Valid optical flow does not imply valid metric motion. Both height readings
    // must exist, and observed image scale must agree with their height ratio.
    if (!height || !previous_altitude) return;
    const double expected_scale = *previous_altitude / *height;
    if (!(std::abs(result.image_scale / expected_scale - 1) <= settings.altitude_scale_tolerance)) {
        result.metric_status = "altitude_scale_mismatch";
        return;
    }
    // normalized displacement = -D R^T t/s. Use measured h2*s for metric scale.
    const auto displacement = result.normalized_displacement * (*height * result.image_scale);
    const auto velocity = displacement / result.dt_s;
    if (!std::isfinite(displacement[0]) || !std::isfinite(displacement[1]) ||
        !std::isfinite(velocity[0]) || !std::isfinite(velocity[1])) return;
    result.planar_displacement_m = displacement;
    result.planar_velocity_mps = velocity;
    result.metric_status = "ok";
}

// The supported model is nadir + small yaw. Roll/pitch are health checks, not
// an excuse to report horizontal motion during an unsupported camera tilt.
std::optional<double> SparseFlowTracker::Impl::prepareImu(std::optional<ImuSample> imu,
                                                        double timestamp, MotionEstimate& result)
{
    current_imu.reset();
    if (!imu) {
        result.imu_status = settings.require_imu ? "imu_unavailable" : "not_supplied";
        return std::nullopt;
    }
    if (!std::isfinite(imu->timestamp_s) || !std::isfinite(imu->roll_rad) ||
        !std::isfinite(imu->pitch_rad) || !std::isfinite(imu->yaw_rad)) {
        result.imu_status = "imu_invalid"; return std::nullopt;
    }
    if (imu->timestamp_s > timestamp) { result.imu_status = "imu_future"; return std::nullopt; }
    if (timestamp - imu->timestamp_s > settings.maximum_imu_age_s) {
        result.imu_status = "imu_stale"; return std::nullopt;
    }
    if (std::hypot(imu->roll_rad, imu->pitch_rad) > settings.maximum_imu_tilt_rad) {
        result.imu_status = "imu_tilt_exceeded"; return std::nullopt;
    }
    current_imu = imu;
    if (!initialized || !previous_imu) { result.imu_status = "imu_initializing"; return std::nullopt; }
    if (imu->timestamp_s <= previous_imu->timestamp_s) {
        current_imu.reset(); result.imu_status = "imu_not_new"; return std::nullopt;
    }
    if (timestamp - previous_time > settings.maximum_frame_gap_s) {
        result.imu_status = "imu_frame_gap"; return std::nullopt;
    }
    const double roll_change = std::remainder(imu->roll_rad - previous_imu->roll_rad, 2 * CV_PI);
    const double pitch_change = std::remainder(imu->pitch_rad - previous_imu->pitch_rad, 2 * CV_PI);
    if (std::hypot(roll_change, pitch_change) > settings.maximum_imu_tilt_change_rad) {
        result.imu_status = "imu_tilt_changed"; return std::nullopt;
    }
    const double yaw = std::remainder(imu->yaw_rad - previous_imu->yaw_rad, 2 * CV_PI);
    if (!std::isfinite(yaw) || std::abs(yaw) > settings.maximum_yaw_rad) {
        result.imu_status = "imu_rotation_exceeded"; return std::nullopt;
    }
    result.imu_status = "ok";
    return yaw;
}

MotionEstimate SparseFlowTracker::processFrame(const cv::Mat& frame, double timestamp,
                                              std::optional<AltitudeSample> altitude, std::optional<ImuSample> imu)
{
    auto& state = *impl_;
    const auto& settings = state.settings;
    if (frame.empty() || frame.size() != cv::Size(state.input_camera.width, state.input_camera.height) ||
        frame.depth() != CV_8U || (frame.channels() != 1 && frame.channels() != 3 && frame.channels() != 4))
        throw std::runtime_error("Motion frame must match calibration and be 8-bit gray/BGR/BGRA");
    if (!std::isfinite(timestamp) || (state.initialized && timestamp <= state.previous_time))
        throw std::runtime_error("Motion timestamps must be finite and strictly increasing");
    const auto start = settings.profiling ? Clock::now() : Clock::time_point{};
    auto stage = start;
    MotionEstimate result;
    state.correspondences.clear();
    std::optional<double> height;
    if (altitude && std::isfinite(altitude->height_m) && altitude->height_m > 0 &&
        std::isfinite(altitude->timestamp_s) && timestamp >= altitude->timestamp_s &&
        timestamp - altitude->timestamp_s <= settings.maximum_altitude_age_s)
        height = altitude->height_m;
    const auto imu_yaw_delta = state.prepareImu(imu, timestamp, result);
    const bool imu_required = settings.require_imu || imu.has_value();
    const auto usable_height = imu_required && !imu_yaw_delta ? std::optional<double>{} : height;
    if (imu_required && !imu_yaw_delta) result.metric_status = result.imu_status;
    state.preprocess(frame);
    if (settings.profiling) { result.timing.preprocessing_ms = milliseconds(stage); stage = Clock::now(); }
    if (state.initialized) {
        result.dt_s = timestamp - state.previous_time;
        if (!std::isfinite(result.dt_s) || result.dt_s > settings.maximum_frame_gap_s) {
            state.tracks.clear(); state.anchors.clear(); state.detected = false;
            state.clearMetricChain(); result.status = "frame_gap";
        } else if (!state.tracks.empty()) {
            if (settings.profiling) stage = Clock::now();
            state.buildCurrentPyramid();
            if (settings.profiling) result.timing.flow_ms += milliseconds(stage);
            result.status = "poor_tracking";
            state.track(result);
            if (settings.retain_correspondences) state.correspondences = state.tracks;
            if (settings.profiling) stage = Clock::now();
            state.estimateTrackedMotion(result, usable_height, imu_yaw_delta);
            if (settings.profiling) result.timing.geometry_ms = milliseconds(stage);
        } else result.status = "insufficient_tracks";
    }
    if (settings.profiling) stage = Clock::now();
    state.updateMap(result, usable_height);
    if (settings.profiling) { result.timing.triangulation_ms = milliseconds(stage); stage = Clock::now(); }
    state.replenish(result);
    if (settings.profiling) { result.timing.detection_ms = milliseconds(stage); stage = Clock::now(); }
    // Only build a pyramid if a future frame has something to track. If track()
    // built this frame's pyramid, keep it instead of building it a second time.
    if (!state.tracks.empty()) {
        if (!result.tracked_points) {
            state.buildCurrentPyramid();
        }
        state.previous_pyramid.swap(state.current_pyramid); state.previous_levels = state.current_levels;
    }
    if (settings.profiling) result.timing.flow_ms += milliseconds(stage);
    result.active_tracks = state.tracks.size();
    state.previous_imu = state.current_imu;
    state.previous_time = timestamp; state.previous_altitude = height; state.initialized = true; ++state.frame_index;
    if (settings.profiling) result.timing.total_ms = milliseconds(start);
    return result;
}
} // namespace metric_mapping

// --- Tracks ---

namespace metric_mapping {
using namespace motion_detail;
void SparseFlowTracker::Impl::replenish(MotionEstimate& out)
{
    if (tracks.size() >= std::size_t(settings.replenish_below) ||
        (detected && frame_index - last_detection < std::uint64_t(settings.detection_interval))) return;
    detected = true; last_detection = frame_index; out.detection_ran = true;
    mask.create(gray.size(), CV_8U); mask.setTo(255);
    std::fill(cells.begin(), cells.end(), 0);
    const int margin = settings.window_size / 2 + 1;
    mask.rowRange(0, margin).setTo(0); mask.rowRange(mask.rows - margin, mask.rows).setTo(0);
    mask.colRange(0, margin).setTo(0); mask.colRange(mask.cols - margin, mask.cols).setTo(0);
    const auto cell = [&](const cv::Point2f& p) {
        return int(p.y * settings.grid_rows / camera.height) * settings.grid_columns +
               int(p.x * settings.grid_columns / camera.width);
    };
    const int radius = int(std::ceil(settings.corner_distance_px));
    for (const auto& t : tracks) {
        ++cells[cell(t.current_px)]; cv::circle(mask, t.current_px, radius, cv::Scalar(0), -1);
    }
    const int quota = (settings.maximum_tracks + int(cells.size()) - 1) / int(cells.size());
    for (int row = 0; row < settings.grid_rows; ++row) {
        for (int col = 0; col < settings.grid_columns; ++col) {
            const int wanted = std::min(quota - cells[row * settings.grid_columns + col],
                                        settings.maximum_tracks - int(tracks.size()));
            if (wanted <= 0) continue;
            const int x = col * camera.width / settings.grid_columns;
            const int y = row * camera.height / settings.grid_rows;
            const cv::Rect roi(x, y, (col + 1) * camera.width / settings.grid_columns - x,
                                (row + 1) * camera.height / settings.grid_rows - y);
            // Visit only cells needing corners, preserving every surviving ID.
            cv::goodFeaturesToTrack(gray(roi), corners, wanted, settings.corner_quality,
                                    settings.corner_distance_px, mask(roi), settings.corner_block_size, false);
            for (std::size_t i = 0; i < corners.total(); ++i) {
                auto point = corners.ptr<cv::Point2f>()[i];
                point += cv::Point2f(float(x), float(y));
                tracks.push_back({next_id++, 1, point, point});
                Anchor anchor;
                if (settings.triangulation && metric_chain) {
                    anchor.valid = true; anchor.pixel = point; anchor.pose = pose; anchor.age = 1;
                }
                anchors.push_back(anchor);
                cv::circle(mask, point, radius, cv::Scalar(0), -1); ++out.replenished_points;
            }
        }
    }
}

void SparseFlowTracker::Impl::track(MotionEstimate& out)
{
    const auto start = settings.profiling ? Clock::now() : Clock::time_point{};
    previous.clear();
    for (const auto& t : tracks) previous.push_back(t.current_px);
    out.tracked_points = previous.size();
    // Size output storage here instead of asking an uninstrumented OpenCV
    // binary to grow reserved STL vectors across an ASan/libc++ boundary.
    next.resize(previous.size()); status.resize(previous.size()); errors.resize(previous.size());
    const cv::Size window(settings.window_size, settings.window_size);
    const int levels = std::min(previous_levels, current_levels);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    settings.lk_iterations, settings.lk_epsilon);
    cv::calcOpticalFlowPyrLK(previous_pyramid, current_pyramid, previous, next, status,
                           errors, window, levels, criteria, 0, 1e-4);
    auto stage = settings.profiling ? Clock::now() : Clock::time_point{};
    if (settings.profiling) out.timing.flow_ms += milliseconds(start);
    // Compact before an optional backward pass: never pass failed/nonfinite LK
    // outputs into another solver, and don't pay twice for already failed tracks.
    std::size_t kept = 0;
    const float margin = float(settings.window_size / 2 + 1);
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        const auto& q = next[i];
        if (!status[i] || !std::isfinite(q.x) || !std::isfinite(q.y) ||
            !std::isfinite(errors[i]) || errors[i] > settings.maximum_lk_error ||
            q.x < margin || q.y < margin || q.x >= camera.width - margin || q.y >= camera.height - margin)
            continue;
        auto t = tracks[i]; t.previous_px = t.current_px; t.current_px = q;
        if (t.age < std::numeric_limits<std::uint32_t>::max()) ++t.age;
        tracks[kept] = t; anchors[kept] = anchors[i];
        previous[kept] = t.previous_px; next[kept] = q; ++kept;
    }
    tracks.resize(kept); anchors.resize(kept); previous.resize(kept); next.resize(kept);
    if (settings.profiling) out.timing.filtering_ms += milliseconds(stage);
    if (settings.forward_backward && kept) {
        if (settings.profiling) stage = Clock::now();
        backward.resize(kept); backward_status.resize(kept); backward_errors.resize(kept);
        cv::calcOpticalFlowPyrLK(current_pyramid, previous_pyramid, next, backward,
            backward_status, backward_errors, window, levels, criteria, 0, 1e-4);
        if (settings.profiling) { out.timing.flow_ms += milliseconds(stage); stage = Clock::now(); }
        kept = 0;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (!backward_status[i] || !std::isfinite(backward[i].x) || !std::isfinite(backward[i].y) ||
                cv::norm(backward[i] - previous[i]) > settings.maximum_forward_backward_error_px) continue;
            tracks[kept] = tracks[i]; anchors[kept] = anchors[i]; ++kept;
        }
        tracks.resize(kept); anchors.resize(kept);
        if (settings.profiling) out.timing.filtering_ms += milliseconds(stage);
    }
}
} // namespace metric_mapping

// --- Landmarks ---

namespace metric_mapping {
void SparseFlowTracker::Impl::clearMetricChain()
{
    metric_chain = false; landmarks.clear(); landmark_cursor = 0; triangulation_cursor = 0;
    for (auto& anchor : anchors) anchor = Anchor{};
}

void SparseFlowTracker::Impl::updateMap(MotionEstimate& out, std::optional<double> height)
{
    if (!out.planar_displacement_m) {
        clearMetricChain();
        if (height) {
            ++segment_id; metric_chain = true; origin_height = *height; pose = {};
        }
    } else {
        if (!metric_chain) {
            clearMetricChain();
            ++segment_id;
            metric_chain = true;
            // The displacement just estimated spans previous -> current. Anchor
            // the new segment at the previous frame so that measurement is not
            // silently discarded on every segment start.
            origin_height = previous_altitude.value_or(*height);
            pose = {};
        }
        const auto d = *out.planar_displacement_m;
        const double c = std::cos(pose.yaw_rad), s = std::sin(pose.yaw_rad);
        pose.position_m += cv::Vec3d(c * d[0] - s * d[1], s * d[0] + c * d[1], 0);
        pose.position_m[2] = *height - origin_height;
        pose.yaw_rad = std::remainder(pose.yaw_rad + out.yaw_delta_rad, 2 * CV_PI);
    }
    out.segment_id = segment_id;
    if (!metric_chain) return;
    out.pose = pose;
    if (!settings.triangulation) return;
    for (std::size_t i = 0; i < tracks.size(); ++i) if (!anchors[i].valid) {
        anchors[i] = {true, false, tracks[i].current_px, pose, tracks[i].age};
    }
    if (!out.planar_displacement_m || tracks.empty() ||
        frame_index % std::uint64_t(settings.triangulation_interval) != 0) return;
    // Rotate the starting index to prevent early tracks monopolizing the budget.
    for (std::size_t visited = 0; visited < tracks.size() &&
         out.triangulation_attempts < std::size_t(settings.maximum_triangulations_per_frame); ++visited) {
        const std::size_t i = triangulation_cursor++ % tracks.size();
        auto& anchor = anchors[i];
        const auto& t = tracks[i];
        if (anchor.triangulated || t.age - anchor.age + 1 < std::uint32_t(settings.minimum_track_age) ||
            cv::norm(pose.position_m - anchor.pose.position_m) < settings.triangulation_limits.minimum_baseline_m)
            continue;
        ++out.triangulation_attempts;
        const auto point = triangulateTrack(anchor.pixel, t.current_px, camera, anchor.pose,
                                            pose, settings.triangulation_limits);
        if (!point) continue;
        const SparseLandmark landmark{t.id, *point};
        if (landmarks.size() < std::size_t(settings.maximum_landmarks)) landmarks.push_back(landmark);
        else landmarks[landmark_cursor++ % landmarks.size()] = landmark;
        anchor.triangulated = true; ++out.new_landmarks;
    }
}
} // namespace metric_mapping

// --- Debug ---

namespace metric_mapping {
cv::Mat SparseFlowTracker::debugImage(const MotionEstimate& estimate) const
{
    const auto& p = *impl_;
    if (p.gray.empty()) return {};
    cv::Mat image;
    cv::cvtColor(p.gray, image, cv::COLOR_GRAY2BGR);
    for (const auto& t : p.tracks) {
        if (t.age > 1) cv::arrowedLine(image, t.previous_px, t.current_px, {0, 255, 0}, 1);
        else cv::circle(image, t.current_px, 2, {0, 200, 255}, -1);
    }
    std::ostringstream text;
    text << estimate.status << " " << estimate.accepted_points << '/' << estimate.tracked_points
         << " q=" << std::fixed << std::setprecision(2) << estimate.quality;
    cv::putText(image, text.str(), {5, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 255, 255}, 1);
    return image;
}
} // namespace metric_mapping

// --- Motion Sequence ---

namespace metric_mapping::demo {
cv::Mat motionTexture(cv::Size size)
{
    cv::Mat image(size, CV_8U);
    cv::RNG random(73129);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, {3, 3}, 0.7);
    return image;
}

void warpMotion(const cv::Mat& source, cv::Mat& destination,
                const CameraIntrinsics& k, double dx, double dy, double yaw, double scale)
{
    const double c = scale * std::cos(yaw), s = scale * std::sin(yaw);
    const double xy = -s * k.fx / k.fy, yx = s * k.fy / k.fx;
    const cv::Matx23d affine(c, xy, k.cx - c * k.cx - xy * k.cy + dx,
                             yx, c, k.cy - yx * k.cx - c * k.cy + dy);
    cv::warpAffine(source, destination, affine, source.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}
} // namespace metric_mapping::demo
