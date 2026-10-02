#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

// --- Visual Internal ---

namespace metric_mapping {
struct VisualFeatureExtractor::Impl {
    VisualFeatureSettings settings;
    std::optional<CameraIntrinsics> source_camera;
    VisualFeatureFrame output;
    cv::Mat gx, gy, magnitude, jxx, jxy, jyy, temporary, peaks;
    cv::Mat opponent_rg, opponent_by, dx, dy, color_energy, canny_dx, canny_dy, edge_mask;
    cv::Mat prepared, smooth, remap_x, remap_y, ycrcb, previous_dense, dense_flow;
    cv::Mat dense_u, dense_v, dense_valid, dense_counts, analytical_canny, analytical_contours;
    cv::Mat harris, contour_map, chroma_cb, chroma_cr, hog_angle;
    std::array<cv::Mat,3> color_planes;
    std::array<cv::Mat,9> hog;
    std::array<cv::Mat,28> bank, bank_masks;
    std::vector<std::uint64_t> seen_landmarks;
    std::uint64_t map_segment = 0;
    std::vector<cv::KeyPoint> candidates;
    std::vector<std::vector<cv::Point>> raw_contours;
    std::vector<std::size_t> contour_order;
    std::vector<int> cells;
#ifdef HAVE_VISUAL_MOTION
    std::unique_ptr<SparseFlowTracker> tracker;
#endif
    bool frame_seen = false;
    double last_timestamp = 0;
    explicit Impl(const VisualFeatureSettings& config, std::optional<CameraIntrinsics> camera);
    void validateFrame(const cv::Mat& bgr, double timestamp) const;
    void prepare(const cv::Mat& bgr);
    void extractSpatialFeatures();
    void derivatives();
    void edges();
    void structureTensor();
    void corners();
    void keypoints();
    void selectPoints(std::vector<cv::KeyPoint>& destination);
    void colorTransitions();
    void binaryTexture();
    void boundaries();
    void motion(double timestamp, std::optional<AltitudeSample> altitude, std::optional<ImuSample> imu);
    bool analytic(AnalyticFeature f) const { return settings.analytical.enabled && settings.analytical.has(f); }
    bool geometryNeeded() const;
    NamedStageTiming& analyticalTiming(const char* name);
    void setAnalyticalTiming(const char* name, double elapsed_ms, bool ran);
    void analyticalSpatial();
    void denseHog();
    void denseMotion(double timestamp);
    void scatterDenseFlow();
    void geometricFeatures(double timestamp);
#ifdef HAVE_VISUAL_MOTION
    void updateRelativePose(const MotionEstimate& motion);
    void resetMapSegment(const MotionEstimate& motion);
    void pruneLocalMap(double timestamp, const MotionEstimate& motion);
    void addNewLandmarks(double timestamp, const MotionEstimate& motion);
    std::vector<ColoredPoint> localCloud();
    bool buildTerrain(const std::vector<ColoredPoint>& cloud);
    cv::Mat terrainConfidence(double timestamp, const MotionEstimate& motion);
    void projectGeometry(const cv::Mat& confidence, const MotionEstimate& motion);
#endif
    void packTensor();
    void reduceChannel(int channel, cv::Mat& values, cv::Mat& valid);
    void normalizeChannel(int channel, const cv::Mat& source, cv::Mat& destination);
    template<class F> void stage(const char* name, F operation) {
        auto& timing = analyticalTiming(name);
        timing.ran = true;
        if(!settings.profiling){operation();return;}
        const auto start = std::chrono::steady_clock::now();
        operation();
        timing.elapsed_ms += std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    template<class F> void run(VisualStage stage, F operation) {
        auto& timing = output.stages[std::size_t(stage)];
        timing.ran = true;
        if (!settings.profiling) { operation(); return; }
        const auto start = std::chrono::steady_clock::now();
        operation();
        timing.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
};
} // namespace metric_mapping

// --- Extractor ---

namespace metric_mapping {
const char* visualFeatureName(VisualFeature feature)
{
    static constexpr const char* names[] = {"gradients", "edge_magnitude", "edge_orientation",
        "corners", "keypoints", "color_transitions", "texture_orientation", "local_binary_texture",
        "contours", "shape_boundaries", "motion"};
    if (std::size_t(feature) >= visual_feature_count) throw std::runtime_error("Invalid visual feature");
    return names[std::size_t(feature)];
}
const char* visualStageName(VisualStage stage)
{
    static constexpr const char* names[] = {"preparation", "derivatives", "edges", "structure_tensor",
        "corners", "keypoints", "color_transitions", "local_binary_texture", "boundaries", "motion"};
    if (std::size_t(stage) >= visual_stage_count) throw std::runtime_error("Invalid visual stage");
    return names[std::size_t(stage)];
}
VisualFeatureSelection selectVisualFeatures(const std::string& names)
{
    if (names == "all") return VisualFeatureSelection{}.set();
    if (names == "none") return {};
    if (names.empty() || names.back() == ',') throw std::runtime_error("Empty feature selection");
    VisualFeatureSelection selected;
    std::istringstream stream(names); std::string name;
    while (std::getline(stream, name, ',')) {
        std::size_t i = 0;
        for (; i < visual_feature_count; ++i) if (name == visualFeatureName(VisualFeature(i))) break;
        if (i == visual_feature_count) throw std::runtime_error("Unknown visual feature: " + name);
        selected.set(i);
    }
    return selected;
}

VisualFeatureExtractor::Impl::Impl(const VisualFeatureSettings& config, std::optional<CameraIntrinsics> camera)
    : settings(config), source_camera(camera)
{
    const auto& s = settings;
    for (double v : {double(s.corner_quality), double(s.point_spacing_px), double(s.minimum_gradient),
                     double(s.minimum_texture_energy), double(s.minimum_texture_coherence),
                     s.canny_low, s.canny_high, s.contour_epsilon_px})
        if (!std::isfinite(v) || v <= 0) throw std::runtime_error("Visual settings must be positive and finite");
    if (s.maximum_width < 32 || s.maximum_height < 32 || s.maximum_width > 1024 || s.maximum_height > 1024 ||
        s.maximum_points < 1 || s.maximum_points > 1000 || s.grid_columns < 1 || s.grid_columns > 16 ||
        s.grid_rows < 1 || s.grid_rows > 16 || s.corner_quality >= 1 || s.point_spacing_px > 100 ||
        s.fast_threshold < 1 || s.fast_threshold > 255 || s.tensor_window < 3 || s.tensor_window > 15 ||
        s.tensor_window % 2 == 0 || s.minimum_texture_coherence > 1 || s.canny_low >= s.canny_high ||
        s.maximum_contours < 1 || s.maximum_contours > 1000 || s.minimum_contour_points < 2 ||
        s.maximum_contour_points < s.minimum_contour_points || s.maximum_contour_points > 100000)
        throw std::runtime_error("Invalid visual feature settings");
    if (camera) validateCamera(*camera);
    detail::validateAnalyticalSettings(settings);
    if(!s.distortion.empty()&&!camera)throw std::runtime_error("Distortion correction requires calibration");
    output.requested = settings.features;
    output.corners.reserve(s.maximum_points); output.keypoints.reserve(s.maximum_points);
    cells.resize(s.grid_rows * s.grid_columns);
}
void VisualFeatureExtractor::Impl::prepare(const cv::Mat& bgr)
{
    const auto size = settings.analytical.enabled ? settings.analytical.working_size :
        detail::fitImageSize(bgr.size(), settings.maximum_width, settings.maximum_height);
    if (size.width < 16 || size.height < 16) throw std::runtime_error("Visual image is too small or narrow");
    if(!settings.distortion.empty()) {
        if(remap_x.empty()) {
            const auto k=*source_camera, scaled=detail::resizeCamera(k,size);
            const cv::Matx33d original(k.fx,0,k.cx,0,k.fy,k.cy,0,0,1), resized(scaled.fx,0,scaled.cx,0,scaled.fy,scaled.cy,0,0,1);
            cv::initUndistortRectifyMap(original,settings.distortion,cv::Mat(),resized,size,CV_32FC1,remap_x,remap_y);
        }
        cv::remap(bgr,output.bgr,remap_x,remap_y,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
    }
    else if (bgr.size() == size) bgr.copyTo(output.bgr);
    else cv::resize(bgr, output.bgr, size, 0, 0, cv::INTER_AREA);
    if(analytic(AnalyticFeature::Appearance)||analytic(AnalyticFeature::Chroma))stage("color",[&]{
        output.bgr.convertTo(prepared,CV_32F,1.0/255);
        // OpenCV stores Y,Cr,Cb, so explicitly reorder the public bank to Y,Cb,Cr.
        cv::cvtColor(prepared,ycrcb,cv::COLOR_BGR2YCrCb);
        for(int i=0;i<3;++i)cv::extractChannel(ycrcb,color_planes[i],i);
        color_planes[0].convertTo(output.gray,CV_8U,255);
    });
    else cv::cvtColor(output.bgr, output.gray, cv::COLOR_BGR2GRAY);
    output.source_size = bgr.size(); output.working_size = size;
    output.pixel_to_source = detail::pixelToSource(bgr.size(), size);
    if (source_camera) output.camera = detail::resizeCamera(*source_camera, size);
}
void VisualFeatureExtractor::Impl::motion(double timestamp, std::optional<AltitudeSample> altitude, std::optional<ImuSample> imu)
{
    if (!source_camera) { output.motion_status = "unavailable: calibrated temporal input required"; return; }
#ifdef HAVE_VISUAL_MOTION
    if (!tracker) {
        auto config = settings.motion;
        config.maximum_width = output.working_size.width;
        config.maximum_height = output.working_size.height;
        if(geometryNeeded()) {config.triangulation=true;config.retain_correspondences=true;}
        config.profiling=settings.profiling;
        tracker = std::make_unique<SparseFlowTracker>(*output.camera, config);
    }
    output.motion = tracker->processFrame(output.gray, timestamp, altitude, imu);
    output.motion_status = output.motion->status;
    output.tracks = tracker->tracks();
    output.computed.set(std::size_t(VisualFeature::Motion));
#else
    (void)timestamp; (void)altitude; (void)imu;
    output.motion_status = "unavailable: BUILD_MOTION is disabled";
#endif
}

VisualFeatureExtractor::VisualFeatureExtractor(const VisualFeatureSettings& settings,
                                               std::optional<CameraIntrinsics> camera)
    : impl_(std::make_unique<Impl>(settings, camera)) {}
VisualFeatureExtractor::~VisualFeatureExtractor() = default;
void VisualFeatureExtractor::resetMotion()
{
    impl_->frame_seen = false;
    impl_->previous_dense.release();impl_->output.local_map.clear();impl_->seen_landmarks.clear();
#ifdef HAVE_VISUAL_MOTION
    if (impl_->tracker) impl_->tracker->reset();
#endif
}
void VisualFeatureExtractor::Impl::validateFrame(const cv::Mat& bgr, double timestamp) const
{
    if (bgr.empty() || bgr.type() != CV_8UC3 || !std::isfinite(timestamp))
        throw std::runtime_error("Visual extraction requires BGR8 and a finite timestamp");
    if (source_camera && bgr.size() != cv::Size(source_camera->width, source_camera->height))
        throw std::runtime_error("Visual input size does not match calibration");
    if ((settings.analytical.enabled || (settings.has(VisualFeature::Motion) && source_camera)) && frame_seen && timestamp <= last_timestamp)
        throw std::runtime_error("Visual motion requires increasing timestamps");
}

void VisualFeatureExtractor::Impl::extractSpatialFeatures()
{
    // Decide dependencies once. Turning off a feature also skips unused work.
    using Feature = VisualFeature;
    const bool a_derivative=analytic(AnalyticFeature::Gradients)||analytic(AnalyticFeature::Hog)||analytic(AnalyticFeature::Harris)||
        analytic(AnalyticFeature::Canny)||analytic(AnalyticFeature::Contours);
    const bool need_edges = settings.has(Feature::EdgeMagnitude) || settings.has(Feature::EdgeOrientation);
    const bool need_tensor = settings.has(Feature::Corners) || settings.has(Feature::TextureOrientation);
    const bool need_boundaries = settings.has(Feature::Contours) || settings.has(Feature::ShapeBoundaries);

    if (settings.has(Feature::Gradients) || need_edges || need_tensor || need_boundaries || a_derivative)
        run(VisualStage::Derivatives, [&] { derivatives(); });
    if (need_edges || analytic(AnalyticFeature::Gradients)||analytic(AnalyticFeature::Hog)) run(VisualStage::Edges, [&] { edges(); });
    if (need_tensor || analytic(AnalyticFeature::Harris)) run(VisualStage::StructureTensor, [&] { structureTensor(); });
    if (settings.has(Feature::Corners)) run(VisualStage::Corners, [&] { corners(); });
    if (settings.has(Feature::Keypoints)) run(VisualStage::Keypoints, [&] { keypoints(); });
    if (settings.has(Feature::ColorTransitions)) run(VisualStage::ColorTransitions, [&] { colorTransitions(); });
    if (settings.has(Feature::LocalBinaryTexture)) run(VisualStage::LocalBinaryTexture, [&] { binaryTexture(); });
    if (need_boundaries || analytic(AnalyticFeature::Canny)||analytic(AnalyticFeature::Contours)) run(VisualStage::Boundaries, [&] { boundaries(); });

    auto spatial = settings.features;
    spatial.reset(std::size_t(Feature::Motion));
    output.computed |= spatial;
}

const VisualFeatureFrame& VisualFeatureExtractor::extract(const cv::Mat& bgr, double timestamp,
                                                         std::optional<AltitudeSample> altitude, std::optional<ImuSample> imu)
{
    auto& state = *impl_;
    auto& result = state.output;
    state.validateFrame(bgr, timestamp);
    const auto start = state.settings.profiling ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    result.stages = {};
    result.total_ms = 0;
    result.computed.reset();
    result.timestamp_s = timestamp;
    if(state.settings.analytical.enabled) {
        result.analytical={};result.relative_pose={};result.pose_correspondences.clear();result.terrain={};result.surface={};
        result.geometry_confidence.release();result.analytical_stages.clear();
        for(const char* name:{"preparation","color","sobel","magnitude","hog","harris","canny","contours","chroma",
            "dense_flow","sparse_tracking","pose","triangulation","map_update","voxel","terrain_grid","idw","projection",
            "depth_gradients","terrain_gradients","slope","diffusion","roughness","resizing","packing"})
            result.analytical_stages.push_back({name,false,0});
        for(auto& p:state.bank)p.release();for(auto& p:state.bank_masks)p.release();
    }

    state.run(VisualStage::Preparation, [&] {
        if(state.settings.analytical.enabled)state.stage("preparation",[&]{state.prepare(bgr);});
        else state.prepare(bgr);
    });
    if(state.settings.analytical.enabled) {
        result.analytical_stages[0].elapsed_ms=std::max(0.0,result.analytical_stages[0].elapsed_ms-result.analytical_stages[1].elapsed_ms);
    }
    state.extractSpatialFeatures();
    if (state.settings.has(VisualFeature::Motion)||state.geometryNeeded())
        state.run(VisualStage::Motion, [&] { state.motion(timestamp, altitude, imu); });
    if(state.settings.analytical.enabled) {
        state.analyticalSpatial();state.denseMotion(timestamp);state.geometricFeatures(timestamp);state.packTensor();
    }

    ++result.frame_id;
    state.frame_seen = true;
    state.last_timestamp = timestamp;
    if (state.settings.profiling)
        result.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace metric_mapping

// --- Appearance ---

namespace metric_mapping {
void VisualFeatureExtractor::Impl::derivatives()
{
    // Sobel / 8 estimates the derivative; /255 fixes intensity units across frames.
    const auto calculate=[&] {
        if(settings.analytical.enabled)cv::GaussianBlur(output.gray,smooth,{settings.analytical.gaussian_kernel,settings.analytical.gaussian_kernel},settings.analytical.gaussian_sigma);
        const auto& input=settings.analytical.enabled?smooth:output.gray;
        cv::Sobel(input, gx, CV_32F, 1, 0, 3, 1.0 / (8 * 255), 0, cv::BORDER_REFLECT_101);
        cv::Sobel(input, gy, CV_32F, 0, 1, 3, 1.0 / (8 * 255), 0, cv::BORDER_REFLECT_101);
    };
    if(settings.analytical.enabled)stage("sobel",calculate);else calculate();
    if (settings.has(VisualFeature::Gradients)) { output.gradient_x = gx; output.gradient_y = gy; }
}
void VisualFeatureExtractor::Impl::edges()
{
    if(settings.analytical.enabled)stage("magnitude",[&]{cv::magnitude(gx,gy,magnitude);});
    else cv::magnitude(gx, gy, magnitude);
    if (settings.has(VisualFeature::EdgeMagnitude)) output.edge_magnitude = magnitude;
    if (settings.has(VisualFeature::EdgeOrientation)) {
        if(analytic(AnalyticFeature::Hog)) {
            cv::phase(gx,gy,hog_angle,false);hog_angle.copyTo(output.edge_orientation_rad);
        } else cv::phase(gx, gy, output.edge_orientation_rad, false); // [0,2pi), clockwise in image coordinates.
        cv::compare(magnitude, settings.minimum_gradient, output.edge_orientation_valid, cv::CMP_GE);
        output.edge_orientation_valid.row(0).setTo(0); output.edge_orientation_valid.row(output.gray.rows - 1).setTo(0);
        output.edge_orientation_valid.col(0).setTo(0); output.edge_orientation_valid.col(output.gray.cols - 1).setTo(0);
        output.edge_orientation_rad.setTo(0, output.edge_orientation_valid == 0);
    }
}
void VisualFeatureExtractor::Impl::structureTensor()
{
    const auto started=std::chrono::steady_clock::now();
    cv::multiply(gx, gx, jxx); cv::multiply(gx, gy, jxy); cv::multiply(gy, gy, jyy);
    const cv::Size window(settings.tensor_window, settings.tensor_window);
    for (auto* plane : {&jxx, &jxy, &jyy}) cv::boxFilter(*plane, *plane, -1, window, {-1, -1}, true, cv::BORDER_REFLECT_101);
    const bool corners = settings.has(VisualFeature::Corners), texture = settings.has(VisualFeature::TextureOrientation);
    if (corners) output.corner_response.create(output.gray.size(), CV_32F);
    if(analytic(AnalyticFeature::Harris))harris.create(output.gray.size(),CV_32F);
    if (texture) {
        output.texture_orientation_rad.create(output.gray.size(), CV_32F);
        output.texture_coherence.create(output.gray.size(), CV_32F);
        output.texture_orientation_valid.create(output.gray.size(), CV_8U);
    }
    const int border = settings.tensor_window / 2 + 1;
    for (int y = 0; y < output.gray.rows; ++y) {
        const auto* a = jxx.ptr<float>(y); const auto* b = jxy.ptr<float>(y); const auto* c = jyy.ptr<float>(y);
        auto* response = corners ? output.corner_response.ptr<float>(y) : nullptr;
        auto* orientation = texture ? output.texture_orientation_rad.ptr<float>(y) : nullptr;
        auto* coherence = texture ? output.texture_coherence.ptr<float>(y) : nullptr;
        auto* valid = texture ? output.texture_orientation_valid.ptr<std::uint8_t>(y) : nullptr;
        for (int x = 0; x < output.gray.cols; ++x) {
            const float trace = a[x] + c[x], difference = a[x] - c[x];
            const float discriminant = (corners || texture) ? std::sqrt(difference * difference + 4 * b[x] * b[x]) : 0;
            if (corners) response[x] = std::max(0.0F, (trace - discriminant) * 0.5F);
            if(analytic(AnalyticFeature::Harris))harris.at<float>(y,x)=
                a[x]*c[x]-b[x]*b[x]-float(settings.analytical.harris_k)*trace*trace;
            if (texture) {
                coherence[x] = trace > settings.minimum_texture_energy ? std::min(1.0F, discriminant / trace) : 0;
                valid[x] = trace > settings.minimum_texture_energy && coherence[x] >= settings.minimum_texture_coherence &&
                    x >= border && y >= border && x < output.gray.cols - border && y < output.gray.rows - border ? 255 : 0;
                // Principal gradient direction + pi/2 is the texture tangent, modulo pi.
                float angle = valid[x] ? 0.5F * std::atan2(2 * b[x], difference) + float(CV_PI / 2) : 0;
                if (angle >= float(CV_PI)) angle -= float(CV_PI);
                orientation[x] = angle;
            }
        }
    }
    if(analytic(AnalyticFeature::Harris)) {
        auto& t=*std::find_if(output.analytical_stages.begin(),output.analytical_stages.end(),[](const auto& t){return t.name=="harris";});
        t.ran=true;if(settings.profiling)t.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
}
void VisualFeatureExtractor::Impl::colorTransitions()
{
    opponent_rg.create(output.gray.size(), CV_32F); opponent_by.create(output.gray.size(), CV_32F);
    for (int y = 0; y < output.bgr.rows; ++y) {
        const auto* color = output.bgr.ptr<cv::Vec3b>(y);
        auto* rg = opponent_rg.ptr<float>(y); auto* by = opponent_by.ptr<float>(y);
        for (int x = 0; x < output.bgr.cols; ++x) {
            rg[x] = (float(color[x][2]) - color[x][1]) / 255;
            by[x] = (float(color[x][0]) - 0.5F * (color[x][2] + color[x][1])) / 255;
        }
    }
    color_energy.create(output.gray.size(), CV_32F); color_energy.setTo(0);
    for (const auto* channel : {&opponent_rg, &opponent_by}) {
        cv::Sobel(*channel, dx, CV_32F, 1, 0, 3, 1.0 / 8);
        cv::Sobel(*channel, dy, CV_32F, 0, 1, 3, 1.0 / 8);
        cv::multiply(dx, dx, temporary); cv::add(color_energy, temporary, color_energy);
        cv::multiply(dy, dy, temporary); cv::add(color_energy, temporary, color_energy);
    }
    cv::sqrt(color_energy, output.color_transition_magnitude);
}
void VisualFeatureExtractor::Impl::binaryTexture()
{
    output.local_binary_texture.create(output.gray.size(), CV_8U); output.local_binary_texture.setTo(0);
    output.local_binary_valid.create(output.gray.size(), CV_8U); output.local_binary_valid.setTo(0);
    // Clockwise bits: NW,N,NE,E,SE,S,SW,W. Equality sets a bit. Codes are categorical.
    for (int y = 1; y < output.gray.rows - 1; ++y) {
        auto* codes = output.local_binary_texture.ptr<std::uint8_t>(y);
        auto* valid = output.local_binary_valid.ptr<std::uint8_t>(y);
        const auto* center = output.gray.ptr<std::uint8_t>(y);
        const auto* above = output.gray.ptr<std::uint8_t>(y - 1);
        const auto* below = output.gray.ptr<std::uint8_t>(y + 1);
        for (int x = 1; x < output.gray.cols - 1; ++x) {
            const auto value = center[x];
            const unsigned code = unsigned(above[x-1] >= value) |
                (unsigned(above[x] >= value) << 1) | (unsigned(above[x+1] >= value) << 2) |
                (unsigned(center[x+1] >= value) << 3) | (unsigned(below[x+1] >= value) << 4) |
                (unsigned(below[x] >= value) << 5) | (unsigned(below[x-1] >= value) << 6) |
                (unsigned(center[x-1] >= value) << 7);
            codes[x] = std::uint8_t(code); valid[x] = 255;
        }
    }
}
} // namespace metric_mapping

// --- Structure ---

namespace metric_mapping {
void VisualFeatureExtractor::Impl::selectPoints(std::vector<cv::KeyPoint>& destination)
{
    destination.clear(); std::fill(cells.begin(), cells.end(), 0);
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.response != b.response) return a.response > b.response;
        if (a.pt.y != b.pt.y) return a.pt.y < b.pt.y;
        return a.pt.x < b.pt.x;
    });
    const int quota = (settings.maximum_points + int(cells.size()) - 1) / int(cells.size());
    for (const auto& point : candidates) {
        const int col = std::clamp(int(point.pt.x * settings.grid_columns / output.gray.cols), 0, settings.grid_columns - 1);
        const int row = std::clamp(int(point.pt.y * settings.grid_rows / output.gray.rows), 0, settings.grid_rows - 1);
        const int cell = row * settings.grid_columns + col;
        if (cells[cell] >= quota) continue;
        bool close = false;
        for (const auto& kept : destination) {
            const auto delta = point.pt - kept.pt;
            if (delta.dot(delta) < settings.point_spacing_px * settings.point_spacing_px) { close = true; break; }
        }
        if (close) continue;
        destination.push_back(point); ++cells[cell];
        if (destination.size() == std::size_t(settings.maximum_points)) break;
    }
}
void VisualFeatureExtractor::Impl::corners()
{
    candidates.clear();
    double maximum = 0; cv::minMaxLoc(output.corner_response, nullptr, &maximum);
    if (maximum <= 1e-12) { output.corners.clear(); return; }
    cv::dilate(output.corner_response, peaks, cv::Mat());
    const float threshold = float(maximum * settings.corner_quality);
    const int border = settings.tensor_window / 2 + 1;
    for (int y = border; y < output.gray.rows - border; ++y) {
        const auto* response = output.corner_response.ptr<float>(y); const auto* peak = peaks.ptr<float>(y);
        for (int x = border; x < output.gray.cols - border; ++x)
            if (response[x] >= threshold && response[x] == peak[x])
                candidates.emplace_back(float(x), float(y), float(settings.tensor_window), -1, response[x]);
    }
    selectPoints(output.corners);
}
void VisualFeatureExtractor::Impl::keypoints()
{
    // OpenCV owns growth of this temporary vector; don't pass a reserved/cleared
    // instrumented STL vector across the packaged binary's ASan boundary.
    std::vector<cv::KeyPoint> detected;
    cv::FAST(output.gray, detected, settings.fast_threshold, true);
    candidates.assign(detected.begin(), detected.end());
    selectPoints(output.keypoints);
}
void VisualFeatureExtractor::Impl::boundaries()
{
    // Reuse the shared Sobel derivatives; Canny expects raw signed 16-bit units.
    const auto canny=[&]{gx.convertTo(canny_dx,CV_16S,8*255);gy.convertTo(canny_dy,CV_16S,8*255);
        cv::Canny(canny_dx,canny_dy,edge_mask,settings.canny_low,settings.canny_high,true);};
    if(settings.analytical.enabled)stage("canny",canny);else canny();
    if(!settings.has(VisualFeature::Contours)&&!settings.has(VisualFeature::ShapeBoundaries)&&!analytic(AnalyticFeature::Contours))return;
    const auto trace=[&]{
        cv::findContours(edge_mask,raw_contours,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
        if(analytic(AnalyticFeature::Contours)) {
            contour_map.create(output.gray.size(),CV_8U);contour_map.setTo(0);
            cv::drawContours(contour_map,raw_contours,-1,cv::Scalar(255),1);
        }
    };
    if(settings.analytical.enabled)stage("contours",trace);else trace();
    contour_order.resize(raw_contours.size()); std::iota(contour_order.begin(), contour_order.end(), 0);
    std::sort(contour_order.begin(), contour_order.end(), [&](std::size_t a, std::size_t b) {
        return raw_contours[a].size() > raw_contours[b].size();
    });
    output.contours.clear(); output.contours_truncated = false;
    if (settings.has(VisualFeature::ShapeBoundaries)) {
        output.shape_boundaries.create(output.gray.size(), CV_8U); output.shape_boundaries.setTo(0);
    }
    int retained = 0, points = 0;
    std::vector<cv::Point> polygon;
    for (const auto index : contour_order) {
        const auto& contour = raw_contours[index];
        if (contour.size() < std::size_t(settings.minimum_contour_points)) continue;
        if (retained >= settings.maximum_contours || points + contour.size() > std::size_t(settings.maximum_contour_points)) {
            output.contours_truncated = true; continue;
        }
        ++retained; points += int(contour.size());
        if (settings.has(VisualFeature::Contours)) output.contours.push_back(contour);
        if (settings.has(VisualFeature::ShapeBoundaries)) {
            cv::approxPolyDP(contour, polygon, settings.contour_epsilon_px, true);
            cv::polylines(output.shape_boundaries, polygon, true, cv::Scalar(255), 1);
        }
    }
}
} // namespace metric_mapping

// --- Representation ---

namespace metric_mapping {
std::vector<VisualFeaturePlane> visualFeaturePlanes(const VisualFeatureFrame& f)
{
    std::vector<VisualFeaturePlane> planes;
    const auto add = [&](const char* name, const char* units, const cv::Mat& values, const cv::Mat& mask = cv::Mat()) {
        if (!values.empty()) planes.push_back({name, units, values, mask});
    };
    add("gradient_x", "normalized_intensity/working_pixel", f.gradient_x);
    add("gradient_y", "normalized_intensity/working_pixel", f.gradient_y);
    add("edge_magnitude", "normalized_intensity/working_pixel", f.edge_magnitude);
    add("edge_orientation", "radians_[0,2pi)_image_normal", f.edge_orientation_rad, f.edge_orientation_valid);
    add("corner_response", "normalized_gradient_squared_min_eigenvalue", f.corner_response);
    add("color_transitions", "opponent_color/working_pixel", f.color_transition_magnitude);
    add("texture_orientation", "radians_[0,pi)_image_tangent", f.texture_orientation_rad, f.texture_orientation_valid);
    add("texture_coherence", "unitless_[0,1]", f.texture_coherence);
    add("local_binary_texture", "categorical_8bit_code", f.local_binary_texture, f.local_binary_valid);
    add("shape_boundaries", "binary_0_or_255", f.shape_boundaries);
    return planes;
}
std::size_t visualFeaturePayloadBytes(const VisualFeatureFrame& f)
{
    std::size_t bytes = f.bgr.total() * f.bgr.elemSize() + f.gray.total() * f.gray.elemSize();
    for (const auto& p : visualFeaturePlanes(f)) bytes += p.values.total() * p.values.elemSize() + p.valid.total() * p.valid.elemSize();
    bytes += (f.corners.size() + f.keypoints.size()) * sizeof(cv::KeyPoint) + f.tracks.size() * sizeof(TrackedFeature);
    for (const auto& c : f.contours) bytes += c.size() * sizeof(cv::Point);
    if (f.motion) bytes += sizeof(MotionEstimate);
    return bytes; // Logical raster/vector payload only; excludes scratch, metadata and capacities.
}
void writeVisualFeatures(const std::filesystem::path& path, const VisualFeatureFrame& f,
                         const VisualFeatureSettings& s)
{
    cv::FileStorage file(path.string(), cv::FileStorage::WRITE);
    if (!file.isOpened()) throw std::runtime_error("Could not write visual feature data: " + path.string());
    file << "schema_version" << f.schema_version << "frame_id" << std::to_string(f.frame_id)
         << "timestamp_s" << f.timestamp_s << "source_size" << f.source_size << "working_size" << f.working_size
         << "pixel_to_source" << cv::Mat(f.pixel_to_source) << "channel_order" << "BGR"
         << "requested" << "[";
    for (std::size_t i = 0; i < visual_feature_count; ++i) if (f.requested[i]) file << visualFeatureName(VisualFeature(i));
    file << "]" << "computed" << "[";
    for (std::size_t i = 0; i < visual_feature_count; ++i) if (f.computed[i]) file << visualFeatureName(VisualFeature(i));
    file << "]" << "settings" << "{" << "maximum_width" << s.maximum_width << "maximum_height" << s.maximum_height
         << "maximum_points" << s.maximum_points << "grid_columns" << s.grid_columns << "grid_rows" << s.grid_rows
         << "corner_quality" << s.corner_quality << "point_spacing_px" << s.point_spacing_px
         << "fast_threshold" << s.fast_threshold << "tensor_window" << s.tensor_window
         << "minimum_gradient" << s.minimum_gradient << "minimum_texture_energy" << s.minimum_texture_energy
         << "minimum_texture_coherence" << s.minimum_texture_coherence << "canny_low" << s.canny_low
         << "canny_high" << s.canny_high << "contour_epsilon_px" << s.contour_epsilon_px
         << "minimum_contour_points" << s.minimum_contour_points << "maximum_contours" << s.maximum_contours
         << "maximum_contour_points" << s.maximum_contour_points << "}"
         << "bgr" << f.bgr << "gray" << f.gray << "planes" << "[";
    for (const auto& p : visualFeaturePlanes(f))
        file << "{" << "name" << p.name << "units" << p.units << "values" << p.values << "valid" << p.valid << "}";
    file << "]" << "corners" << f.corners << "keypoints" << f.keypoints << "contours" << "[";
    for (const auto& c : f.contours) file << c;
    file << "]" << "contours_truncated" << int(f.contours_truncated) << "motion_status" << f.motion_status;
    if (f.camera) file << "camera" << "{" << "fx" << f.camera->fx << "fy" << f.camera->fy
        << "cx" << f.camera->cx << "cy" << f.camera->cy << "width" << f.camera->width << "height" << f.camera->height << "}";
    file << "tracks" << "[";
    for (const auto& t : f.tracks) file << "{" << "id" << std::to_string(t.id) << "age" << double(t.age)
        << "previous" << t.previous_px << "current" << t.current_px << "}";
    file << "]";
    if (f.motion) {
        const auto& m = *f.motion;
        file << "motion" << "{" << "valid" << int(m.valid) << "dt_s" << m.dt_s << "flow" << m.median_flow_px
             << "yaw_delta_rad" << m.yaw_delta_rad << "image_scale" << m.image_scale
             << "quality" << m.quality << "coverage" << m.coverage
             << "inlier_fraction" << m.inlier_fraction << "residual_px" << m.residual_px
             << "tracked_points" << double(m.tracked_points)
             << "accepted_points" << double(m.accepted_points)
             << "active_tracks" << double(m.active_tracks)
             << "status" << m.status << "metric_status" << m.metric_status
             << "imu_status" << m.imu_status << "imu_used" << int(m.imu_used);
        if (m.planar_displacement_m) file << "displacement_m" << *m.planar_displacement_m << "velocity_mps" << *m.planar_velocity_mps;
        file << "}";
    }
    file << "total_ms" << f.total_ms << "logical_payload_bytes" << double(visualFeaturePayloadBytes(f)) << "stages" << "[";
    for (std::size_t i = 0; i < visual_stage_count; ++i) file << "{" << "name" << visualStageName(VisualStage(i))
        << "ran" << int(f.stages[i].ran) << "ms" << f.stages[i].elapsed_ms << "}";
    file << "]";
    if(s.analytical.enabled) {
        const auto& d=f.analytical;
        file << "analytical" << "{" << "layout" << (s.analytical.batch_dimension?"NCHW":"CHW")
             << "tensor" << f.feature_tensor.values << "valid" << f.feature_tensor.valid
             << "channel_names" << f.feature_tensor.channel_names
             << "calibration_valid" << int(d.calibration_valid) << "scale_valid" << int(d.scale_valid)
             << "geometry_valid" << int(d.geometry_valid) << "geometry_status" << d.geometry_status
             << "dense_flow_valid" << int(d.dense_flow_valid) << "geometry_coverage_percent" << d.geometry_coverage_percent
             << "sparse_tracks" << double(d.num_sparse_tracks) << "valid_tracks" << double(d.num_valid_tracks)
             << "pose_inliers" << double(d.num_pose_inliers) << "triangulated_points" << double(d.num_triangulated_points)
             << "map_points" << double(d.local_map_point_count) << "mean_reprojection_error_px" << d.mean_reprojection_error
             << "payload_bytes" << double(d.payload_bytes) << "stages" << "[";
        for(const auto& t:f.analytical_stages)file << "{" << "name" << t.name << "ran" << int(t.ran) << "ms" << t.elapsed_ms << "}";
        file << "]" << "relative_pose_valid" << int(f.relative_pose.valid)
             << "relative_pose_scale_valid" << int(f.relative_pose.scale_valid)
             << "relative_rotation_21" << cv::Mat(f.relative_pose.rotation_21)
             << "relative_translation_direction" << f.relative_pose.translation_direction
             << "relative_points_unit_baseline" << f.relative_pose.relative_points
             << "flow_scale_px" << s.analytical.flow_scale_px << "depth_scale_m" << s.analytical.depth_scale_m
             << "depth_gradient_scale" << s.analytical.depth_gradient_scale << "roughness_scale_m" << s.analytical.roughness_scale_m
             << "harris_scale" << s.analytical.harris_scale << "pose_correspondences" << "[";
        for(std::size_t i=0;i<f.pose_correspondences.size();++i) {
            const auto& p=f.pose_correspondences[i];
            file << "{" << "track_id" << std::to_string(p.id) << "previous" << p.previous_px
                 << "current" << p.current_px << "inlier" << int(f.relative_pose.inliers[i]) << "}";
        }
        file << "]" << "local_map" << "[";
        for(const auto& p:f.local_map)file << "{" << "xyz_m" << cv::Vec3f(p.point.x,p.point.y,p.point.z)
            << "rgb" << cv::Vec3i(p.point.r,p.point.g,p.point.b) << "track_id" << std::to_string(p.track_id)
            << "timestamp_s" << p.timestamp_s << "observations" << double(p.observations)
            << "confidence" << p.confidence << "reprojection_error_px" << p.reprojection_error_px << "}";
        file << "]";
        // Detection registration consumes this exact local-map camera pose.  It
        // uses the same nadir convention as geometry projection; absence means
        // that 2-D observations must remain unresolved rather than invent XYZ.
        if(f.motion && f.motion->pose)
            file << "camera_to_local_map" << cv::Mat(detail::nadirTransform(*f.motion->pose));
        file << "}";
    }
    file.release();
}

void writeAnalyticalTensor(const std::filesystem::path& path, const VisualFeatureFrame& f,
                           const VisualFeatureSettings& s)
{
    if(!s.analytical.enabled)
        throw std::runtime_error("Analytical tensor export requires the analytical bank");
    cv::FileStorage file(path.string(), cv::FileStorage::WRITE);
    if(!file.isOpened()) throw std::runtime_error("Could not write analytical tensor: " + path.string());
    const auto& d=f.analytical;
    file << "schema_version" << f.schema_version << "timestamp_s" << f.timestamp_s
         << "analytical" << "{" << "tensor" << f.feature_tensor.values
         << "valid" << f.feature_tensor.valid << "channel_names" << f.feature_tensor.channel_names
         << "geometry_valid" << int(d.geometry_valid)
         << "dense_flow_valid" << int(d.dense_flow_valid)
         << "depth_scale_m" << s.analytical.depth_scale_m;
    if(f.motion && f.motion->pose)
        file << "camera_to_local_map" << cv::Mat(detail::nadirTransform(*f.motion->pose));
    file << "}";
    file.release();
}
} // namespace metric_mapping

// --- Primary analytical bank: one extractor, one result, deterministic channels ---
namespace metric_mapping {
namespace {
constexpr const char* channel_names[]={"Y","Cb","Cr","Gx","Gy","GradientMagnitude",
    "HOG_0","HOG_1","HOG_2","HOG_3","HOG_4","HOG_5","HOG_6","HOG_7","HOG_8",
    "HarrisResponse","CannyEdge","ContourMap","ChromaGradientCb","ChromaGradientCr",
    "OpticalFlowU","OpticalFlowV","Depth","DepthGradientX","DepthGradientY","Slope","Roughness","GeometryConfidence"};
constexpr AnalyticFeature channel_family[]={AnalyticFeature::Appearance,AnalyticFeature::Appearance,AnalyticFeature::Appearance,
    AnalyticFeature::Gradients,AnalyticFeature::Gradients,AnalyticFeature::Gradients,
    AnalyticFeature::Hog,AnalyticFeature::Hog,AnalyticFeature::Hog,AnalyticFeature::Hog,AnalyticFeature::Hog,
    AnalyticFeature::Hog,AnalyticFeature::Hog,AnalyticFeature::Hog,AnalyticFeature::Hog,
    AnalyticFeature::Harris,AnalyticFeature::Canny,AnalyticFeature::Contours,AnalyticFeature::Chroma,AnalyticFeature::Chroma,
    AnalyticFeature::DenseFlow,AnalyticFeature::DenseFlow,AnalyticFeature::Depth,AnalyticFeature::DepthGradients,
    AnalyticFeature::DepthGradients,AnalyticFeature::Slope,AnalyticFeature::Roughness,AnalyticFeature::GeometryConfidence};
}
cv::Mat AnalyticalTensor::plane(std::size_t c) const
{
    if(c>=channel_names.size()||values.empty())throw std::out_of_range("Tensor channel");
    const int h=values.size[values.dims-2],w=values.size[values.dims-1];
    return cv::Mat(h,w,CV_32F,const_cast<float*>(values.ptr<float>()+c*h*w));
}
cv::Mat AnalyticalTensor::mask(std::size_t c) const
{
    if(c>=channel_names.size()||valid.empty())throw std::out_of_range("Tensor mask");
    const int h=valid.size[valid.dims-2],w=valid.size[valid.dims-1];
    return cv::Mat(h,w,CV_8U,const_cast<unsigned char*>(valid.ptr<unsigned char>()+c*h*w));
}
bool VisualFeatureExtractor::Impl::geometryNeeded() const
{
    if(!settings.analytical.enabled||!settings.analytical.enable_geometry)return false;
    for(auto f:{AnalyticFeature::Depth,AnalyticFeature::DepthGradients,AnalyticFeature::Slope,
        AnalyticFeature::Roughness,AnalyticFeature::GeometryConfidence})if(analytic(f))return true;
    return false;
}
NamedStageTiming& VisualFeatureExtractor::Impl::analyticalTiming(const char* name)
{
    const auto timing = std::find_if(
        output.analytical_stages.begin(), output.analytical_stages.end(),
        [name](const NamedStageTiming& item) { return item.name == name; });
    if (timing == output.analytical_stages.end())
        throw std::logic_error(std::string("Unknown analytical timing stage: ") + name);
    return *timing;
}
void VisualFeatureExtractor::Impl::setAnalyticalTiming(
    const char* name, double elapsed_ms, bool ran)
{
    auto& timing = analyticalTiming(name);
    timing.ran = ran;
    timing.elapsed_ms = elapsed_ms;
}
void VisualFeatureExtractor::Impl::analyticalSpatial()
{
    if(analytic(AnalyticFeature::Appearance)){
        bank[0]=color_planes[0];bank[1]=color_planes[2];bank[2]=color_planes[1];
    }
    if(analytic(AnalyticFeature::Gradients)){bank[3]=gx;bank[4]=gy;bank[5]=magnitude;}
    if(analytic(AnalyticFeature::Harris))bank[15]=harris;
    if(analytic(AnalyticFeature::Canny)){
        edge_mask.convertTo(analytical_canny,CV_32F,1.0/255);bank[16]=analytical_canny;
    }
    if(analytic(AnalyticFeature::Contours)){
        contour_map.convertTo(analytical_contours,CV_32F,1.0/255);bank[17]=analytical_contours;
    }
    if(analytic(AnalyticFeature::Chroma))stage("chroma",[&]{
        for(int i=0;i<2;++i){
            cv::Sobel(color_planes[2-i],dx,CV_32F,1,0,3,1.0/8);
            cv::Sobel(color_planes[2-i],dy,CV_32F,0,1,3,1.0/8);
            cv::magnitude(dx,dy,i==0?chroma_cb:chroma_cr);bank[18+i]=i==0?chroma_cb:chroma_cr;
        }
    });
    if(analytic(AnalyticFeature::Hog))stage("hog",[&]{denseHog();});
}
void VisualFeatureExtractor::Impl::denseHog()
{
    const auto grid = settings.analytical.grid_size;
    if(!settings.has(VisualFeature::EdgeOrientation))
        cv::phase(gx, gy, hog_angle, false);
    for(auto& plane : hog) {
        plane.create(grid, CV_32F);
        plane.setTo(0);
    }

    for(int y = 0; y < gx.rows; ++y) {
        const float* angles = hog_angle.ptr<float>(y);
        const float* strengths = magnitude.ptr<float>(y);
        const int cell_y = y * grid.height / gx.rows;
        for(int x = 0; x < gx.cols; ++x) {
            float angle = angles[x];
            if(angle >= float(CV_PI)) angle -= float(CV_PI);
            const float bin = angle * float(9 / CV_PI);
            const int lower = std::min(8, int(bin));
            const int upper = (lower + 1) % 9;
            const float upper_weight = bin - lower;
            const int cell_x = x * grid.width / gx.cols;
            hog[lower].at<float>(cell_y, cell_x) += strengths[x] * (1 - upper_weight);
            hog[upper].at<float>(cell_y, cell_x) += strengths[x] * upper_weight;
        }
    }

    // Normalize each cell independently so every bin remains image-aligned.
    for(int y = 0; y < grid.height; ++y) {
        for(int x = 0; x < grid.width; ++x) {
            float squared_norm = 1e-12F;
            for(const auto& plane : hog) {
                const float value = plane.at<float>(y, x);
                squared_norm += value * value;
            }
            const float norm = std::sqrt(squared_norm);
            for(auto& plane : hog) plane.at<float>(y, x) /= norm;
        }
    }
    for(int bin = 0; bin < 9; ++bin) bank[6 + bin] = hog[bin];
}
void VisualFeatureExtractor::Impl::denseMotion(double timestamp)
{
    if(!analytic(AnalyticFeature::DenseFlow)) return;
    const auto& analytical = settings.analytical;
    // Shared smoothed luminance when derivatives ran; otherwise prepare it here.
    if(!output.analytical_stages[2].ran)
        cv::GaussianBlur(output.gray, smooth,
            {analytical.gaussian_kernel, analytical.gaussian_kernel},
            analytical.gaussian_sigma);
    for(auto* plane : {&dense_u, &dense_v, &dense_counts}) {
        plane->create(output.working_size, CV_32F);
        plane->setTo(0);
    }
    dense_valid.create(output.working_size, CV_8U);
    dense_valid.setTo(0);
    bank[20] = dense_u;
    bank[21] = dense_v;
    bank_masks[20] = dense_valid;
    bank_masks[21] = dense_valid;

    const bool compatible_previous = !previous_dense.empty() &&
        previous_dense.size() == smooth.size();
    const bool recent_previous = timestamp - last_timestamp <=
        settings.motion.maximum_frame_gap_s;
    if(compatible_previous && recent_previous) {
        stage("dense_flow", [&] {
            cv::calcOpticalFlowFarneback(
                previous_dense, smooth, dense_flow,
                analytical.flow_pyramid_scale, analytical.flow_levels,
                analytical.flow_window, analytical.flow_iterations,
                analytical.flow_poly_n, analytical.flow_poly_sigma, 0);
            scatterDenseFlow();
        });
    }
    smooth.copyTo(previous_dense);
}
void VisualFeatureExtractor::Impl::scatterDenseFlow()
{
    // Farneback's forward flow lives at old pixels. Move each vector to its
    // current endpoint so it aligns with the current-frame feature planes.
    for(int y = 0; y < dense_flow.rows; ++y) {
        for(int x = 0; x < dense_flow.cols; ++x) {
            const auto flow = dense_flow.at<cv::Vec2f>(y, x);
            const double current_x = x + double(flow[0]);
            const double current_y = y + double(flow[1]);
            const bool inside = std::isfinite(current_x) && std::isfinite(current_y) &&
                current_x >= 0 && current_y >= 0 &&
                current_x <= dense_flow.cols - 1 && current_y <= dense_flow.rows - 1;
            if(!inside) continue;
            const int target_x = int(std::floor(current_x + 0.5));
            const int target_y = int(std::floor(current_y + 0.5));
            dense_u.at<float>(target_y, target_x) += flow[0];
            dense_v.at<float>(target_y, target_x) += flow[1];
            dense_counts.at<float>(target_y, target_x) += 1;
        }
    }
    for(int y = 0; y < dense_counts.rows; ++y) {
        for(int x = 0; x < dense_counts.cols; ++x) {
            const float count = dense_counts.at<float>(y, x);
            if(count == 0) continue;
            dense_u.at<float>(y, x) /= count;
            dense_v.at<float>(y, x) /= count;
            dense_valid.at<unsigned char>(y, x) = 255;
        }
    }
    output.analytical.dense_flow_valid = cv::countNonZero(dense_valid) > 0;
}
void VisualFeatureExtractor::Impl::geometricFeatures(double timestamp)
{
    auto& diagnostics = output.analytical;
    diagnostics.calibration_valid = source_camera.has_value();
    if(!geometryNeeded()) {
        diagnostics.geometry_status = "disabled";
        return;
    }
    if(!output.camera) {
        diagnostics.geometry_status = "calibration_unavailable";
        return;
    }
#ifndef HAVE_VISUAL_MOTION
    (void)timestamp;
    diagnostics.geometry_status = "sparse_motion_not_built";
#else
    if(!tracker || !output.motion) {
        diagnostics.geometry_status = "tracking_unavailable";
        return;
    }

    const auto& motion = *output.motion;
    updateRelativePose(motion);
    diagnostics.scale_valid = motion.valid &&
        motion.planar_displacement_m.has_value() && motion.pose.has_value();
    diagnostics.num_triangulated_points = motion.new_landmarks;
    resetMapSegment(motion);
    if(!diagnostics.scale_valid) {
        diagnostics.geometry_status = motion.metric_status;
        return;
    }

    pruneLocalMap(timestamp, motion);
    addNewLandmarks(timestamp, motion);
    diagnostics.local_map_point_count = output.local_map.size();
    diagnostics.num_valid_3d_points = output.local_map.size();
    if(output.local_map.empty()) {
        diagnostics.geometry_status = "insufficient_parallax_or_support";
        return;
    }

    const auto cloud = localCloud();
    if(!buildTerrain(cloud)) return;
    const cv::Mat confidence = terrainConfidence(timestamp, motion);
    projectGeometry(confidence, motion);
#endif
}
#ifdef HAVE_VISUAL_MOTION
void VisualFeatureExtractor::Impl::updateRelativePose(const MotionEstimate& motion)
{
    auto& diagnostics = output.analytical;
    const auto& analytical = settings.analytical;
    diagnostics.num_sparse_tracks = motion.tracked_points;
    diagnostics.num_valid_tracks = tracker->correspondences().size();
    setAnalyticalTiming("sparse_tracking",
        motion.timing.preprocessing_ms + motion.timing.detection_ms +
        motion.timing.flow_ms + motion.timing.filtering_ms, true);
    setAnalyticalTiming("pose", motion.timing.geometry_ms, motion.tracked_points > 0);
    setAnalyticalTiming("triangulation", motion.timing.triangulation_ms,
        motion.triangulation_attempts > 0);

    const bool pose_frame = output.frame_id % std::uint64_t(analytical.pose_interval) == 0;
    const bool enough_points = tracker->correspondences().size() >=
        std::size_t(analytical.relative_pose.minimum_inliers);
    if(pose_frame && enough_points) {
        output.pose_correspondences = tracker->correspondences();
        output.relative_pose = estimateRelativePose(
            output.pose_correspondences, *output.camera, analytical.relative_pose);
        const double pose_ms = settings.profiling ? output.relative_pose.pose_ms : 0;
        const double triangulation_ms = settings.profiling ?
            output.relative_pose.triangulation_ms : 0;
        setAnalyticalTiming("pose", motion.timing.geometry_ms + pose_ms, true);
        setAnalyticalTiming("triangulation",
            motion.timing.triangulation_ms + triangulation_ms,
            motion.triangulation_attempts > 0 || triangulation_ms > 0);
    }
    diagnostics.num_pose_inliers = output.relative_pose.valid ?
        std::count(output.relative_pose.inliers.begin(), output.relative_pose.inliers.end(), 1) :
        motion.accepted_points;
}
void VisualFeatureExtractor::Impl::resetMapSegment(const MotionEstimate& motion)
{
    if(motion.pose && map_segment == motion.segment_id) return;
    output.local_map.clear();
    seen_landmarks.clear();
    map_segment = motion.segment_id;
}
void VisualFeatureExtractor::Impl::pruneLocalMap(
    double timestamp, const MotionEstimate& motion)
{
    const auto& analytical = settings.analytical;
    stage("map_update", [&] {
        output.local_map.erase(std::remove_if(
            output.local_map.begin(), output.local_map.end(),
            [&](const LocalMapPoint& point) {
                const auto& p = point.point;
                const bool expired = timestamp - point.timestamp_s > analytical.map_age_s;
                const bool distant = cv::norm(
                    cv::Vec3d(p.x, p.y, p.z) - motion.pose->position_m) >
                    analytical.map_radius_m;
                return expired || distant;
            }), output.local_map.end());
    });
}
void VisualFeatureExtractor::Impl::addNewLandmarks(
    double timestamp, const MotionEstimate& motion)
{
    const auto& analytical = settings.analytical;
    stage("voxel", [&] {
        const auto world_to_camera = invertRigidTransform(detail::nadirTransform(*motion.pose));
        for(const auto& landmark : tracker->landmarks()) {
            if(std::find(seen_landmarks.begin(), seen_landmarks.end(), landmark.track_id) !=
               seen_landmarks.end()) continue;

            const auto& world = landmark.point.world_m;
            if(cv::norm(world - motion.pose->position_m) > analytical.map_radius_m) continue;
            const auto camera_point = transformPoint(world_to_camera, world);
            if(camera_point[2] <= 0) continue;

            const auto& camera = *output.camera;
            const double u = camera.fx * camera_point[0] / camera_point[2] + camera.cx;
            const double v = camera.fy * camera_point[1] / camera_point[2] + camera.cy;
            if(!std::isfinite(u) || !std::isfinite(v) ||
               u < 0 || v < 0 || u >= camera.width || v >= camera.height) continue;

            const auto color = output.bgr.at<cv::Vec3b>(int(v), int(u));
            LocalMapPoint item;
            item.point = {float(world[0]), float(world[1]), float(world[2]),
                          color[2], color[1], color[0]};
            item.track_id = landmark.track_id;
            item.timestamp_s = timestamp;
            item.reprojection_error_px = landmark.point.reprojection_error_px;
            item.confidence = motion.quality /
                (1 + item.reprojection_error_px /
                    settings.motion.triangulation_limits.maximum_reprojection_error_px);

            const auto same_voxel = [&](const LocalMapPoint& old) {
                const double size = analytical.voxel_size_m;
                return std::floor(old.point.x / size) == std::floor(item.point.x / size) &&
                       std::floor(old.point.y / size) == std::floor(item.point.y / size) &&
                       std::floor(old.point.z / size) == std::floor(item.point.z / size);
            };
            auto existing = std::find_if(
                output.local_map.begin(), output.local_map.end(), same_voxel);
            if(existing == output.local_map.end()) {
                output.local_map.push_back(item);
                continue;
            }

            VoxelGridAccumulator merged(analytical.voxel_size_m, 2);
            merged.add(existing->point);
            merged.add(item.point);
            const auto points = merged.points();
            if(points.size() == 1) item.point = points.front();
            item.observations = existing->observations + 1;
            item.confidence = std::min(item.confidence, existing->confidence);
            *existing = item;
        }

        seen_landmarks.clear();
        for(const auto& landmark : tracker->landmarks())
            seen_landmarks.push_back(landmark.track_id);
        if(output.local_map.size() > analytical.maximum_map_points) {
            const auto excess = output.local_map.size() - analytical.maximum_map_points;
            output.local_map.erase(output.local_map.begin(), output.local_map.begin() + excess);
        }
    });
}
std::vector<ColoredPoint> VisualFeatureExtractor::Impl::localCloud()
{
    std::vector<ColoredPoint> cloud;
    cloud.reserve(output.local_map.size());
    double total_error = 0;
    for(const auto& point : output.local_map) {
        cloud.push_back(point.point);
        total_error += point.reprojection_error_px;
    }
    output.analytical.mean_reprojection_error = total_error / output.local_map.size();
    return cloud;
}
bool VisualFeatureExtractor::Impl::buildTerrain(const std::vector<ColoredPoint>& cloud)
{
    const auto& analytical = settings.analytical;
    const auto bounds = computeBounds(cloud);
    const double columns = std::ceil(
        (bounds.maximum[0] - bounds.minimum[0]) / analytical.grid_resolution_m) + 1;
    const double rows = std::ceil(
        (bounds.maximum[1] - bounds.minimum[1]) / analytical.grid_resolution_m) + 1;
    if(columns * rows > analytical.maximum_grid_cells) {
        output.analytical.geometry_status = "grid_budget_exceeded";
        return false;
    }

    stage("terrain_grid", [&] {
        output.terrain = createTerrainGrid(
            cloud, analytical.grid_resolution_m, analytical.maximum_grid_cells);
    });
    if(analytical.idw.enabled)
        stage("idw", [&] { interpolateIdw(output.terrain, analytical.idw); });

    auto surface_settings = analytical.surface;
    surface_settings.gradients = analytic(AnalyticFeature::Slope);
    surface_settings.slope = analytic(AnalyticFeature::Slope);
    surface_settings.roughness = analytic(AnalyticFeature::Roughness);
    output.surface = measureSurface(output.terrain, surface_settings);
    setAnalyticalTiming("terrain_gradients", output.surface.gradients_ms,
        surface_settings.gradients);
    setAnalyticalTiming("slope", output.surface.slope_ms, surface_settings.slope);
    setAnalyticalTiming("diffusion", output.surface.diffusion_ms,
        surface_settings.roughness && surface_settings.diffusion_iterations > 0);
    setAnalyticalTiming("roughness", output.surface.roughness_ms,
        surface_settings.roughness);
    return true;
}
cv::Mat VisualFeatureExtractor::Impl::terrainConfidence(
    double timestamp, const MotionEstimate& motion)
{
    const auto& analytical = settings.analytical;
    cv::Mat confidence(output.terrain.height, output.terrain.width, CV_32F, cv::Scalar(0));
    // This is a support score rather than a calibrated probability.
    stage("map_update", [&] {
        const auto& grid = output.terrain;
        for(int y = 0; y < grid.height; ++y) {
            for(int x = 0; x < grid.width; ++x) {
                const auto index = grid.index(y, x);
                if(!grid.validity[index]) continue;
                const double world_x = grid.minimum_x_m + x * grid.resolution_m;
                const double world_y = grid.maximum_y_m - y * grid.resolution_m;
                double nearest = std::numeric_limits<double>::infinity();
                double quality = 0;
                for(const auto& point : output.local_map) {
                    const double distance = std::hypot(
                        point.point.x - world_x, point.point.y - world_y);
                    if(distance >= nearest) continue;
                    nearest = distance;
                    const double age = std::max(
                        0.0, 1 - (timestamp - point.timestamp_s) / analytical.map_age_s);
                    quality = point.confidence * age;
                }
                const double measured_weight = grid.validity[index] == 255 ? 1.0 : 0.5;
                confidence.at<float>(y, x) = float(
                    std::min(quality, motion.quality) * measured_weight /
                    (1 + nearest / grid.resolution_m));
            }
        }
    });
    return confidence;
}
void VisualFeatureExtractor::Impl::projectGeometry(
    const cv::Mat& confidence, const MotionEstimate& motion)
{
    const auto& analytical = settings.analytical;
    detail::ProjectedGeometry projected;
    stage("projection", [&] {
        projected = detail::projectSurface(
            output.terrain, output.surface, confidence, *output.camera,
            detail::nadirTransform(*motion.pose), analytical.grid_size,
            settings.motion.triangulation_limits.maximum_depth_m, false);
    });
    if(analytic(AnalyticFeature::DepthGradients)) {
        stage("depth_gradients", [&] {
            detail::depthGradients(
                projected, detail::resizeCamera(*output.camera, analytical.grid_size));
        });
    }
    for(int i = 0; i < 6; ++i) {
        bank[22 + i] = projected.planes[i];
        bank_masks[22 + i] = projected.masks[i];
    }
    output.geometry_confidence = projected.planes[5];
    auto& diagnostics = output.analytical;
    diagnostics.geometry_coverage_percent =
        100.0 * cv::countNonZero(projected.masks[0]) / analytical.grid_size.area();
    diagnostics.geometry_valid = diagnostics.geometry_coverage_percent > 0;
    diagnostics.geometry_status = diagnostics.geometry_valid ?
        "metric_nadir_local_map" : "outside_current_view";
}
#endif
void VisualFeatureExtractor::Impl::packTensor()
{
    const auto& analytical = settings.analytical;
    auto& tensor = output.feature_tensor;
    tensor.channel_names.clear();
    tensor.channel_valid.clear();
    for(int channel = 0; channel < 28; ++channel) {
        if(analytic(channel_family[channel]))
            tensor.channel_names.emplace_back(channel_names[channel]);
    }
    if(tensor.channel_names.empty()) {
        tensor.values.release();
        tensor.valid.release();
        return;
    }

    const int shape[] = {1, int(tensor.channel_names.size()),
                         analytical.grid_size.height, analytical.grid_size.width};
    const bool batched = analytical.batch_dimension;
    tensor.values.create(batched ? 4 : 3, shape + (batched ? 0 : 1), CV_32F);
    tensor.valid.create(batched ? 4 : 3, shape + (batched ? 0 : 1), CV_8U);

    std::array<cv::Mat, 28> reduced;
    std::array<cv::Mat, 28> masks;
    stage("resizing", [&] {
        for(int channel = 0; channel < 28; ++channel) {
            if(analytic(channel_family[channel]))
                reduceChannel(channel, reduced[channel], masks[channel]);
        }
    });
    stage("packing", [&] {
        std::size_t packed_channel = 0;
        for(int channel = 0; channel < 28; ++channel) {
            if(!analytic(channel_family[channel])) continue;
            auto destination = tensor.plane(packed_channel);
            auto valid = tensor.mask(packed_channel);
            if(masks[channel].empty()) valid.setTo(255);
            else masks[channel].copyTo(valid);
            normalizeChannel(channel, reduced[channel], destination);
            destination.setTo(0, valid == 0);
            tensor.channel_valid.push_back(cv::countNonZero(valid) > 0);
            ++packed_channel;
        }
    });
    output.analytical.payload_bytes =
        tensor.values.total() * sizeof(float) + tensor.valid.total() +
        output.local_map.size() * sizeof(LocalMapPoint);
}
void VisualFeatureExtractor::Impl::reduceChannel(
    int channel, cv::Mat& values, cv::Mat& valid)
{
    const auto grid = settings.analytical.grid_size;
    if(bank[channel].empty()) {
        values = cv::Mat::zeros(grid, CV_32F);
        valid = cv::Mat::zeros(grid, CV_8U);
        if(channel == 27) valid.setTo(255); // A missing confidence score is a known zero.
        return;
    }
    if(bank[channel].size() == grid) {
        values = bank[channel];
        valid = bank_masks[channel];
        return;
    }
    if(bank_masks[channel].empty()) {
        cv::resize(bank[channel], values, grid, 0, 0, cv::INTER_AREA);
        if(channel == 16 || channel == 17) {
            // One contributing edge is enough for the output cell to be occupied.
            cv::compare(values, 0, values, cv::CMP_GT);
            values.convertTo(values, CV_32F, 1.0 / 255);
        }
        return;
    }

    cv::Mat weights, weighted, support;
    bank_masks[channel].convertTo(weights, CV_32F, 1.0 / 255);
    cv::multiply(bank[channel], weights, weighted);
    cv::resize(weighted, values, grid, 0, 0, cv::INTER_AREA);
    cv::resize(weights, support, grid, 0, 0, cv::INTER_AREA);
    cv::compare(support, 0.5, valid, cv::CMP_GE);
    cv::max(support, 1e-6, support);
    cv::divide(values, support, values);
    values.setTo(0, valid == 0);
}
void VisualFeatureExtractor::Impl::normalizeChannel(
    int channel, const cv::Mat& source, cv::Mat& destination)
{
    const auto& analytical = settings.analytical;
    double scale = 1;
    double minimum = 0;
    if(channel == 3 || channel == 4) { scale = 2; minimum = -1; }
    if(channel == 5 || channel == 18 || channel == 19) scale = std::sqrt(2.0);
    if(channel == 15) { scale = 1 / analytical.harris_scale; minimum = -1; }
    if(channel == 20 || channel == 21) {
        scale = 1 / analytical.flow_scale_px;
        minimum = -1;
    }
    if(channel == 22) scale = 1 / analytical.depth_scale_m;
    if(channel == 23 || channel == 24) {
        scale = 1 / analytical.depth_gradient_scale;
        minimum = -1;
    }
    if(channel == 25) scale = 2 / CV_PI;
    if(channel == 26) scale = 1 / analytical.roughness_scale_m;

    if(channel == 15) {
        // A fixed signed square root compresses Harris response without using
        // the current frame's strongest corner as a scale.
        for(int y = 0; y < destination.rows; ++y) {
            for(int x = 0; x < destination.cols; ++x) {
                const float response = source.at<float>(y, x);
                destination.at<float>(y, x) = float(std::copysign(
                    std::sqrt(std::abs(response)) * scale, response));
            }
        }
    } else {
        source.convertTo(destination, CV_32F, scale);
    }
    cv::max(destination, minimum, destination);
    cv::min(destination, 1, destination);
}
} // namespace metric_mapping
