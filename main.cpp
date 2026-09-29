#include "src/internal.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_MOTION_CAMERA
#include <opencv2/videoio.hpp>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// ============================================================================
// Two View
// ============================================================================

#ifdef HAVE_TWO_VIEW

namespace two_view_command {

namespace {
double parseNumber(const char* text, const std::string& name)
{
    std::size_t consumed = 0;
    const std::string value(text);
    double number = 0.0;
    try {
        number = std::stod(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(name + " must be a number, not '" + value +
                                 "'");
    }
    if (consumed != value.size() || !std::isfinite(number))
        throw std::runtime_error(name + " must be a finite number");
    return number;
}

} // namespace

int run(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--demo") {
        try {
            metric_mapping::app::runSyntheticDemo(argv[2], argv[3]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 8 && argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " image1 image2 fx fy cx cy baseline_m [--ultrasonic sensor.yaml]\n"
                  << "   or: " << argv[0]
                  << " --demo image1 image2\n";
        return 1;
    }

    try {
        std::optional<metric_mapping::UltrasonicMeasurement> ultrasonic;
        if (argc == 10) {
            if (std::string(argv[8]) != "--ultrasonic")
                throw std::runtime_error("Expected --ultrasonic sensor.yaml");
            ultrasonic = metric_mapping::loadUltrasonicConfig(argv[9]);
        }
        const std::filesystem::path image_1_path(argv[1]);
        const std::filesystem::path image_2_path(argv[2]);
        const cv::Mat probe = cv::imread(image_1_path.string(),
                                         cv::IMREAD_COLOR);
        if (probe.empty())
            throw std::runtime_error("Could not load " +
                                     image_1_path.string());

        metric_mapping::CameraIntrinsics camera;
        camera.width = probe.cols;
        camera.height = probe.rows;
        camera.fx = parseNumber(argv[3], "fx");
        camera.fy = parseNumber(argv[4], "fy");
        camera.cx = parseNumber(argv[5], "cx");
        camera.cy = parseNumber(argv[6], "cy");
        const double baseline_m = parseNumber(argv[7], "baseline_m");

        metric_mapping::app::runStereo(image_1_path, image_2_path, camera, baseline_m, ultrasonic);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
}
#endif

// ============================================================================
// Metric Mapper
// ============================================================================

#ifdef HAVE_RGBD

namespace metric_mapper_command {

int run(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " configs/reconstruction.yaml\n";
        return 1;
    }
    try {
        metric_mapping::app::runRgbd(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
}
#endif

// ============================================================================
// Motion Benchmark
// ============================================================================

#ifdef HAVE_MOTION

namespace motion_benchmark_command {

using namespace metric_mapping;
int run(int argc, char** argv)
{
    try {
        OpticalFlowSettings settings; settings.profiling = true;
        int frames = 500, speed = 1;
        std::filesystem::path debug;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--triangulate") settings.triangulation = true;
            else if (arg == "--imu") settings.require_imu = true;
            else if (arg == "--forward-backward") settings.forward_backward = true;
            else if (arg == "--frames" && i + 1 < argc) frames = detail::parseInteger(argv[++i]);
            else if (arg == "--levels" && i + 1 < argc) settings.pyramid_levels = detail::parseInteger(argv[++i]);
            else if (arg == "--speed" && i + 1 < argc) speed = detail::parseInteger(argv[++i]);
            else if (arg == "--debug" && i + 1 < argc) debug = argv[++i];
            else throw std::runtime_error("Usage: motion_benchmark [--frames N] [--levels 0|1|2] [--speed 1..20] [--forward-backward] [--triangulate] [--imu] [--debug directory]");
        }
        if (frames < 30 || frames > 1000000) throw std::runtime_error("Frames must be in [30,1000000]");
        if (speed < 1 || speed > 20) throw std::runtime_error("Speed must be in [1,20]");
        // Library does not modify process-wide threading; runner explicitly uses one CPU thread.
        cv::setNumThreads(1);
        const CameraIntrinsics camera{320, 240, 240, 230, 159.5, 119.5};
        SparseFlowTracker tracker(camera, settings);
        const auto texture = demo::motionTexture();
        cv::Mat frame;
        std::vector<double> durations; durations.reserve(frames);
        MotionTimings sum;
        double cpu_ms = 0, error_sum = 0, initialization_ms = 0, detection_ms = 0;
        std::size_t detection_runs = 0;
        std::size_t valid = 0, tracked = 0, inliers = 0, replenished = 0, fallbacks = 0, landmarks = 0;
        constexpr int warmup = 20;
        MotionEstimate latest;
        for (int i = 0; i < frames + warmup; ++i) {
            const double dx = 18 * std::sin(0.045 * i * speed), dy = 12 * std::sin(0.031 * i * speed);
            const double yaw = 0.03 * std::sin(0.017 * i * speed);
            demo::warpMotion(texture, frame, camera, dx, dy, yaw);
            const auto cpu_start = std::clock();
            const auto imu = settings.require_imu
                ? std::optional<ImuSample>{ImuSample{0, 0, yaw, i * 0.1}} : std::nullopt;
            latest = tracker.processFrame(frame, i * 0.1, AltitudeSample{2.0, i * 0.1}, imu);
            const double cpu = 1000.0 * (std::clock() - cpu_start) / CLOCKS_PER_SEC;
            if (i == 0) initialization_ms = latest.timing.total_ms;
            if (latest.detection_ran) { ++detection_runs; detection_ms += latest.timing.detection_ms; }
            if (i < warmup) continue;
            cpu_ms += cpu; durations.push_back(latest.timing.total_ms);
            sum.preprocessing_ms += latest.timing.preprocessing_ms;
            sum.detection_ms += latest.timing.detection_ms; sum.flow_ms += latest.timing.flow_ms;
            sum.filtering_ms += latest.timing.filtering_ms; sum.geometry_ms += latest.timing.geometry_ms;
            sum.triangulation_ms += latest.timing.triangulation_ms;
            tracked += latest.tracked_points; inliers += latest.accepted_points;
            replenished += latest.replenished_points; fallbacks += latest.consensus_fallback;
            landmarks += latest.new_landmarks;
            if (latest.valid && latest.planar_displacement_m) {
                ++valid;
                // Ground truth camera center: -h D R(yaw)^T [dx/fx,dy/fy].
                const auto position = [&](int j) {
                    const double a = 0.03 * std::sin(0.017 * j * speed);
                    const double x = 18 * std::sin(0.045 * j * speed) / camera.fx;
                    const double y = 12 * std::sin(0.031 * j * speed) / camera.fy;
                    return cv::Vec2d(-2 * (std::cos(a) * x + std::sin(a) * y),
                                     2 * (-std::sin(a) * x + std::cos(a) * y));
                };
                const auto world_delta = position(i) - position(i - 1);
                const double previous_yaw = 0.03 * std::sin(0.017 * (i - 1) * speed);
                const cv::Vec2d expected(std::cos(previous_yaw) * world_delta[0] + std::sin(previous_yaw) * world_delta[1],
                                        -std::sin(previous_yaw) * world_delta[0] + std::cos(previous_yaw) * world_delta[1]);
                error_sum += cv::norm(*latest.planar_displacement_m - expected);
            }
        }
        const auto timing = detail::summarizeTimes(durations);
        std::cout << std::fixed << std::setprecision(4)
            << "Synthetic 320x240, OpenCV " << CV_VERSION << ", one thread, " << frames << " measured frames, " << warmup << " warmup\n"
            << "LK window=" << settings.window_size << " levels=" << settings.pyramid_levels
            << " iterations=" << settings.lk_iterations << " FB=" << settings.forward_backward
            << " IMU=" << settings.require_imu << " speed=" << speed << '\n'
            << "Processing ms mean/median/p95/worst: " << timing.mean << " / " << timing.median
            << " / " << timing.p95 << " / " << timing.worst << '\n'
            << "CPU ms/frame: " << cpu_ms / frames << "; processing capacity FPS: " << 1000 / timing.mean << '\n'
            << "Stage mean ms preprocess/detect/LK/filter/geometry/triangulate: "
            << sum.preprocessing_ms / frames << " / " << sum.detection_ms / frames << " / " << sum.flow_ms / frames
            << " / " << sum.filtering_ms / frames << " / " << sum.geometry_ms / frames << " / " << sum.triangulation_ms / frames << '\n'
            << "Mean tracked/inliers/replenished: " << double(tracked)/frames << " / " << double(inliers)/frames << " / " << double(replenished)/frames << '\n'
            << "Valid: " << valid << '/' << frames << "; consensus fallbacks: " << fallbacks << "; new landmarks: " << landmarks << '\n'
            << "Initialization ms: " << initialization_ms << "; detection runs (including warmup): " << detection_runs
            << "; mean detection ms/run: " << detection_ms / std::max<std::size_t>(1, detection_runs) << '\n'
            << "Mean metric displacement error (m): " << std::setprecision(7) << (valid ? error_sum / valid : -1) << '\n'
            << "Last median flow: " << latest.median_flow_px << "; yaw: " << latest.yaw_delta_rad
            << "; quality: " << latest.quality << "; active/rejected: " << latest.active_tracks << '/' << latest.tracked_points - latest.accepted_points << '\n';
        if (latest.planar_velocity_mps) std::cout << "Last velocity (m/s): " << *latest.planar_velocity_mps << '\n';
        if (!debug.empty()) {
            std::filesystem::create_directories(debug);
            if (!cv::imwrite((debug / "tracks.png").string(), tracker.debugImage(latest)))
                throw std::runtime_error("Could not write debug image");
        }
        // Correctness smoke check, never a flaky hardware-dependent timing limit.
        return valid >= 0.95 * frames && error_sum / std::max<std::size_t>(1, valid) < 0.003 ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
}
#endif

// ============================================================================
// Motion Camera
// ============================================================================

#ifdef HAVE_MOTION_CAMERA

namespace motion_camera_command {

using namespace metric_mapping;
namespace {
double number(const cv::FileNode& node, const char* key) {
    const auto field = node[key];
    if (!field.isInt() && !field.isReal()) throw std::runtime_error(std::string("Missing numeric camera field: ") + key);
    const double value = double(field);
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite calibration");
    return value;
}


}
int runCamera(int argc, char** argv, bool pixels_only)
{
    try {
        const bool pixel_arguments = pixels_only;
        std::string camera_path, imu_path, ultrasonic_path;
        std::vector<char*> positional{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--camera" || option == "--imu" || option == "--ultrasonic") {
                if (i + 1 >= argc || !argv[i + 1][0] || std::string(argv[i + 1]).rfind("--", 0) == 0)
                    throw std::runtime_error("Missing file path after " + option);
                auto& path = option == "--camera" ? camera_path : option == "--imu" ? imu_path : ultrasonic_path;
                if (!path.empty()) throw std::runtime_error("Repeated option: " + option);
                path = argv[++i];
            } else positional.push_back(argv[i]);
        }
        argc = int(positional.size());
        argv = positional.data();
        if (!pixel_arguments && !camera_path.empty())
            throw std::runtime_error("motion_camera already takes calibration as its first argument");
        if (pixel_arguments && camera_path.empty() && (!imu_path.empty() || !ultrasonic_path.empty()))
            throw std::runtime_error("Sensor-assisted optical_flow requires --camera with measured calibration");
        if ((pixels_only && argc > 4) || (!pixels_only && (argc < 2 || argc > 6)))
            throw std::runtime_error(pixels_only
                ? "Usage: optical_flow [device=0] [frames=300] [debug_directory] [--camera camera.yaml] [--imu imu.txt] [--ultrasonic range.txt]"
                : "Usage: motion_camera camera.yaml [device=0] [frames=300] [range_file|-] [debug_directory] [--imu imu.txt] [--ultrasonic range.txt]");
        OpticalFlowSettings settings;
        settings.profiling = true;
        settings.require_imu = !imu_path.empty();
        pixels_only = pixel_arguments && camera_path.empty();
        cv::Size requested_size(320, 240);
        std::unique_ptr<SparseFlowTracker> tracker;
        std::unique_ptr<PixelFlowTracker> pixel_tracker;
        if (!pixels_only) {
            cv::FileStorage config(pixel_arguments ? camera_path : argv[1], cv::FileStorage::READ);
            if (!config.isOpened()) throw std::runtime_error("Could not open camera YAML");
            const auto k = config["camera"];
            const double width = number(k, "width"), height = number(k, "height");
            if (width < 64 || height < 64 || width > 4096 || height > 4096 ||
                width != std::trunc(width) || height != std::trunc(height))
                throw std::runtime_error("Camera size must be integral and within [64,4096]");
            const CameraIntrinsics camera{int(width), int(height), number(k, "fx"), number(k, "fy"),
                                          number(k, "cx"), number(k, "cy")};
            requested_size = {camera.width, camera.height};
            tracker = std::make_unique<SparseFlowTracker>(camera, settings);
        }
        const int device_index = pixel_arguments ? 1 : 2;
        const int device = argc > device_index ? detail::parseInteger(argv[device_index], 0) : 0;
        const int count = argc > device_index + 1
            ? detail::parseInteger(argv[device_index + 1], 1, 1000000) : 300;
        const std::string positional_range = !pixel_arguments && argc > 4 && std::string(argv[4]) != "-" ? argv[4] : "";
        if (!positional_range.empty() && !ultrasonic_path.empty())
            throw std::runtime_error("Supply the ultrasonic file once, either positional or --ultrasonic");
        const std::string range_path = ultrasonic_path.empty() ? positional_range : ultrasonic_path;
        const int debug_index = pixel_arguments ? 3 : 5;
        const std::filesystem::path debug = argc > debug_index ? argv[debug_index] : "";
        if (!debug.empty()) std::filesystem::create_directories(debug);
        cv::setNumThreads(1);
        cv::VideoCapture capture(device);
        if (!capture.isOpened()) throw std::runtime_error("Could not open camera; check its device index and camera permission");
        capture.set(cv::CAP_PROP_FRAME_WIDTH, requested_size.width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, requested_size.height);
        capture.set(cv::CAP_PROP_FPS, 10);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        std::cerr << "Requesting camera " << device << " at " << requested_size << " and 10 FPS. "
                  << (pixels_only ? "Image-space optical flow only; no metric pose."
                                  : "Calibrated motion; metric scale needs fresh altitude.")
                  << " Timestamp clock: steady_clock seconds since epoch.\n";
        if (pixels_only)
            std::cout << "timestamp_s,valid,status,width_px,height_px,active,tracked,inliers,flow_u_px,flow_v_px,velocity_u_px_s,velocity_v_px_s,quality,processing_ms\n";
        else
            std::cout << "timestamp_s,valid,status,metric_status,active,tracked,inliers,rejected,replenished,flow_u,flow_v,yaw_rad,dx_m,dy_m,vx_mps,vy_mps,quality,processing_ms,processing_fps,imu_used,imu_status\n";
        cv::Mat frame;
        for (int i = 0; i < count; ++i) {
            if (!capture.read(frame)) throw std::runtime_error("Camera capture failed");
            const double timestamp = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (pixels_only) {
                // A camera may ignore the requested resolution. Use the delivered
                // dimensions for 2-D tracking; the tracker still downsamples to its budget.
                if (!pixel_tracker) pixel_tracker = std::make_unique<PixelFlowTracker>(frame.size(), settings);
                const auto result = pixel_tracker->processFrame(frame, timestamp);
                const auto size = pixel_tracker->workingSize();
                std::cout << std::fixed << std::setprecision(6) << timestamp << ',' << result.valid << ','
                    << result.status << ',' << size.width << ',' << size.height << ',' << result.active_tracks << ','
                    << result.tracked_points << ',' << result.accepted_points << ','
                    << result.median_flow_px.x << ',' << result.median_flow_px.y << ',';
                if (result.velocity_px_s)
                    std::cout << result.velocity_px_s->x << ',' << result.velocity_px_s->y;
                else std::cout << ',';
                std::cout << ',' << result.quality << ',' << result.processing_ms << '\n' << std::flush;
                if (!debug.empty() && i % 10 == 0 &&
                    !cv::imwrite((debug / ("tracks_" + std::to_string(i) + ".png")).string(), pixel_tracker->debugImage()))
                    throw std::runtime_error("Could not write optical-flow debug image");
                continue;
            }
            const auto result = tracker->processFrame(frame, timestamp, detail::readAltitudeSnapshot(range_path),
                                                       detail::readImuSnapshot(imu_path));
            std::cout << std::fixed << std::setprecision(6) << timestamp << ',' << result.valid << ','
                << result.status << ',' << result.metric_status << ',' << result.active_tracks << ','
                << result.tracked_points << ',' << result.accepted_points << ','
                << result.tracked_points - result.accepted_points << ',' << result.replenished_points << ','
                << result.median_flow_px.x << ',' << result.median_flow_px.y << ',' << result.yaw_delta_rad << ',';
            if (result.planar_displacement_m)
                std::cout << (*result.planar_displacement_m)[0] << ',' << (*result.planar_displacement_m)[1] << ','
                          << (*result.planar_velocity_mps)[0] << ',' << (*result.planar_velocity_mps)[1];
            else std::cout << ",,,";
            std::cout << ',' << result.quality << ',' << result.timing.total_ms << ','
                      << (result.timing.total_ms > 0 ? 1000 / result.timing.total_ms : 0) << ','
                      << result.imu_used << ',' << result.imu_status << '\n' << std::flush;
            if (!debug.empty() && i % 10 == 0 &&
                !cv::imwrite((debug / ("tracks_" + std::to_string(i) + ".png")).string(), tracker->debugImage(result)))
                throw std::runtime_error("Could not write motion debug image");
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
int run(int argc, char** argv) { return runCamera(argc, argv, false); }
int opticalFlow(int argc, char** argv) { return runCamera(argc, argv, true); }

}
#endif

// ============================================================================
// Visual Features
// ============================================================================

#ifdef HAVE_VISUAL_FEATURES

namespace visual_features_command {

using namespace metric_mapping;
namespace {

void benchmark(const cv::Mat& image, VisualFeatureSettings settings, int frames, std::ostream& csv)
{
    settings.profiling = true;
    csv << "selection,frames,computed,mean_ms,median_ms,p95_ms,worst_ms,payload_bytes";
    for (std::size_t i = 0; i < visual_stage_count; ++i) csv << ',' << visualStageName(VisualStage(i)) << "_ms";
    csv << '\n' << std::fixed << std::setprecision(6);
    // Baseline, each selected spatial feature in isolation, then the selected combination.
    // Motion cannot be benchmarked from a still image without inventing a sequence.
    settings.features.reset(std::size_t(VisualFeature::Motion));
    std::vector<std::pair<std::string, VisualFeatureSelection>> runs{{"preparation_only", {}}};
    for (std::size_t i = 0; i < visual_feature_count - 1; ++i) if (settings.features[i])
        runs.push_back({visualFeatureName(VisualFeature(i)), VisualFeatureSelection{}.set(i)});
    runs.push_back({"selected_combination", settings.features});
    for (const auto& run : runs) {
        settings.features = run.second; VisualFeatureExtractor extractor(settings);
        std::vector<double> times; times.reserve(frames);
        std::array<double, visual_stage_count> stages{};
        const VisualFeatureFrame* last = nullptr;
        constexpr int warmup = 10;
        for (int i = 0; i < frames + warmup; ++i) {
            last = &extractor.extract(image, i * 0.1);
            if (i < warmup) continue;
            times.push_back(last->total_ms);
            for (std::size_t j = 0; j < stages.size(); ++j) stages[j] += last->stages[j].elapsed_ms;
        }
        const auto timing = detail::summarizeTimes(times);
        csv << run.first << ',' << frames << ',' << last->computed.count() << ',' << timing.mean << ','
            << timing.median << ',' << timing.p95 << ','
            << timing.worst << ',' << visualFeaturePayloadBytes(*last);
        for (double stage : stages) csv << ',' << stage / frames;
        csv << '\n';
    }
}
}
int run(int argc, char** argv)
{
    try {
        if (argc < 3) throw std::runtime_error("Usage: visual_features image output_directory [--features all|none|names] [--width N] [--height N] [--benchmark N] [--data]");
        VisualFeatureSettings settings; settings.profiling = true;
        int frames = 0; bool data = false;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--features" && i + 1 < argc) settings.features = selectVisualFeatures(argv[++i]);
            else if (arg == "--width" && i + 1 < argc) settings.maximum_width = detail::parseInteger(argv[++i], 1, 10000);
            else if (arg == "--height" && i + 1 < argc) settings.maximum_height = detail::parseInteger(argv[++i], 1, 10000);
            else if (arg == "--benchmark" && i + 1 < argc) frames = detail::parseInteger(argv[++i], 1, 10000);
            else if (arg == "--data") data = true;
            else throw std::runtime_error("Unknown/incomplete option: " + arg);
        }
        const cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (input.empty()) throw std::runtime_error("Could not load image");
        const std::filesystem::path directory(argv[2]); std::filesystem::create_directories(directory);
        cv::setNumThreads(1); // Runner policy; library does not change global thread count.
        VisualFeatureExtractor extractor(settings);
        const auto& result = extractor.extract(input, 0);
        const auto jpeg = directory / "visual_features.jpg";
        const std::string note = "Motion unavailable: this still-image run has no verified capture timing or calibration; no image differences are shown as flow.";
        if (!cv::imwrite(jpeg.string(), renderVisualFeatures(input, result, note), {cv::IMWRITE_JPEG_QUALITY, 95}))
            throw std::runtime_error("Could not write JPEG");
        if (data) writeVisualFeatures(directory / "visual_features.yml.gz", result, settings);
        std::ofstream metadata(directory / "run.txt");
        if (!metadata) throw std::runtime_error("Could not write run metadata");
        metadata << "Imagination GPS-denied UAV camera pre-extraction\nSource: " << argv[1]
                 << "\nOpenCV: " << CV_VERSION << "\nSource dimensions: " << result.source_size
                 << "\nWorking dimensions: " << result.working_size << "\nThreads: 1\n" << note
                 << "\nRequested features: ";
        for (std::size_t i = 0; i < visual_feature_count; ++i) if (result.requested[i]) metadata << visualFeatureName(VisualFeature(i)) << ' ';
        metadata << "\nLogical payload bytes: " << visualFeaturePayloadBytes(result)
                 << "\nContours truncated: " << result.contours_truncated
                 << "\nBenchmark: repeated still image; 10 warmup iterations excluded; wall time only, no energy or safety claim.\n";
        metadata.close(); if (!metadata) throw std::runtime_error("Could not finish run metadata");
        if (frames) {
            std::ofstream csv(directory / "benchmark.csv");
            if (!csv) throw std::runtime_error("Could not write benchmark CSV");
            benchmark(input, settings, frames, csv); csv.close();
            if (!csv) throw std::runtime_error("Could not finish benchmark CSV");
        }
        std::cout << "JPEG: " << jpeg << "\nWorking resolution: " << result.working_size
                  << "\nComputed feature families: " << result.computed.count() << "/" << result.requested.count()
                  << "\nMotion: " << result.motion_status << "\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
}
#endif

// Each command is listed once. The same table drives help and execution.
#ifdef HAVE_VISUAL_FEATURES
namespace pre_extract_command {
using namespace metric_mapping;
namespace {
enum class InputMode { Still, Sequence, Video, Camera, Synthetic };

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path config;
    std::filesystem::path calibration;
    std::filesystem::path imu_snapshot;
    std::filesystem::path range_snapshot;
    std::string selected_features;
    std::string removed_features;
    InputMode mode = InputMode::Still;
    int frames = 100;
    double fps = 10;
    bool write_data = false;
    bool write_debug = false;
    bool disable_geometry = false;
};

constexpr const char* usage =
    "Usage: pre_extract INPUT OUTPUT [--sequence|--video|--camera|--synthetic] "
    "[--config YAML] [--calibration YAML] [--features names] [--without names] "
    "[--no-geometry] [--frames N] [--fps N] [--data] [--debug] "
    "[--imu FILE] [--ultrasonic FILE]";

double parseFps(const std::string& text)
{
    std::size_t consumed = 0;
    const double fps = std::stod(text, &consumed);
    if(consumed != text.size() || !std::isfinite(fps) || fps <= 0)
        throw std::runtime_error("Invalid FPS");
    return fps;
}

Options parseOptions(int argc, char** argv)
{
    if(argc < 3) throw std::runtime_error(usage);
    Options options;
    options.input = argv[1];
    options.output = argv[2];
    int mode_count = 0;
    for(int i = 3; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&]() {
            if(i + 1 >= argc) throw std::runtime_error("Missing value for " + argument);
            return std::string(argv[++i]);
        };
        const auto mode = [&](InputMode selected) {
            options.mode = selected;
            ++mode_count;
        };

        if(argument == "--sequence") mode(InputMode::Sequence);
        else if(argument == "--video") mode(InputMode::Video);
        else if(argument == "--camera") mode(InputMode::Camera);
        else if(argument == "--synthetic") mode(InputMode::Synthetic);
        else if(argument == "--data") options.write_data = true;
        else if(argument == "--debug") options.write_debug = true;
        else if(argument == "--no-geometry") options.disable_geometry = true;
        else if(argument == "--config") options.config = value();
        else if(argument == "--calibration") options.calibration = value();
        else if(argument == "--features") options.selected_features = value();
        else if(argument == "--without") options.removed_features = value();
        else if(argument == "--imu") options.imu_snapshot = value();
        else if(argument == "--ultrasonic") options.range_snapshot = value();
        else if(argument == "--frames")
            options.frames = detail::parseInteger(value(), 1, 100000);
        else if(argument == "--fps") options.fps = parseFps(value());
        else throw std::runtime_error("Unknown pre_extract argument: " + argument);
    }
    if(mode_count > 1) throw std::runtime_error("Choose one input mode");
    return options;
}

VisualFeatureSettings makeSettings(const Options& options)
{
    auto settings = options.config.empty() ?
        analyticalFeatureSettings() : loadAnalyticalConfig(options.config);
    if(!options.selected_features.empty())
        settings.analytical.features = selectAnalyticFeatures(options.selected_features);
    if(!options.removed_features.empty())
        settings.analytical.features &= ~selectAnalyticFeatures(options.removed_features);
    if(options.disable_geometry) settings.analytical.enable_geometry = false;
    if(!options.imu_snapshot.empty()) settings.motion.require_imu = true;
    return settings;
}

double sequenceNumber(const cv::FileNode& frame, const char* key)
{
    const auto value = frame[key];
    if(!value.isReal() && !value.isInt())
        throw std::runtime_error(std::string("Missing numeric sequence field: ") + key);
    return double(value);
}

class ExtractionRun {
public:
    explicit ExtractionRun(const Options& options)
        : options_(options), settings_(makeSettings(options)),
          camera_(loadVisualCalibration(options.calibration, settings_.distortion)),
          extractor_(settings_, prepareCamera()), directory_(options.output),
          timing_(openTimingFile()) {}

    int execute()
    {
        switch(options_.mode) {
        case InputMode::Still: processStill(); break;
        case InputMode::Sequence: processSequence(); break;
        case InputMode::Video: processCapture(false); break;
        case InputMode::Camera: processCapture(true); break;
        case InputMode::Synthetic: processSynthetic(); break;
        }
        if(!last_) throw std::runtime_error("No input frames");
        writeDebugOutputs();
        detail::requireWritable(timing_, directory_ / "timings.csv");
        writeReport();
        return 0;
    }

private:
    const Options& options_;
    VisualFeatureSettings settings_;
    std::optional<CameraIntrinsics> camera_;
    VisualFeatureExtractor extractor_;
    std::filesystem::path directory_;
    std::ofstream timing_;
    std::vector<double> times_;
    const VisualFeatureFrame* last_ = nullptr;
    int frame_count_ = 0;

    std::optional<CameraIntrinsics> prepareCamera()
    {
        const bool has_snapshots = !options_.imu_snapshot.empty() ||
                                   !options_.range_snapshot.empty();
        if(has_snapshots && options_.mode != InputMode::Camera)
            throw std::runtime_error(
                "Sensor snapshots require live camera timestamps; "
                "use per-frame readings in a sequence manifest");
        if(options_.mode == InputMode::Synthetic) {
            if(camera_) throw std::runtime_error("Synthetic input supplies its own calibration");
            camera_ = CameraIntrinsics{256, 256, 220, 220, 127.5, 127.5};
        }
        cv::setNumThreads(1);
        return camera_;
    }

    std::ofstream openTimingFile()
    {
        std::filesystem::create_directories(directory_);
        std::ofstream file(directory_ / "timings.csv");
        if(!file) throw std::runtime_error("Cannot open timing output");
        file << "frame,timestamp_s,total_ms,stage,ran,stage_ms,tracks,inliers,"
                "map_points,coverage_percent,scale_valid,geometry_valid,"
                "dense_flow_valid,payload_bytes\n";
        return file;
    }

    void consume(const cv::Mat& image, double timestamp,
                 std::optional<AltitudeSample> altitude,
                 std::optional<ImuSample> imu)
    {
        const auto& result = extractor_.extract(image, timestamp, altitude, imu);
        last_ = &result;
        times_.push_back(result.total_ms);
        writeTimings(result, timestamp);
        const auto stem = "frame_" + std::to_string(frame_count_);
        if(options_.write_data)
            writeVisualFeatures(directory_ / (stem + ".yml.gz"), result, settings_);
        if(options_.write_debug && frame_count_ % 10 == 0)
            detail::writeImage(directory_ / (stem + ".jpg"),
                               renderAnalyticalFeatures(result));
        ++frame_count_;
    }

    void writeTimings(const VisualFeatureFrame& result, double timestamp)
    {
        const auto& diagnostics = result.analytical;
        for(const auto& stage : result.analytical_stages) {
            timing_ << frame_count_ << ',' << std::setprecision(12) << timestamp
                << ',' << result.total_ms << ',' << stage.name << ',' << stage.ran
                << ',' << stage.elapsed_ms << ',' << diagnostics.num_sparse_tracks
                << ',' << diagnostics.num_pose_inliers
                << ',' << diagnostics.local_map_point_count
                << ',' << diagnostics.geometry_coverage_percent
                << ',' << diagnostics.scale_valid << ',' << diagnostics.geometry_valid
                << ',' << diagnostics.dense_flow_valid << ',' << diagnostics.payload_bytes
                << '\n';
        }
    }

    void processStill()
    {
        consume(cv::imread(options_.input.string()), 0, std::nullopt, std::nullopt);
    }

    void processSequence()
    {
        cv::FileStorage manifest(options_.input.string(), cv::FileStorage::READ);
        if(!manifest.isOpened() || !manifest["frames"].isSeq())
            throw std::runtime_error("Sequence requires YAML frames list");
        const auto parent = options_.input.parent_path();
        for(const auto& frame : manifest["frames"]) {
            const double timestamp = sequenceNumber(frame, "timestamp_s");
            std::optional<AltitudeSample> altitude;
            std::optional<ImuSample> imu;
            if(!frame["altitude_m"].empty()) {
                altitude = AltitudeSample{
                    sequenceNumber(frame, "altitude_m"),
                    sequenceNumber(frame, "range_timestamp_s")};
            }
            if(!frame["imu"].empty()) {
                std::vector<double> angles;
                frame["imu"] >> angles;
                if(angles.size() != 3)
                    throw std::runtime_error("imu must contain roll,pitch,yaw");
                imu = ImuSample{angles[0], angles[1], angles[2],
                                sequenceNumber(frame, "imu_timestamp_s")};
            }
            const auto image = cv::imread((parent / std::string(frame["image"])).string());
            consume(image, timestamp, altitude, imu);
            if(frame_count_ >= options_.frames) break;
        }
    }

    void processCapture(bool live_camera)
    {
#ifdef HAVE_MOTION_CAMERA
        cv::VideoCapture capture;
        if(live_camera)
            capture.open(detail::parseInteger(options_.input.string(), 0));
        else
            capture.open(options_.input.string());
        if(!capture.isOpened()) throw std::runtime_error("Cannot open camera/video");

        double fps = options_.fps;
        if(live_camera) {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, camera_ ? camera_->width : 256);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, camera_ ? camera_->height : 256);
            capture.set(cv::CAP_PROP_FPS, fps);
        } else {
            const double recorded_fps = capture.get(cv::CAP_PROP_FPS);
            if(std::isfinite(recorded_fps) && recorded_fps > 0) fps = recorded_fps;
        }

        cv::Mat image;
        while(frame_count_ < options_.frames && capture.read(image)) {
            const double timestamp = live_camera ?
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count() :
                frame_count_ / fps;
            consume(image, timestamp,
                    detail::readAltitudeSnapshot(options_.range_snapshot),
                    detail::readImuSnapshot(options_.imu_snapshot));
        }
#else
        (void)live_camera;
        throw std::runtime_error("Video/camera input requires BUILD_MOTION_CAMERA=ON");
#endif
    }

    void processSynthetic()
    {
#ifdef HAVE_MOTION
        const auto texture = demo::motionTexture({256, 256});
        cv::Mat gray, bgr;
        for(int i = 0; i < options_.frames; ++i) {
            const double timestamp = i / options_.fps;
            demo::warpMotion(texture, gray, *camera_, 0.7 * i, 0.12 * i);
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            consume(bgr, timestamp, AltitudeSample{2, timestamp},
                    ImuSample{0, 0, 0, timestamp});
        }
#else
        throw std::runtime_error("Synthetic motion input requires BUILD_MOTION=ON");
#endif
    }

    void writeDebugOutputs()
    {
        if(!options_.write_debug) return;
        detail::writeImage(directory_ / "analytical_features.jpg",
                           renderAnalyticalFeatures(*last_));
        if(!last_->local_map.empty()) {
            std::vector<ColoredPoint> cloud;
            cloud.reserve(last_->local_map.size());
            for(const auto& point : last_->local_map) cloud.push_back(point.point);
            writePly(directory_ / "local_map.ply", cloud,
                     "metres in current local nadir segment; no global registration");
        }
        if(last_->terrain.width) {
            writeDemCsv(directory_ / "dem.csv", last_->terrain);
            writeTerrainImages(directory_, last_->terrain);
        }
    }

    const char* inputDescription() const
    {
        switch(options_.mode) {
        case InputMode::Synthetic: return "synthetic known planar motion";
        case InputMode::Sequence: return "user-supplied timestamped sequence";
        case InputMode::Camera: return "live camera delivery timestamps";
        case InputMode::Video: return "video frame-index/FPS timestamps";
        case InputMode::Still: return "single still; no temporal information";
        }
        return "unknown";
    }

    void writeReport()
    {
        const auto summary = detail::summarizeTimes(times_);
        std::ofstream report(directory_ / "run.txt");
        report << "Input: " << inputDescription()
            << "\nFrames: " << frame_count_
            << "\nTensor channels: " << last_->feature_tensor.channel_names.size()
            << "\nMean/median/p95/worst ms (including initialization; "
               "excludes capture/export): "
            << summary.mean << ' ' << summary.median << ' '
            << summary.p95 << ' ' << summary.worst
            << "\nGeometry: " << last_->analytical.geometry_status
            << "\nMap points: " << last_->analytical.local_map_point_count
            << "\nGeometry coverage percent: "
            << last_->analytical.geometry_coverage_percent
            << "\nNo hardware energy measurement.\n";
        detail::requireWritable(report, directory_ / "run.txt");
        std::cout << "Extracted " << frame_count_ << " frame(s), "
            << last_->feature_tensor.channel_names.size()
            << " channels. Mean/p95: " << summary.mean << '/' << summary.p95
            << " ms. Geometry: " << last_->analytical.geometry_status << '\n';
    }
};
} // namespace

int run(int argc, char** argv)
{
    try {
        const auto options = parseOptions(argc, argv);
        return ExtractionRun(options).execute();
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
} // namespace pre_extract_command
#endif

namespace {
struct Command {
    const char* name;
    int (*run)(int argc, char** argv);
};
const Command commands[] = {
#ifdef HAVE_TWO_VIEW
    {"two_view", two_view_command::run},
#endif
#ifdef HAVE_RGBD
    {"metric_mapper", metric_mapper_command::run},
#endif
#ifdef HAVE_MOTION
    {"motion_benchmark", motion_benchmark_command::run},
#endif
#ifdef HAVE_MOTION_CAMERA
    {"motion_camera", motion_camera_command::run},
    {"optical_flow", motion_camera_command::opticalFlow},
#endif
#ifdef HAVE_VISUAL_FEATURES
    {"visual_features", visual_features_command::run},
    {"pre_extract", pre_extract_command::run},
#endif
    {nullptr, nullptr} // End marker also supports builds with no commands enabled.
};
}

int main(int argc, char** argv)
{
    // Old executable names remain aliases; the new entry point takes a subcommand.
    std::string name = std::filesystem::path(argv[0]).stem().string();
    if (name == "imagination") {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Imagination: choose a command\n";
            for (const auto& command : commands)
                if (command.name) std::cout << "  " << command.name << '\n';
            std::cout << "Run a command without arguments to see its usage.\n";
            return 0;
        }
        name = argv[1];
        --argc;
        ++argv;
    }
    for (const auto& command : commands)
        if (command.name && name == command.name) return command.run(argc, argv);
    std::cerr << "Unknown or disabled command: " << name << '\n';
    return 1;
}

// --- Application workflows: connect algorithms to outputs ---

#ifdef HAVE_TWO_VIEW

namespace metric_mapping::app {

void printLandingSummary(
    const metric_mapping::LandingAnalysis& analysis)
{
    if (analysis.ultrasonic)
        std::cout << "Ultrasonic ground check: " << analysis.ultrasonic->status() << '\n';
    std::cout << "Terrain hazard cells: " << analysis.hazard_cells << '\n';
    std::cout << "Landing candidate cells: "
              << analysis.candidate_cells << '\n';
    if (!analysis.best_site.found) {
        std::cout << "Best landing site: none satisfies every constraint\n";
        return;
    }
    const metric_mapping::LandingSite& site = analysis.best_site;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Best landing XYZ: " << site.world_m[0] << ' '
              << site.world_m[1] << ' ' << site.world_m[2] << '\n';
    std::cout << "Best landing slope/roughness: "
              << site.slope_degrees << " deg / "
              << site.roughness_m << " m\n";
    std::cout << "Best landing global safety index: "
              << site.global_safety_index << "%\n";
}

void runStereo(const std::filesystem::path& image_1_path,
               const std::filesystem::path& image_2_path,
               const CameraIntrinsics& camera, double baseline_m,
               const std::optional<UltrasonicMeasurement>& ultrasonic)
{
    const TwoViewResult result =
        reconstructTwoView(image_1_path, image_2_path,
                                           camera, baseline_m);
    TerrainGrid grid =
        createTerrainGrid(result.points, 0.1, 5000000U);
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 0.3;
    idw.minimum_neighbors = 3;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 0.3;
    interpolateIdw(grid, idw);
    const GridStatistics statistics =
        computeGridStatistics(grid);
    const std::vector<ColoredPoint> completed_points =
        terrainGridPoints(grid);
    const CloudBounds bounds =
        computeBounds(result.points);
    const cv::Vec3d uav_position =
        result.camera_positions_world_m.empty()
            ? cv::Vec3d(0.0, 0.0, 0.0)
            : result.camera_positions_world_m.front();
    const LandingAnalysisConfig landing_config;
    const LandingAnalysis landing =
        analyzeLandingSites(
            grid, uav_position, landing_config, ultrasonic);

    const std::filesystem::path cloud_path = "pointcloud.ply";
    const std::filesystem::path raw_cloud_path = "pointcloud_raw.ply";
    const std::filesystem::path matches_path = "matches.jpg";
    const std::filesystem::path disparity_path = "disparity.png";
    writePly(
        raw_cloud_path, result.points,
        "metres in local nadir z-up frame; camera 1 is the origin");
    writePly(
        cloud_path, completed_points,
        "metres in local nadir z-up frame; 0.1 m terrain grid with "
        "bounded IDW completion");
    writeDemCsv("dem.csv", grid);
    writeTerrainImages(".", grid);
    writeDebugVisualization(
        "debug_overview.png", result.points, bounds, grid,
        result.camera_positions_world_m, cv::Vec3d(0.0, 0.0, 0.0));
    writeLandingAnalysis(
        ".", grid, landing, landing_config, uav_position, false);
    detail::writeTwoViewMetadata("metadata.json", camera, baseline_m, result,
                         grid, statistics, completed_points.size());
    if (!cv::imwrite(matches_path.string(), result.matches_image) ||
        !cv::imwrite(disparity_path.string(),
                     result.disparity_preview)) {
        throw std::runtime_error(
            "Could not write matches/disparity images");
    }

    std::cout << "Features image 1: "
              << result.diagnostics.features_image_1 << '\n';
    std::cout << "Features image 2: "
              << result.diagnostics.features_image_2 << '\n';
    std::cout << "Good matches: "
              << result.diagnostics.good_matches << '\n';
    std::cout << "RANSAC inliers: "
              << result.diagnostics.ransac_inliers << '\n';
    std::cout << "Dense stereo points: " << result.points.size()
              << '\n';
    std::cout << "Measured terrain cells: "
              << statistics.measured_cells << '\n';
    std::cout << "IDW interpolated cells: "
              << statistics.interpolated_cells << '\n';
    std::cout << "Final 3D points: "
              << completed_points.size() << '\n';
    printLandingSummary(landing);
    std::cout << "Output: pointcloud.ply\n";
    std::cout << "DEM: dem.csv\n";
    std::cout << "Orthomosaic: orthomosaic.png\n";
    std::cout << "Landing analysis: landing_analysis_overview.png\n";
    std::cout << "WARNING: Metric accuracy requires a measured baseline, "
                 "calibration at the input resolution, and a nadir-facing camera.\n";
}
} // namespace metric_mapping::app

namespace metric_mapping::app {
void runSyntheticDemo(const std::filesystem::path& image_1_path,
                      const std::filesystem::path& image_2_path)
{
    const auto scene = makeSyntheticScene(image_1_path, image_2_path);
    const auto& synthetic_points = scene.points;
    const auto& matches_image = scene.matches_image;
    TerrainGrid grid = createTerrainGrid(synthetic_points, 0.05, 1000000U);
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 0.3;
    idw.minimum_neighbors = 3;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 0.3;
    interpolateIdw(grid, idw);
    const std::vector<ColoredPoint> completed = terrainGridPoints(grid);
    const GridStatistics statistics = computeGridStatistics(grid);
    const CloudBounds bounds = computeBounds(synthetic_points);
    const cv::Vec3d demo_uav_position(
        0.5 * (bounds.minimum[0] + bounds.maximum[0]),
        0.5 * (bounds.minimum[1] + bounds.maximum[1]),
        bounds.maximum[2] + 10.0);
    const LandingAnalysisConfig landing_config;
    const LandingAnalysis landing = analyzeLandingSites(
        grid, demo_uav_position, landing_config);

    const std::filesystem::path output_directory = "demo_output";
    std::filesystem::create_directories(output_directory);
    writePly(output_directory / "pointcloud_raw.ply", synthetic_points,
             "SYNTHETIC DEMO geometry; not reconstructed from the images");
    writePly(output_directory / "pointcloud.ply", completed,
             "SYNTHETIC DEMO one point per image pixel; not reconstructed "
             "from the images");
    writeDemCsv(output_directory / "dem.csv", grid);
    writeTerrainImages(output_directory, grid);
    writeDebugVisualization(output_directory / "debug_overview.png",
                            synthetic_points, bounds, grid, {}, std::nullopt);
    writeLandingAnalysis(output_directory, grid, landing, landing_config,
                         demo_uav_position, true);
    if (!cv::imwrite((output_directory / "matches.jpg").string(),
                     matches_image)) {
        throw std::runtime_error("Could not write demo matches image");
    }

    detail::writeDemoMetadata(output_directory, image_1_path, image_2_path,
                              scene.alignment_matches, synthetic_points.size(),
                              completed.size(), statistics);

    std::cout << "DEMO ONLY: Geometry is synthetic and was not reconstructed "
                 "from the images.\n";
    std::cout << "Aligned feature matches: " << scene.alignment_matches
              << '\n';
    std::cout << "Synthetic raw points: " << synthetic_points.size()
              << '\n';
    std::cout << "IDW interpolated cells: "
              << statistics.interpolated_cells << '\n';
    std::cout << "Final demo points: " << completed.size() << '\n';
    printLandingSummary(landing);
    std::cout << "Output directory: demo_output\n";
}

} // namespace metric_mapping::app

#endif

#ifdef HAVE_RGBD
namespace metric_mapping::app {
void runRgbd(const std::filesystem::path& config_path)
{
    const ReconstructionConfig config = loadConfig(config_path);
    std::filesystem::create_directories(config.output_directory);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coordinate frame: map_z_up (metres)\n";
    std::cout << "Camera: " << config.camera.width << 'x'
              << config.camera.height << " fx=" << config.camera.fx
              << " fy=" << config.camera.fy
              << " cx=" << config.camera.cx
              << " cy=" << config.camera.cy << '\n';
    std::cout << "Depth unit: " << depthUnitName(config.depth.unit)
              << " -> metres\n";
    std::cout << "Pose input: "
              << poseConventionName(config.pose_convention) << '\n';

    VoxelGridAccumulator fusion(config.fusion.voxel_size_m,
                                config.fusion.max_voxels);
    std::vector<ColoredPoint> raw_points;
    raw_points.reserve(std::min<std::size_t>(
        config.fusion.max_raw_points, 1000000U));
    std::vector<cv::Vec3d> camera_positions_world_m;
    camera_positions_world_m.reserve(config.frames.size());

    for (std::size_t index = 0; index < config.frames.size(); ++index) {
        const FrameSpec& frame = config.frames[index];
        const cv::Matx44d camera_to_world = cameraToWorldTransform(
            frame.supplied_pose, config.pose_convention);
        const FrameResult result = reconstructFrame(
            frame.rgb_path, frame.depth_path, config.camera,
            config.depth, camera_to_world);

        if (result.world_points.size() >
            config.fusion.max_raw_points - raw_points.size()) {
            throw std::runtime_error(
                "Raw point export exceeded fusion.max_raw_points; use a "
                "larger pixel_stride or explicitly raise the safety "
                "limit");
        }

        raw_points.insert(raw_points.end(), result.world_points.begin(),
                          result.world_points.end());
        fusion.add(result.world_points);
        const cv::Vec3d camera_position(camera_to_world(0, 3),
                                        camera_to_world(1, 3),
                                        camera_to_world(2, 3));
        camera_positions_world_m.push_back(camera_position);

        const FrameDiagnostics& diagnostics = result.diagnostics;
        std::cout << "Frame " << index << ": "
                  << frame.rgb_path.filename().string() << '\n';
        std::cout << "  Image: " << diagnostics.width << 'x'
                  << diagnostics.height << '\n';
        std::cout << "  Depth min/median/max: "
                  << diagnostics.minimum_depth_m << " / "
                  << diagnostics.median_depth_m << " / "
                  << diagnostics.maximum_depth_m << " m\n";
        std::cout << "  Valid/invalid depth pixels: "
                  << diagnostics.valid_depth_pixels << " / "
                  << diagnostics.invalid_depth_pixels << '\n';
        std::cout << "  Emitted points: "
                  << diagnostics.emitted_points << '\n';
        std::cout << "  T_WC translation: [" << camera_position[0]
                  << ", " << camera_position[1] << ", "
                  << camera_position[2] << "] m\n";
        std::cout << "  T_WC rotation rows: ["
                  << camera_to_world(0, 0) << ", "
                  << camera_to_world(0, 1) << ", "
                  << camera_to_world(0, 2) << "] ["
                  << camera_to_world(1, 0) << ", "
                  << camera_to_world(1, 1) << ", "
                  << camera_to_world(1, 2) << "] ["
                  << camera_to_world(2, 0) << ", "
                  << camera_to_world(2, 1) << ", "
                  << camera_to_world(2, 2) << "]\n";
    }

    const std::vector<ColoredPoint> filtered_points = fusion.points();
    if (filtered_points.empty()) {
        throw std::runtime_error(
            "No world points remain after voxel fusion");
    }
    const CloudBounds bounds = computeBounds(filtered_points);
    TerrainGrid grid = createTerrainGrid(
        filtered_points, config.map.resolution_m, config.map.max_cells);
    interpolateIdw(grid, config.map.idw);
    const GridStatistics grid_statistics = computeGridStatistics(grid);

    writePly(config.output_directory / "cloud_raw.ply", raw_points);
    writePly(config.output_directory / "cloud_filtered.ply",
             filtered_points);
    writeDemCsv(config.output_directory / "dem.csv", grid);
    writeTerrainImages(config.output_directory, grid);
    if (config.uav_position_world_m) {
        const LandingAnalysisConfig landing_config;
        const LandingAnalysis landing = analyzeLandingSites(
            grid, *config.uav_position_world_m, landing_config, config.ultrasonic);
        if (landing.ultrasonic)
            std::cout << "Ultrasonic ground check: " << landing.ultrasonic->status() << '\n';
        writeLandingAnalysis(config.output_directory, grid, landing,
                             landing_config,
                             *config.uav_position_world_m, false);
        std::cout << "Landing candidate cells: "
                  << landing.candidate_cells << '\n';
        if (landing.best_site.found) {
            std::cout << "Best landing XYZ: ["
                      << landing.best_site.world_m[0] << ", "
                      << landing.best_site.world_m[1] << ", "
                      << landing.best_site.world_m[2] << "] m\n";
            std::cout << "Best landing global safety index: "
                      << landing.best_site.global_safety_index
                      << "%\n";
        } else {
            std::cout << "Best landing site: none satisfies every "
                         "constraint\n";
        }
    }
    writeDebugVisualization(
        config.output_directory / "debug_overview.png", filtered_points,
        bounds, grid, camera_positions_world_m,
        config.uav_position_world_m);
    writeMetadata(config.output_directory / "metadata.json", config,
                  raw_points.size(), filtered_points, bounds, grid,
                  grid_statistics, camera_positions_world_m);

    const std::size_t total_cells = grid.validity.size();
    const auto percent = [total_cells](std::size_t count) {
        return total_cells == 0
                   ? 0.0
                   : 100.0 * static_cast<double>(count) /
                         static_cast<double>(total_cells);
    };

    std::cout << "Point cloud raw/filtered: " << raw_points.size()
              << " / " << filtered_points.size() << '\n';
    std::cout << "Point bounds X: [" << bounds.minimum[0] << ", "
              << bounds.maximum[0] << "] m\n";
    std::cout << "Point bounds Y: [" << bounds.minimum[1] << ", "
              << bounds.maximum[1] << "] m\n";
    std::cout << "Point bounds Z: [" << bounds.minimum[2] << ", "
              << bounds.maximum[2] << "] m\n";
    std::cout << "Map: " << grid.width << 'x' << grid.height
              << " at " << grid.resolution_m << " m/cell\n";
    std::cout << "Map measured/interpolated/unknown: "
              << percent(grid_statistics.measured_cells) << "% / "
              << percent(grid_statistics.interpolated_cells) << "% / "
              << percent(grid_statistics.unknown_cells) << "%\n";
    if (!config.uav_position_world_m) {
        std::cout << "UAV position: not supplied; paper-style landing "
                     "analysis skipped\n";
    }
    std::cout << "Output directory: "
              << config.output_directory.string() << '\n';
}
} // namespace metric_mapping::app

#endif
