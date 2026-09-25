#include "test_support.hpp"
#include "metric_mapping/config.hpp"
#include "metric_mapping/io.hpp"
#include "metric_mapping/terrain.hpp"
#include "metric_mapping/ultrasonic.hpp"
#include <fstream>

namespace metric_mapping::tests {
namespace {
void testRangeGatesLanding()
{
    auto grid = makeFlatGrid(81, 81, 0.1);
    const cv::Vec3d uav(0, 0, 10);
    const auto original = analyzeLandingSites(grid, uav);
    require(original.best_site.found && !original.ultrasonic,
            "Flat terrain must have a landing site without a sensor");
    UltrasonicMeasurement reading{10.0, 0.2, {0, 0, 0}};
    auto checked = analyzeLandingSites(grid, uav, {}, reading);
    require(checked.ultrasonic->consistent && checked.best_site.found &&
            checked.candidate_cells == original.candidate_cells &&
            checked.best_site.row == original.best_site.row &&
            checked.best_site.column == original.best_site.column,
            "Matching range must preserve landing selection");
    reading.distance_m = 5;
    checked = analyzeLandingSites(grid, uav, {}, reading);
    require(!checked.best_site.found && checked.candidate_cells == 0 &&
            std::string(checked.ultrasonic->status()) == "mismatch",
            "Range mismatch must withhold landing recommendation");
    reading.distance_m = 10;
    for (const std::uint8_t validity : {0, 127}) {
        grid.validity[grid.index(40, 40)] = validity;
        checked = analyzeLandingSites(grid, uav, {}, reading);
        require(!checked.best_site.found && checked.candidate_cells == 0 &&
                !checked.ultrasonic->mapped_distance_m,
                "Unknown and interpolated ground must not corroborate a reading");
    }
}

void testRangeGeometryAndValidation()
{
    auto grid = makeFlatGrid(11, 11, 1);
    grid.elevation_m[grid.index(3, 6)] = 2;
    UltrasonicMeasurement reading{7.5, 0.0, {1, 2, -0.5}};
    auto check = checkGroundRange(grid, {0, 0, 10}, reading);
    require(check.consistent, "World XYZ mounting offset must select the correct cell and height");
    requireNear(*check.mapped_distance_m, 7.5, 0, "Distance must be sensor height minus ground height");
    reading.distance_m = 7.75;
    reading.maximum_error_m = 0.25;
    require(checkGroundRange(grid, {0, 0, 10}, reading).consistent,
            "Tolerance boundary must be inclusive");
    reading.distance_m = 7.751;
    require(!checkGroundRange(grid, {0, 0, 10}, reading).consistent,
            "Reading beyond tolerance must fail");
    for (const cv::Vec3d& position : {cv::Vec3d(100, 0, 10), cv::Vec3d(0, 0, 1)})
        require(!checkGroundRange(grid, position, reading).mapped_distance_m,
                "Outside map or ground above sensor must be unavailable");
    for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        reading.distance_m = invalid;
        requireThrows([&] { checkGroundRange(grid, {0, 0, 10}, reading); },
                      "Invalid distances must be rejected");
    }
    reading.distance_m = 7.5;
    reading.maximum_error_m = -0.1;
    requireThrows([&] { validateUltrasonicMeasurement(reading); }, "Reject negative tolerance");
    reading.maximum_error_m = 0.2;
    reading.sensor_offset_world_m[0] = std::numeric_limits<double>::infinity();
    requireThrows([&] { validateUltrasonicMeasurement(reading); }, "Reject nonfinite offsets");
}

void testSensorConfigAndOutput()
{
    TemporaryDirectory directory;
    const auto path = directory.path / "sensor.yaml";
    const std::string yaml = "%YAML:1.0\n---\nultrasonic:\n  distance_m: 10.0\n"
        "  maximum_error_m: 0.2\n  direction: world_down\n"
        "  sensor_offset_world_m: [0, 0, 0]\n";
    const auto read = [&](const std::string& text) {
        { std::ofstream output(path); output << text; }
        return loadUltrasonicConfig(path);
    };
    const auto reading = read(yaml);
    requireNear(reading.distance_m, 10, 0, "Sensor YAML distance");
    for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
             {"10.0", "0.0"}, {"10.0", "\"ten\""}, {"10.0", ".Inf"},
             {"0.2", "-1.0"}, {"world_down", "camera_down"},
             {"[0, 0, 0]", "[0, 0]"}, {"distance_m:", "range:"}}) {
        auto invalid = yaml;
        invalid.replace(invalid.find(replacement.first), replacement.first.size(), replacement.second);
        requireThrows([&] { read(invalid); }, "Reject invalid sensor YAML");
    }
    auto grid = makeFlatGrid(11, 11, 1);
    for (const std::string status : {"consistent", "mismatch", "ground_unavailable"}) {
        auto input = reading;
        if (status == "mismatch") input.distance_m = 5;
        if (status == "ground_unavailable") grid.validity[grid.index(5, 5)] = 0;
        const auto analysis = analyzeLandingSites(grid, {0, 0, 10}, {}, input);
        writeLandingAnalysis(directory.path, grid, analysis, {}, {0, 0, 10}, false);
        cv::FileStorage json((directory.path / "landing_site.json").string(),
                             cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        require(json.isOpened() && std::string(json["ultrasonic"]["status"]) == status,
                "Range status must round-trip through JSON");
        if (status != "ground_unavailable")
            requireNear(double(json["ultrasonic"]["mapped_distance_m"]), 10, 0,
                        "Export mapped distance");
    }
}
} // namespace
std::vector<TestCase> ultrasonicTests()
{
    return {{"ultrasonic landing gate", testRangeGatesLanding},
            {"ultrasonic geometry and validation", testRangeGeometryAndValidation},
            {"ultrasonic configuration and JSON", testSensorConfigAndOutput}};
}
} // namespace metric_mapping::tests
