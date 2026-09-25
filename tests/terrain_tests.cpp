#include "test_support.hpp"
#include "metric_mapping/terrain.hpp"

namespace metric_mapping::tests {
namespace {
void testDemAndOrthomosaicRegistration()
{
    const std::vector<ColoredPoint> points{
        {0.0F, 0.0F, 1.0F, 10, 20, 30},
        {1.0F, 0.0F, 2.0F, 40, 50, 60}};
    const TerrainGrid grid = createTerrainGrid(points, 1.0, 100);
    require(grid.width == 2 && grid.height == 1,
            "Known DEM must have expected dimensions");
    requireNear(grid.elevation_m[grid.index(0, 0)], 1.0, 1e-6,
                "DEM first elevation");
    requireNear(grid.elevation_m[grid.index(0, 1)], 2.0, 1e-6,
                "DEM second elevation");
    const cv::Vec3b first_bgr = grid.color_bgr[grid.index(0, 0)];
    require(first_bgr == cv::Vec3b(30, 20, 10),
            "Orthomosaic color must stay registered with DEM cell");
}

void testPositiveZElevationAndIdwMask()
{
    const std::vector<ColoredPoint> stacked{
        {0.0F, 0.0F, 1.0F, 10, 10, 10},
        {0.0F, 0.0F, 3.0F, 30, 30, 30}};
    const TerrainGrid surface = createTerrainGrid(stacked, 1.0, 10);
    requireNear(surface.elevation_m[0], 3.0, 1e-6,
                "Increasing elevation must always be +Z");

    TerrainGrid grid;
    grid.width = 3;
    grid.height = 1;
    grid.resolution_m = 1.0;
    grid.minimum_x_m = 0.0;
    grid.maximum_y_m = 0.0;
    grid.elevation_m = {1.0F,
                        std::numeric_limits<float>::quiet_NaN(), 3.0F};
    grid.color_bgr = {cv::Vec3b(10, 10, 10), cv::Vec3b(0, 0, 0),
                      cv::Vec3b(30, 30, 30)};
    grid.validity = {255, 0, 255};
    IdwConfig idw;
    idw.enabled = true;
    idw.search_radius_m = 1.1;
    idw.minimum_neighbors = 2;
    idw.power = 2.0;
    idw.maximum_interpolation_distance_m = 1.1;
    interpolateIdw(grid, idw);
    requireNear(grid.elevation_m[1], 2.0, 1e-6,
                "IDW must interpolate the centered elevation");
    require(grid.validity[1] == 127,
            "Interpolated cells must remain distinguishable from measured");

    const std::vector<ColoredPoint> completed = terrainGridPoints(grid);
    require(completed.size() == 3,
            "Completed terrain cloud must contain measured and IDW cells");
    requireNear(completed[1].x, 1.0, 1e-6,
                "Completed cloud must preserve grid X registration");
    requireNear(completed[1].z, 2.0, 1e-6,
                "Completed cloud must preserve interpolated elevation");

    grid.width = 5;
    grid.elevation_m.assign(5, std::numeric_limits<float>::quiet_NaN());
    grid.color_bgr.assign(5, cv::Vec3b(0, 0, 0));
    grid.validity.assign(5, 0);
    grid.elevation_m[0] = 1.0F;
    grid.validity[0] = 255;
    idw.minimum_neighbors = 1;
    interpolateIdw(grid, idw);
    require(grid.validity[1] == 127 && grid.validity[2] == 0,
            "In-place IDW must never propagate newly interpolated cells");
}

void testPaperStyleLandingAnalysis()
{
    TerrainGrid grid = makeFlatGrid(121, 121, 0.1);
    const int center_row = grid.height / 2;
    const int center_column = grid.width / 2;
    for (int row = center_row - 4; row <= center_row + 4; ++row) {
        for (int column = center_column - 4;
             column <= center_column + 4; ++column) {
            grid.elevation_m[grid.index(row, column)] = 0.8F;
            grid.color_bgr[grid.index(row, column)] =
                cv::Vec3b(20, 20, 20);
        }
    }

    const LandingAnalysisConfig config;
    const LandingAnalysis result = analyzeLandingSites(
        grid, cv::Vec3d(0.0, 0.0, 10.0), config);
    require(result.hazard_mask.at<std::uint8_t>(
                center_row, center_column - 4) != 0 &&
                result.nearest_hazard_distance_m.at<float>(
                    center_row, center_column) <
                    config.minimum_hazard_distance_m,
            "Raised obstacle edge must be hazardous and its top must fail "
            "the required hazard clearance");
    require(result.admissible_cells > 0 && result.clearance_cells > 0,
            "Flat terrain away from the obstacle must remain admissible");
    require(result.best_site.found && result.candidate_cells > 0,
            "Known flat terrain must yield at least one landing candidate");
    require(result.best_site.slope_degrees <=
                config.maximum_slope_degrees &&
                result.best_site.roughness_m <=
                    config.maximum_roughness_m &&
                result.best_site.nearest_hazard_distance_m >=
                    config.minimum_hazard_distance_m &&
                result.best_site.global_safety_index >=
                    config.minimum_global_safety_index,
            "Selected site must satisfy every configured constraint");
}

void testMetricSlopeEstimation()
{
    TerrainGrid grid = makeFlatGrid(51, 51, 0.1);
    constexpr double rise_per_metre = 0.1;
    for (int row = 0; row < grid.height; ++row) {
        for (int column = 0; column < grid.width; ++column) {
            const double x = grid.minimum_x_m +
                             column * grid.resolution_m;
            grid.elevation_m[grid.index(row, column)] =
                static_cast<float>(rise_per_metre * x);
        }
    }
    LandingAnalysisConfig config;
    config.uav_footprint_diagonal_m = 0.3;
    const LandingAnalysis result = analyzeLandingSites(
        grid, cv::Vec3d(0.0, 0.0, 10.0), config);
    const double expected_degrees =
        std::atan(rise_per_metre) * 180.0 / CV_PI;
    requireNear(result.slope_degrees.at<float>(25, 25),
                expected_degrees, 0.2,
                "Sobel DEM slope must preserve metric rise/run");
}

void testDecimalGridBoundaries()
{
    std::vector<ColoredPoint> points;
    for (int row = 0; row < 512; ++row)
        for (int column = 0; column < 512; ++column)
            points.push_back({float((column - 256) * 0.05),
                              float((256 - row) * 0.05),
                              float(row * 512 + column), 1, 2, 3});
    const auto grid = createTerrainGrid(points, 0.05, points.size());
    const auto statistics = computeGridStatistics(grid);
    require(grid.width == 512 && grid.height == 512 &&
            statistics.measured_cells == points.size() && statistics.unknown_cells == 0,
            "A complete decimal lattice must not manufacture holes");
    for (std::size_t index = 0; index < points.size(); ++index)
        requireNear(grid.elevation_m[index], points[index].z, 0.0,
                    "Every original lattice measurement must survive");
    const auto interior = createTerrainGrid(
        {{0, 0, 1, 0, 0, 0}, {0.049F, 0, 2, 0, 0, 0}, {0.1F, 0, 3, 0, 0, 0}}, 0.05, 10);
    requireNear(interior.elevation_m[0], 2, 0, "Ordinary points must retain floor-based binning");
}

void testRidgedTerrainRejection()
{
    for (bool checkerboard : {false, true}) {
        auto grid = makeFlatGrid(121, 121, 0.1);
        for (int row = 0; row < grid.height; ++row)
            for (int column = 0; column < grid.width; ++column)
                grid.elevation_m[grid.index(row, column)] =
                    ((column + (checkerboard ? row : 0)) % 2) ? 0.3F : 0.0F;
        const auto result = analyzeLandingSites(grid, {0, 0, 10});
        require(!result.best_site.found && result.candidate_cells == 0 &&
                result.hazard_mask.at<std::uint8_t>(60, 60) == 255,
                "Measured 30 cm ridges must be hazards despite Sobel cancellation");
    }
}
} // namespace

std::vector<TestCase> terrainTests()
{
    return {
        {"DEM and orthomosaic registration", testDemAndOrthomosaicRegistration},
        {"+Z elevation and IDW validity", testPositiveZElevationAndIdwMask},
        {"paper-style landing analysis", testPaperStyleLandingAnalysis},
        {"metric DEM slope estimation", testMetricSlopeEstimation},
        {"decimal grid boundaries", testDecimalGridBoundaries},
        {"ridged terrain rejection", testRidgedTerrainRejection}
    };
}
} // namespace metric_mapping::tests
