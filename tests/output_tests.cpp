#include "test_support.hpp"
#include "metric_mapping/io.hpp"
#include "metric_mapping/point_cloud.hpp"
#include "metric_mapping/terrain.hpp"
#include "metric_mapping/visualization.hpp"
#include <opencv2/imgcodecs.hpp>

namespace metric_mapping::tests {
namespace {
void testOutputSerialization()
{
    TemporaryDirectory temporary;
    const auto& directory = temporary.path;

    const std::vector<ColoredPoint> points{
        {0.0F, 0.0F, 1.0F, 255, 0, 0},
        {1.0F, 0.0F, 2.0F, 0, 255, 0}};
    const CloudBounds bounds = computeBounds(points);
    const TerrainGrid grid = createTerrainGrid(points, 1.0, 100);
    const GridStatistics statistics = computeGridStatistics(grid);
    ReconstructionConfig config;
    config.camera = {2, 1, 1.0, 1.0, 0.0, 0.0};
    config.depth = metricDepthConfig();
    config.fusion.voxel_size_m = 0.1;
    config.map.resolution_m = 1.0;
    config.world_frame = "map_z_up";

    writePly(directory / "cloud.ply", points);
    writeDemCsv(directory / "dem.csv", grid);
    writeTerrainImages(directory, grid);
    const LandingAnalysisConfig landing_config;
    const cv::Vec3d uav_position(0.0, 0.0, 3.0);
    const LandingAnalysis landing = analyzeLandingSites(
        grid, uav_position, landing_config);
    writeLandingAnalysis(directory, grid, landing, landing_config,
                         uav_position, false);
    writeDebugVisualization(directory / "debug.png", points, bounds, grid,
                            {cv::Vec3d(0.0, 0.0, 2.0)}, cv::Vec3d(0.0, 0.0, 3.0));
    writeMetadata(directory / "metadata.json", config, points.size(), points,
                  bounds, grid, statistics,
                  {cv::Vec3d(0.0, 0.0, 2.0)});

    for (const std::string& filename :
         {"cloud.ply", "dem.csv", "orthomosaic.png",
          "validity_mask.png", "slope.png", "roughness.png",
          "hazard_mask.png", "nearest_hazard_distance.png",
          "safety_index.png", "distance_index.png",
          "global_safety_index.png", "best_landing_site.png",
          "landing_analysis_overview.png", "landing_site.json",
          "debug.png", "metadata.json"}) {
        const std::filesystem::path path = directory / filename;
        require(std::filesystem::is_regular_file(path) &&
                    std::filesystem::file_size(path) > 0,
                "Expected serialized output: " + filename);
    }
}

void testThinGridExports()
{
    TemporaryDirectory directory;
    for (const auto size : {cv::Size(1000, 1), cv::Size(1, 1000)}) {
        const auto grid = makeFlatGrid(size.width, size.height, 0.1);
        const auto points = terrainGridPoints(grid);
        const auto analysis = analyzeLandingSites(grid, {0, 0, 10});
        writeLandingAnalysis(directory.path, grid, analysis, {}, {0, 0, 10}, false);
        writeDebugVisualization(directory.path / "debug.png", points,
                                computeBounds(points), grid, {}, std::nullopt);
        require(!cv::imread((directory.path / "debug.png").string()).empty(),
                "Thin grid debug visualization must be readable");
        require(!cv::imread((directory.path / "landing_analysis_overview.png").string()).empty(),
                "Thin grid landing visualization must be readable");
    }
}

void testJsonEscaping()
{
    require(escapeJson("a\"b\\c\n\t\r\b\f") ==
            "a\\\"b\\\\c\\u000a\\u0009\\u000d\\u0008\\u000c",
            "JSON must escape quotes, backslashes, and control characters");
    for (unsigned char character = 0; character < 32; ++character)
        require(escapeJson(std::string(1, char(character))).size() == 6,
                "Every JSON control character must be escaped");
    require(escapeJson("terrain-é.png") == "terrain-é.png", "UTF-8 must be preserved");
}
} // namespace

std::vector<TestCase> outputTests()
{
    return {
        {"output serialization", testOutputSerialization},
        {"thin grid exports", testThinGridExports},
        {"JSON escaping", testJsonEscaping}
    };
}
} // namespace metric_mapping::tests
