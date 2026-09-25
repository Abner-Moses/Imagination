#include "output_internal.hpp"
#include <iomanip>

namespace metric_mapping::detail {
void writeLandingJson(const std::filesystem::path& output_directory,
                      const LandingAnalysis& analysis, const LandingAnalysisConfig& config,
                      const cv::Vec3d& uav_position_world_m, bool demo_only)
{
    const std::filesystem::path json_path =
        output_directory / "landing_site.json";
    std::ofstream json(json_path);
    requireWritable(json, json_path);
    json << std::fixed << std::setprecision(8);
    json << "{\n";
    json << "  \"demo_only\": " << (demo_only ? "true" : "false")
         << ",\n";
    json << "  \"warning\": \""
         << (demo_only
                 ? "Synthetic appearance-derived geometry; not valid for flight"
                 : "Verify calibration, scale, thresholds, and flight regulations before use")
         << "\",\n";
    json << "  \"water_detection\": "
            "\"disabled; dry-scene assumption (paper uses a neural model)\",\n";
    json << "  \"uav_position_world_m\": ["
         << uav_position_world_m[0] << ", "
         << uav_position_world_m[1] << ", "
         << uav_position_world_m[2] << "],\n";
    json << "  \"ultrasonic\": ";
    if (!analysis.ultrasonic) {
        json << "null,\n";
    } else {
        const auto& check = *analysis.ultrasonic;
        const auto& reading = check.measurement;
        json << "{\n    \"status\": \"" << check.status() << "\",\n"
             << "    \"direction\": \"world_down\",\n"
             << "    \"distance_m\": " << reading.distance_m << ",\n"
             << "    \"maximum_error_m\": " << reading.maximum_error_m << ",\n"
             << "    \"sensor_offset_world_m\": [" << reading.sensor_offset_world_m[0]
             << ", " << reading.sensor_offset_world_m[1] << ", "
             << reading.sensor_offset_world_m[2] << "],\n"
             << "    \"mapped_distance_m\": ";
        if (check.mapped_distance_m) json << *check.mapped_distance_m;
        else json << "null";
        json << "\n  },\n";
    }
    json << "  \"constraints\": {\n";
    json << "    \"maximum_slope_degrees\": "
         << config.maximum_slope_degrees << ",\n";
    json << "    \"maximum_roughness_m\": "
         << config.maximum_roughness_m << ",\n";
    json << "    \"minimum_hazard_distance_m\": "
         << config.minimum_hazard_distance_m << ",\n";
    json << "    \"uav_footprint_diagonal_m\": "
         << config.uav_footprint_diagonal_m << ",\n";
    json << "    \"minimum_global_safety_index\": "
         << config.minimum_global_safety_index << "\n";
    json << "  },\n";
    json << "  \"hazard_cells\": " << analysis.hazard_cells << ",\n";
    json << "  \"admissible_cells\": "
         << analysis.admissible_cells << ",\n";
    json << "  \"clearance_cells\": "
         << analysis.clearance_cells << ",\n";
    json << "  \"candidate_cells\": "
         << analysis.candidate_cells << ",\n";
    json << "  \"maximum_global_safety_index\": "
         << analysis.maximum_global_safety_index << ",\n";
    json << "  \"landing_site_found\": "
         << (analysis.best_site.found ? "true" : "false") << ",\n";
    json << "  \"best_landing_site\": ";
    if (!analysis.best_site.found) {
        json << "null\n";
    } else {
        const LandingSite& site = analysis.best_site;
        json << "{\n";
        json << "    \"grid_row\": " << site.row << ",\n";
        json << "    \"grid_column\": " << site.column << ",\n";
        json << "    \"world_m\": [" << site.world_m[0] << ", "
             << site.world_m[1] << ", " << site.world_m[2] << "],\n";
        json << "    \"slope_degrees\": " << site.slope_degrees
             << ",\n";
        json << "    \"roughness_m\": " << site.roughness_m << ",\n";
        json << "    \"nearest_hazard_distance_m\": "
             << site.nearest_hazard_distance_m << ",\n";
        json << "    \"spatial_distance_m\": "
             << site.spatial_distance_m << ",\n";
        json << "    \"safety_index\": " << site.safety_index << ",\n";
        json << "    \"distance_index\": " << site.distance_index
             << ",\n";
        json << "    \"global_safety_index\": "
             << site.global_safety_index << "\n";
        json << "  }\n";
    }
    json << "}\n";
    json.close();
    requireWritable(json, json_path);
}

} // namespace metric_mapping::detail
