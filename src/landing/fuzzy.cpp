#include "fuzzy.hpp"
#include <algorithm>
#include <cmath>

namespace metric_mapping::detail {
namespace {
constexpr int fuzzy_lut_size = 256;

float leftShoulder(float value, float full_until, float zero_at)
{
    if (value <= full_until)
        return 1.0F;
    if (value >= zero_at)
        return 0.0F;
    return (zero_at - value) / (zero_at - full_until);
}

float rightShoulder(float value, float zero_until, float full_at)
{
    if (value <= zero_until)
        return 0.0F;
    if (value >= full_at)
        return 1.0F;
    return (value - zero_until) / (full_at - zero_until);
}

float triangle(float value, float left, float peak, float right)
{
    if (value <= left || value >= right)
        return 0.0F;
    if (value == peak)
        return 1.0F;
    return value < peak ? (value - left) / (peak - left)
                        : (right - value) / (right - peak);
}

std::array<float, 3> terrainMembership(float ratio)
{
    // Fig. 16: good falls to zero at half the allowed value, normal peaks
    // there, and bad reaches full membership at the allowed limit.
    return {leftShoulder(ratio, 0.0F, 0.5F),
            triangle(ratio, 0.0F, 0.5F, 1.0F),
            rightShoulder(ratio, 0.0F, 1.0F)};
}

std::array<float, 5> safetyOutputMembership(float percent)
{
    // Fig. 17. Labels: unsafe, very risky, risky, little risky, safe.
    return {leftShoulder(percent, 5.0F, 10.0F),
            triangle(percent, 5.0F, 15.0F, 25.0F),
            triangle(percent, 15.0F, 30.0F, 45.0F),
            triangle(percent, 30.0F, 50.0F, 70.0F),
            rightShoulder(percent, 50.0F, 75.0F)};
}

std::array<float, 5> distanceOutputMembership(float percent)
{
    // Figs. 21 and 24. Labels: poor, bad, normal, good, very good.
    return {leftShoulder(percent, 20.0F, 40.0F),
            triangle(percent, 0.0F, 20.0F, 40.0F),
            triangle(percent, 20.0F, 40.0F, 70.0F),
            triangle(percent, 40.0F, 60.0F, 100.0F),
            triangle(percent, 60.0F, 80.0F, 100.0F)};
}

enum class OutputPartition {
    Safety,
    Distance
};

float outputMembership(OutputPartition partition, int label, int percent)
{
    struct Samples {
        std::array<std::array<float, 101>, 5> safety{};
        std::array<std::array<float, 101>, 5> distance{};
    };
    static const Samples samples = [] {
        Samples result;
        for (int value = 0; value <= 100; ++value) {
            const auto safety =
                safetyOutputMembership(static_cast<float>(value));
            const auto distance =
                distanceOutputMembership(static_cast<float>(value));
            for (int output = 0; output < 5; ++output) {
                result.safety[static_cast<std::size_t>(output)]
                             [static_cast<std::size_t>(value)] =
                    safety[static_cast<std::size_t>(output)];
                result.distance[static_cast<std::size_t>(output)]
                               [static_cast<std::size_t>(value)] =
                    distance[static_cast<std::size_t>(output)];
            }
        }
        return result;
    }();
    const auto& selected = partition == OutputPartition::Safety
                               ? samples.safety
                               : samples.distance;
    return selected[static_cast<std::size_t>(label)]
                   [static_cast<std::size_t>(percent)];
}

template <std::size_t Rows, std::size_t Columns>
float fuzzyCentroid(
    const std::array<float, Rows>& rows,
    const std::array<float, Columns>& columns,
    const std::array<std::array<int, Columns>, Rows>& rules,
    OutputPartition output_partition)
{
    std::array<float, 5> activation{};
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Columns; ++column) {
            const float firing = std::min(rows[row], columns[column]);
            const int label = rules[row][column];
            activation[static_cast<std::size_t>(label)] = std::max(
                activation[static_cast<std::size_t>(label)], firing);
        }
    }

    double weighted_sum = 0.0;
    double membership_sum = 0.0;
    for (int percent = 0; percent <= 100; ++percent) {
        float aggregated = 0.0F;
        for (int label = 0; label < 5; ++label) {
            aggregated = std::max(
                aggregated,
                std::min(activation[static_cast<std::size_t>(label)],
                         outputMembership(output_partition, label,
                                          percent)));
        }
        weighted_sum += percent * aggregated;
        membership_sum += aggregated;
    }
    return membership_sum > 0.0
               ? static_cast<float>(weighted_sum / membership_sum)
               : 0.0F;
}

} // namespace

float FuzzyScores::get(FuzzyStage stage, int first, int second)
{
    auto& table = tables_[static_cast<std::size_t>(stage)];
    if (table.empty())
        table.assign(fuzzy_lut_size * fuzzy_lut_size, -1.0F);
    float& score = table[static_cast<std::size_t>(first) * fuzzy_lut_size + second];
    if (score < 0)
        score = evaluate(stage, float(first) / (fuzzy_lut_size - 1),
                         float(second) / (fuzzy_lut_size - 1));
    return score;
}

float FuzzyScores::evaluate(FuzzyStage stage, float first_unit, float second_unit)
{
    // Table IV. Rows: good/normal/bad roughness. Columns:
    // good/normal/bad slope. Output labels run unsafe..safe (0..4).
    constexpr std::array<std::array<int, 3>, 3> safety_rules{{
        {{4, 2, 0}},
        {{3, 1, 0}},
        {{0, 0, 0}},
    }};
    // Table V, reordered to rows bad/normal/good nearest-hazard distance
    // and columns near/normal/far spatial distance. Outputs are poor..very
    // good (0..4).
    constexpr std::array<std::array<int, 3>, 3> distance_rules{{
        {{1, 0, 0}},
        {{2, 2, 1}},
        {{3, 4, 3}},
    }};
    // Table VI, reordered to rows poor..very-good distance index and columns
    // unsafe..safe terrain index. Outputs are poor..very-good (0..4).
    constexpr std::array<std::array<int, 5>, 5> global_rules{{
        {{0, 0, 0, 0, 1}},
        {{0, 0, 1, 1, 1}},
        {{0, 1, 1, 2, 3}},
        {{0, 1, 2, 2, 3}},
        {{0, 1, 2, 3, 4}},
    }};

    if (stage == FuzzyStage::Safety)
        return fuzzyCentroid(terrainMembership(2.0F * first_unit),
                             terrainMembership(2.0F * second_unit),
                             safety_rules, OutputPartition::Safety);
    if (stage == FuzzyStage::Global)
        return fuzzyCentroid(distanceOutputMembership(100.0F * first_unit),
                             safetyOutputMembership(100.0F * second_unit),
                             global_rules, OutputPartition::Distance);
    const float hazard_ratio = 4.0F * first_unit;
    const std::array<float, 3> hazard_distance{
        leftShoulder(hazard_ratio, 0.0F, 2.5F),
        triangle(hazard_ratio, 1.0F, 2.5F, 4.0F),
        rightShoulder(hazard_ratio, 2.5F, 4.0F)};
    const std::array<float, 3> spatial_distance{
        leftShoulder(second_unit, 0.0F, 0.5F),
        triangle(second_unit, 0.0F, 0.5F, 1.0F),
        rightShoulder(second_unit, 0.5F, 1.0F)};
    return fuzzyCentroid(hazard_distance, spatial_distance, distance_rules,
                         OutputPartition::Distance);
}


int quantize(double value, double maximum)
{
    const double normalized = std::clamp(value / maximum, 0.0, 1.0);
    return static_cast<int>(std::lround(
        normalized * (fuzzy_lut_size - 1)));
}

} // namespace metric_mapping::detail
