#pragma once
#include <array>
#include <vector>

namespace metric_mapping::detail {
enum class FuzzyStage { Safety, Distance, Global };

// Cache only scores requested by this analysis. Scores are percentages.
class FuzzyScores {
public:
    float get(FuzzyStage stage, int first, int second);
private:
    std::array<std::vector<float>, 3> tables_;
    static float evaluate(FuzzyStage stage, float first_unit, float second_unit);
};
int quantize(double value, double maximum);
} // namespace metric_mapping::detail
