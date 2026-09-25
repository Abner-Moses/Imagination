#pragma once

#include "metric_mapping/types.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace metric_mapping {

class VoxelGridAccumulator {
public:
    VoxelGridAccumulator(double voxel_size_m, std::size_t max_voxels);

    void add(const ColoredPoint& point);
    void add(const std::vector<ColoredPoint>& points);
    std::vector<ColoredPoint> points() const;
    std::size_t voxelCount() const;

private:
    struct Key {
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;

        bool operator==(const Key& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const;
    };

    struct Accumulator {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        std::size_t count = 0;
    };

    double voxel_size_m_ = 0.0;
    std::size_t max_voxels_ = 0;
    std::unordered_map<Key, Accumulator, KeyHash> voxels_;
};

CloudBounds computeBounds(const std::vector<ColoredPoint>& points);

}  // namespace metric_mapping
