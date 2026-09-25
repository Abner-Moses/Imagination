#include "metric_mapping/point_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace metric_mapping {

std::size_t VoxelGridAccumulator::KeyHash::operator()(const Key& key) const
{
    const std::size_t hx = std::hash<std::int64_t>{}(key.x);
    const std::size_t hy = std::hash<std::int64_t>{}(key.y);
    const std::size_t hz = std::hash<std::int64_t>{}(key.z);
    return hx ^ (hy + 0x9e3779b9U + (hx << 6U) + (hx >> 2U)) ^
           (hz + 0x9e3779b9U + (hy << 6U) + (hy >> 2U));
}

VoxelGridAccumulator::VoxelGridAccumulator(double voxel_size_m,
                                           std::size_t max_voxels)
    : voxel_size_m_(voxel_size_m), max_voxels_(max_voxels)
{
    if (!std::isfinite(voxel_size_m_) || voxel_size_m_ <= 0.0 ||
        max_voxels_ == 0) {
        throw std::runtime_error(
            "Voxel size and maximum voxel count must be positive");
    }
}

void VoxelGridAccumulator::add(const ColoredPoint& point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
        throw std::runtime_error("Cannot fuse a non-finite point");
    }

    const Key key{
        static_cast<std::int64_t>(std::floor(point.x / voxel_size_m_)),
        static_cast<std::int64_t>(std::floor(point.y / voxel_size_m_)),
        static_cast<std::int64_t>(std::floor(point.z / voxel_size_m_))};

    auto iterator = voxels_.find(key);
    if (iterator == voxels_.end()) {
        if (voxels_.size() >= max_voxels_) {
            throw std::runtime_error(
                "Voxel map exceeded fusion.max_voxels; increase the limit or "
                "increase voxel_size_m");
        }
        iterator = voxels_.emplace(key, Accumulator{}).first;
    }

    Accumulator& accumulator = iterator->second;
    accumulator.x += point.x;
    accumulator.y += point.y;
    accumulator.z += point.z;
    accumulator.r += point.r;
    accumulator.g += point.g;
    accumulator.b += point.b;
    ++accumulator.count;
}

void VoxelGridAccumulator::add(const std::vector<ColoredPoint>& points)
{
    for (const ColoredPoint& point : points) {
        add(point);
    }
}

std::vector<ColoredPoint> VoxelGridAccumulator::points() const
{
    std::vector<ColoredPoint> filtered;
    filtered.reserve(voxels_.size());

    for (const auto& entry : voxels_) {
        const Accumulator& accumulator = entry.second;
        const double inverse_count =
            1.0 / static_cast<double>(accumulator.count);
        const auto color = [&](double sum) {
            return static_cast<std::uint8_t>(std::clamp(
                std::lround(sum * inverse_count), 0L, 255L));
        };
        filtered.push_back(
            {static_cast<float>(accumulator.x * inverse_count),
             static_cast<float>(accumulator.y * inverse_count),
             static_cast<float>(accumulator.z * inverse_count),
             color(accumulator.r),
             color(accumulator.g),
             color(accumulator.b)});
    }

    std::sort(filtered.begin(), filtered.end(),
              [](const ColoredPoint& left, const ColoredPoint& right) {
                  if (left.z != right.z)
                      return left.z < right.z;
                  if (left.y != right.y)
                      return left.y < right.y;
                  return left.x < right.x;
              });
    return filtered;
}

std::size_t VoxelGridAccumulator::voxelCount() const
{
    return voxels_.size();
}

CloudBounds computeBounds(const std::vector<ColoredPoint>& points)
{
    if (points.empty()) {
        throw std::runtime_error("Cannot compute bounds of an empty cloud");
    }

    CloudBounds bounds;
    bounds.minimum = cv::Vec3d(points.front().x,
                               points.front().y,
                               points.front().z);
    bounds.maximum = bounds.minimum;

    for (const ColoredPoint& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
            throw std::runtime_error("Cannot compute bounds of a non-finite point");
        bounds.minimum[0] = std::min(bounds.minimum[0],
                                     static_cast<double>(point.x));
        bounds.minimum[1] = std::min(bounds.minimum[1],
                                     static_cast<double>(point.y));
        bounds.minimum[2] = std::min(bounds.minimum[2],
                                     static_cast<double>(point.z));
        bounds.maximum[0] = std::max(bounds.maximum[0],
                                     static_cast<double>(point.x));
        bounds.maximum[1] = std::max(bounds.maximum[1],
                                     static_cast<double>(point.y));
        bounds.maximum[2] = std::max(bounds.maximum[2],
                                     static_cast<double>(point.z));
    }
    return bounds;
}

}  // namespace metric_mapping
