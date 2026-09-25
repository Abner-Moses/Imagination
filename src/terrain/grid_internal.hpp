#pragma once
#include "metric_mapping/types.hpp"

namespace metric_mapping::detail {
// Shared precondition for grid processing; no file I/O or reconstruction here.
void validateGrid(const TerrainGrid& grid);
}
