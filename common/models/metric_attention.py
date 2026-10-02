"""Portable dense and gather-based sparse spatial attention helpers."""

from __future__ import annotations

from time import perf_counter
import torch


def positions_from_analytical(
    analytical: torch.Tensor,
    validity: torch.Tensor,
    calibration: torch.Tensor,
    depth_scale_m: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project the IMF depth plane to camera-frame XYZ at 32x32.

    Calibration rows are ``[width,height,fx,fy,cx,cy]`` in source-image pixels.
    The output uses the OpenCV camera frame (+x right, +y down, +z forward).
    Depth is the measured IMF channel, scaled by the extractor's configured
    depth range; target-only simulator depth is never read here.
    """
    if analytical.ndim != 4 or analytical.shape[1:] != (28, 32, 32):
        raise ValueError("Metric positions require Bx28x32x32 IMF input")
    if validity.shape != analytical.shape:
        raise ValueError("Analytical validity must match the IMF tensor")
    batch, _, height, width = analytical.shape
    calibration = torch.as_tensor(calibration, dtype=analytical.dtype, device=analytical.device)
    if calibration.ndim == 1:
        calibration = calibration.unsqueeze(0).expand(batch, -1)
    if calibration.shape != (batch, 6):
        raise ValueError("Calibration must be [B,6] = width,height,fx,fy,cx,cy")
    scale = torch.as_tensor(depth_scale_m, dtype=analytical.dtype, device=analytical.device)
    if scale.ndim == 0:
        scale = scale.expand(batch)
    scale = scale.reshape(batch, 1, 1)
    image_w, image_h, fx, fy, cx, cy = calibration.unbind(dim=1)
    fx_grid = fx * width / image_w
    fy_grid = fy * height / image_h
    cx_grid = (cx + 0.5) * width / image_w - 0.5
    cy_grid = (cy + 0.5) * height / image_h - 0.5
    rows, cols = torch.meshgrid(
        torch.arange(height, device=analytical.device, dtype=analytical.dtype),
        torch.arange(width, device=analytical.device, dtype=analytical.dtype),
        indexing="ij",
    )
    depth = analytical[:, 22] * scale
    positions = torch.stack(
        (
            (cols[None] - cx_grid[:, None, None]) * depth / fx_grid[:, None, None],
            (rows[None] - cy_grid[:, None, None]) * depth / fy_grid[:, None, None],
            depth,
        ),
        dim=-1,
    )
    valid = (validity[:, 22] > 0) & torch.isfinite(depth) & (depth > 0)
    valid = valid & torch.isfinite(positions).all(dim=-1)
    return torch.nan_to_num(positions), valid.to(analytical.dtype)


def downsample_metric_positions(
    positions: torch.Tensor, validity: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validity-weighted spatial reduction to an HTransformer stage grid."""
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape BxHxWx3")
    mask = validity.to(positions.dtype).unsqueeze(1)
    numerator = torch.nn.functional.adaptive_avg_pool2d(
        (positions * validity.to(positions.dtype).unsqueeze(-1)).permute(0, 3, 1, 2), size
    )
    support = torch.nn.functional.adaptive_avg_pool2d(mask, size)
    valid = support[:, 0] > 0
    reduced = numerator / support.clamp_min(1e-8)
    reduced = reduced.permute(0, 2, 3, 1)
    return torch.nan_to_num(reduced), valid.to(positions.dtype)


def downsample_imf_descriptors(
    descriptors: torch.Tensor,
    validity: torch.Tensor,
    reliability: torch.Tensor,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validity-weighted reduction of bounded IMF relation descriptors."""
    if descriptors.ndim != 4 or validity.shape != descriptors.shape:
        raise ValueError("IMF descriptors and validity must be BxHxWxD")
    if reliability.shape != descriptors.shape[:3]:
        raise ValueError("IMF reliability must be BxHxW")
    values = descriptors.permute(0, 3, 1, 2)
    masks = validity.to(values.dtype).permute(0, 3, 1, 2)
    support = torch.nn.functional.adaptive_avg_pool2d(masks, size)
    numerator = torch.nn.functional.adaptive_avg_pool2d(values * masks, size)
    reduced = numerator / support.clamp_min(1e-8)
    reduced_validity = (support > 0).to(values.dtype)
    reduced_reliability = torch.nn.functional.adaptive_avg_pool2d(
        reliability[:, None].to(values.dtype), size
    )[:, 0]
    return (
        torch.nan_to_num(reduced.permute(0, 2, 3, 1).flatten(1, 2)),
        reduced_validity.permute(0, 2, 3, 1).flatten(1, 2),
        reduced_reliability.flatten(1),
    )


def positions_from_candidate_projection(
    candidate_grid: torch.Tensor,
    projected_depth_m: torch.Tensor,
    projection_validity: torch.Tensor,
    candidate_visibility: torch.Tensor,
    calibration: torch.Tensor,
    *,
    visible_code: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project prior-map tokens with valid current-camera projections.

    Candidate feature XYZ is egocentric and is never mistaken for map XYZ here.
    The projection path supplies current-image cells and optical-axis depth,
    which are converted to OpenCV camera-frame XYZ using current calibration.
    Only map tokens explicitly classified visible are admitted to attention.
    """
    if candidate_grid.ndim != 3 or candidate_grid.shape[-1] != 2:
        raise ValueError("candidate_grid must have shape BxMx2")
    batch, count, _ = candidate_grid.shape
    calibration = torch.as_tensor(
        calibration, dtype=candidate_grid.dtype, device=candidate_grid.device
    )
    if calibration.ndim == 1:
        calibration = calibration.unsqueeze(0).expand(batch, -1)
    if calibration.shape != (batch, 6):
        raise ValueError("Calibration must be [B,6]")
    width, height, fx, fy, cx, cy = calibration.unbind(dim=1)
    fx_grid, fy_grid = fx * (32 / width), fy * (32 / height)
    cx_grid = (cx + 0.5) * 32 / width - 0.5
    cy_grid = (cy + 0.5) * 32 / height - 0.5
    grid = candidate_grid.to(dtype=calibration.dtype)
    depth = projected_depth_m.to(dtype=calibration.dtype)
    x = (grid[..., 0] - cx_grid[:, None]) * depth / fx_grid[:, None]
    y = (grid[..., 1] - cy_grid[:, None]) * depth / fy_grid[:, None]
    positions = torch.stack((x, y, depth), dim=-1)
    valid = (projection_validity > 0) & (candidate_visibility == int(visible_code))
    valid = valid & torch.isfinite(positions).all(dim=-1) & (depth > 0)
    return torch.nan_to_num(positions), valid.to(calibration.dtype)


def grid_neighborhood(
    height: int, width: int, count: int, *, global_anchors: int = 4, device=None
) -> torch.Tensor:
    """Build deterministic fixed-grid neighborhoods without an NxN score tensor.

    The first slots preserve a small set of global anchor tokens. Remaining
    slots use nearest grid neighbors (distance, then row-major index tie-break).
    ``count >= H*W`` returns all tokens and is useful as a dense-equivalence
    reference. The neighborhood contains the query itself whenever count > 0.
    """
    total = height * width
    if count < 1:
        raise ValueError("Sparse attention neighborhood count must be positive")
    count = min(int(count), total)
    anchors = sorted(set((0, width - 1, total - 1, ((height - 1) // 2) * width + (width - 1) // 2)))
    rows = []
    for query in range(total):
        if count == total:
            rows.append(list(range(total)))
            continue
        qy, qx = divmod(query, width)
        ordered = sorted(
            range(total),
            key=lambda item: (
                (item // width - qy) ** 2 + (item % width - qx) ** 2,
                item,
            ),
        )
        selected = [query]
        for anchor in anchors[: max(0, global_anchors)]:
            if anchor not in selected and len(selected) < count:
                selected.append(anchor)
        for item in ordered:
            if item not in selected and len(selected) < count:
                selected.append(item)
        rows.append(selected)
    return torch.tensor(rows, dtype=torch.long, device=device)


def metric_neighborhood(
    positions: torch.Tensor,
    validity: torch.Tensor,
    candidate_indices: torch.Tensor,
    count: int,
    *,
    grid_shape: tuple[int, int] | None = None,
    global_anchors: int = 4,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select metric-nearest neighbors from a bounded spatial candidate pool.

    Work is ``O(B*N*P)`` for candidate-pool width P, rather than building an
    all-pairs distance matrix. The query and configured global anchors are
    retained; remaining supported candidates sort by 3-D distance. Unsupported
    geometry falls back deterministically to the candidate pool's grid order.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape BxNx3")
    batch, total, _ = positions.shape
    if grid_shape is None:
        side = int(total**0.5)
        if side * side != total:
            raise ValueError("grid_shape is required when the token grid is not square")
        grid_shape = (side, side)
    if grid_shape[0] * grid_shape[1] != total:
        raise ValueError("grid_shape must multiply to N")
    if validity.shape != (batch, total):
        raise ValueError("validity must have shape BxN")
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != total:
        raise ValueError("candidate_indices must have shape NxP")
    if not 1 <= count <= total:
        raise ValueError("neighbor count must be in [1,N]")
    pool = candidate_indices.shape[1]
    candidate_indices = candidate_indices.to(device=positions.device, dtype=torch.long)
    if query_indices is None:
        query_ids = torch.arange(total, device=positions.device)[None, :].expand(batch, -1)
        indices = candidate_indices.unsqueeze(0).expand(batch, -1, -1)
        query_positions = positions
        query_validity = validity > 0
    else:
        query_ids = query_indices.to(device=positions.device, dtype=torch.long)
        if query_ids.ndim != 2 or query_ids.shape[0] != batch:
            raise ValueError("query_indices must have shape BxQ")
        indices = candidate_indices[query_ids]
        batch_ids = torch.arange(batch, device=positions.device)[:, None]
        query_positions = positions[batch_ids, query_ids]
        query_validity = (validity > 0)[batch_ids, query_ids]
    if count == total:
        return indices

    batch_indices = torch.arange(batch, device=positions.device)[:, None, None]
    selected_positions = positions[batch_indices, indices]
    query_positions = query_positions[:, :, None, :]
    squared = (query_positions - selected_positions).square().sum(dim=-1)
    supported = validity > 0
    selected_support = torch.gather(supported, 1, indices.reshape(batch, -1)).reshape(
        batch, query_ids.shape[1], pool
    )
    pair_support = query_validity[:, :, None] & selected_support

    # Prefer supported metric neighbors. For unsupported pairs use their stable
    # grid-pool rank, after all supported neighbors.
    maximum = squared.masked_fill(~pair_support, 0.0).amax(dim=-1, keepdim=True)
    fallback = (
        maximum
        + 1.0
        + torch.arange(pool, device=positions.device, dtype=positions.dtype)[None, None]
    )
    order_score = torch.where(pair_support, squared, fallback)

    height, width = grid_shape
    anchor_ids = sorted(
        set(
            (
                0,
                width - 1,
                total - 1,
                ((height - 1) // 2) * width + (width - 1) // 2,
            )
        )
    )[: max(0, global_anchors)]
    query_count = query_ids.shape[1]
    forced = torch.zeros((batch, query_count, pool), dtype=torch.bool, device=positions.device)
    forced = forced | (indices == query_ids[:, :, None])
    for anchor in anchor_ids:
        forced = forced | (indices == anchor)
    forced_priority = (
        torch.arange(pool, device=positions.device, dtype=positions.dtype)[None] * 1e-5 - 1e6
    )
    order_score = torch.where(forced, forced_priority, order_score)
    selected_order = torch.argsort(order_score, dim=-1, stable=True)[..., :count]
    return torch.gather(indices, 2, selected_order)


def _relative_gather(relative_bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather relative bias [heads,Q,N] at [Q,K], without expanding scores."""
    heads, query_count, _ = relative_bias.shape
    if indices.ndim != 2 or indices.shape[0] != query_count:
        raise ValueError("Sparse indices must be QxK")
    return relative_bias.gather(2, indices[None].expand(heads, -1, -1))


def _relational_penalty(
    query_descriptors: torch.Tensor,
    key_descriptors: torch.Tensor,
    query_validity: torch.Tensor,
    key_validity: torch.Tensor,
    scales,
    weights,
    query_reliability: torch.Tensor | None = None,
    key_reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized IMF relational dissimilarity for selected token pairs.

    Each component is dimensionless after its configured scale. Missing
    components are omitted pairwise; if no component is shared, the prior is
    neutral. Reliability attenuates the prior rather than acting as a distance.
    """
    scale = torch.as_tensor(scales, dtype=query_descriptors.dtype, device=query_descriptors.device)
    alpha = torch.as_tensor(weights, dtype=query_descriptors.dtype, device=query_descriptors.device)
    if (
        scale.ndim != 1
        or alpha.shape != scale.shape
        or bool((scale <= 0).any())
        or bool((alpha < 0).any())
    ):
        raise ValueError("IMF dissimilarity scales must be positive and weights nonnegative")
    if query_descriptors.shape[-1] != len(scale) or key_descriptors.shape[-1] != len(scale):
        raise ValueError("IMF descriptor channel count does not match scales/weights")
    differences = (query_descriptors[:, :, None, :] - key_descriptors[:, None, :, :]) / scale
    shared = (query_validity[:, :, None, :] > 0) & (key_validity[:, None, :, :] > 0)
    weighted_support = shared.to(differences.dtype) * alpha
    weight_sum = weighted_support.sum(dim=-1)
    total_weight = alpha.sum().clamp_min(1e-12)
    squared = (differences.square() * weighted_support).sum(dim=-1) / weight_sum.clamp_min(1e-12)
    # The available-component fraction prevents a single surviving descriptor
    # from receiving the full prior strength when most measurements are absent.
    strength = weight_sum / total_weight
    if query_reliability is not None and key_reliability is not None:
        pair_reliability = (
            query_reliability[:, :, None].clamp(0, 1) * key_reliability[:, None, :].clamp(0, 1)
        ).sqrt()
        strength = strength * pair_reliability
    return squared * strength


def spatial_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    mode: str = "dense",
    positions: torch.Tensor | None = None,
    position_validity: torch.Tensor | None = None,
    relative_bias: torch.Tensor | None = None,
    position_bias_mode: str = "relative",
    metric_weight: float = 0.5,
    metric_scale_m: float = 4.0,
    neighbor_indices: torch.Tensor | None = None,
    key_validity: torch.Tensor | None = None,
    qk_normalize: bool = False,
    context_key: torch.Tensor | None = None,
    context_value: torch.Tensor | None = None,
    context_positions: torch.Tensor | None = None,
    context_validity: torch.Tensor | None = None,
    attention_debug: dict | None = None,
    profile_timing: bool = False,
    query_indices: torch.Tensor | None = None,
    query_validity: torch.Tensor | None = None,
    imf_descriptors: torch.Tensor | None = None,
    imf_validity: torch.Tensor | None = None,
    imf_reliability: torch.Tensor | None = None,
    imf_scales=(0.25, 0.25, 0.25, 0.25),
    imf_weights=(1.0, 0.5, 0.5, 0.5),
    imf_bias_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compute dense or actual gather-based attention over image tokens.

    Metric attention uses squared Euclidean camera-frame distance normalized by
    ``metric_scale_m``. Unsupported geometry contributes neutral (zero) bias.
    This is a partial spatial dissimilarity, not a total metric over invalid
    points. Sparse mode gathers K keys/values and forms only BxHxQxK scores.
    """
    if mode not in {"dense", "sparse"}:
        raise ValueError("attention mode must be dense or sparse")
    if position_bias_mode not in {"none", "relative", "metric", "both"}:
        raise ValueError("position_bias_mode must be none, relative, metric, or both")
    if metric_weight < 0 or metric_scale_m <= 0:
        raise ValueError("metric_weight must be nonnegative and metric_scale_m positive")
    batch, heads, query_count, dim = query.shape
    key_count = key.shape[2]
    if profile_timing and query.device.type == "cuda":
        torch.cuda.synchronize(query.device)
    attention_started = perf_counter() if profile_timing else 0.0
    context_attention_ms = 0.0
    sparse_gather_ms = 0.0
    if key.shape[:2] != query.shape[:2] or key.shape[-1] != dim:
        raise ValueError("Q and K must have matching batch/head/projected dimensions")
    if value.shape[:3] != (batch, heads, key_count):
        raise ValueError("V must use the same batch, head, and key-token dimensions")
    if (context_key is None) != (context_value is None):
        raise ValueError("Map context must provide both key and value projections")
    if context_key is not None:
        if context_key.shape[:2] != query.shape[:2] or context_key.shape[-1] != dim:
            raise ValueError("Context K must match spatial Q/K batch, head, and projected width")
        if context_value.shape[:2] != value.shape[:2] or context_value.shape[-1] != value.shape[-1]:
            raise ValueError("Context V must match spatial value batch, head, and value width")
    if key_count != grid_shape[0] * grid_shape[1]:
        raise ValueError("Token count does not match attention grid")
    if query_validity is not None and query_validity.shape != (batch, query_count):
        raise ValueError("query_validity must have shape BxQ")
    if query_indices is not None and query_indices.shape != (batch, query_count):
        raise ValueError("query_indices must have shape BxQ")

    def query_points():
        points = positions.reshape(batch, key_count, 3)
        if query_indices is None:
            return points
        batch_ids = torch.arange(batch, device=points.device)[:, None]
        return points[batch_ids, query_indices]

    def query_position_support():
        if position_validity is None:
            return torch.ones((batch, query_count), dtype=torch.bool, device=query.device)
        support = position_validity.reshape(batch, key_count) > 0
        if query_indices is None:
            return support
        batch_ids = torch.arange(batch, device=support.device)[:, None]
        return support[batch_ids, query_indices]

    if qk_normalize:
        query = torch.nn.functional.normalize(query, dim=-1, eps=1e-6)
        key = torch.nn.functional.normalize(key, dim=-1, eps=1e-6)
        if context_key is not None:
            context_key = torch.nn.functional.normalize(context_key, dim=-1, eps=1e-6)

    use_relative = position_bias_mode in {"relative", "both"} and relative_bias is not None
    use_metric = position_bias_mode in {"metric", "both"} and positions is not None
    use_imf_dissimilarity = imf_descriptors is not None and imf_bias_weight > 0
    if use_imf_dissimilarity:
        if (
            imf_validity is None
            or imf_descriptors.ndim != 3
            or imf_validity.shape != imf_descriptors.shape
        ):
            raise ValueError("IMF descriptors and per-component validity must both be BxNxD")
        if imf_descriptors.shape[:2] != (batch, key_count):
            raise ValueError("IMF descriptors must align with all spatial keys")
        if (
            len(imf_scales) != imf_descriptors.shape[-1]
            or len(imf_weights) != imf_descriptors.shape[-1]
        ):
            raise ValueError("IMF dissimilarity scales and weights must match descriptor width")
    scale = dim**-0.5

    if mode == "dense":
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if use_relative:
            scores = scores + (
                relative_bias if relative_bias.ndim == 4 else relative_bias.unsqueeze(0)
            )
        if use_metric:
            points = query_points()
            key_points = positions.reshape(batch, key_count, 3)
            squared = torch.cdist(points.float(), key_points.float(), p=2).square().to(scores.dtype)
            supported = (
                position_validity.reshape(batch, key_count) > 0
                if position_validity is not None
                else torch.ones((batch, key_count), dtype=torch.bool, device=scores.device)
            )
            pair_valid = query_position_support()[:, :, None] & supported[:, None, :]
            penalty = metric_weight * squared / (metric_scale_m**2)
            scores = scores - torch.where(pair_valid, penalty, torch.zeros_like(penalty))[:, None]
        if use_imf_dissimilarity:
            qdesc = imf_descriptors
            qvalid = imf_validity
            qrel = imf_reliability
            if query_indices is not None:
                batch_ids = torch.arange(batch, device=query.device)[:, None]
                qdesc = qdesc[batch_ids, query_indices]
                qvalid = qvalid[batch_ids, query_indices]
                if qrel is not None:
                    qrel = qrel[batch_ids, query_indices]
            relational = _relational_penalty(
                qdesc,
                imf_descriptors,
                qvalid,
                imf_validity,
                imf_scales,
                imf_weights,
                qrel,
                imf_reliability,
            )
            scores = scores - float(imf_bias_weight) * relational[:, None]
        if key_validity is not None:
            scores = scores.masked_fill(key_validity[:, None, None, :] <= 0, -1e4)
        values = value
        if context_key is not None and context_value is not None:
            if profile_timing and query.device.type == "cuda":
                torch.cuda.synchronize(query.device)
            context_started = perf_counter() if profile_timing else 0.0
            context_scores = torch.matmul(query, context_key.transpose(-2, -1)) * scale
            if use_metric and context_positions is not None:
                points = query_points()
                map_points = context_positions.reshape(batch, -1, 3)
                squared = (
                    torch.cdist(points.float(), map_points.float(), p=2).square().to(scores.dtype)
                )
                map_valid = (
                    context_validity > 0
                    if context_validity is not None
                    else torch.ones(map_points.shape[:2], dtype=torch.bool, device=scores.device)
                )
                image_valid = query_position_support()
                pair_valid = image_valid[:, :, None] & map_valid[:, None, :]
                penalty = metric_weight * squared / (metric_scale_m**2)
                context_scores = (
                    context_scores
                    - torch.where(pair_valid, penalty, torch.zeros_like(penalty))[:, None]
                )
            if context_validity is not None:
                context_scores = context_scores.masked_fill(
                    context_validity[:, None, None, :] <= 0, -1e4
                )
            scores = torch.cat((scores, context_scores), dim=-1)
            values = torch.cat((value, context_value), dim=2)
            context_pairs = query_count * context_key.shape[2]
            if profile_timing:
                if query.device.type == "cuda":
                    torch.cuda.synchronize(query.device)
                context_attention_ms = (perf_counter() - context_started) * 1000.0
        else:
            context_pairs = 0
        weights = scores.softmax(dim=-1)
        if attention_debug is not None:
            attention_debug["weights"] = weights.detach().float().cpu()
        result = torch.matmul(weights, values)
        pair_count = batch * heads * (query_count * key_count + context_pairs)
    else:
        if neighbor_indices is None:
            neighbor_indices = grid_neighborhood(
                *grid_shape, min(key_count, 32), device=query.device
            )
        indices = neighbor_indices.to(device=query.device, dtype=torch.long)
        if indices.ndim == 2:
            indices = indices.unsqueeze(0).expand(batch, -1, -1)
        if indices.shape[:2] != (batch, query_count):
            raise ValueError("neighbor_indices must be QxK or BxQxK")
        k_count = indices.shape[-1]
        batch_index = torch.arange(batch, device=query.device)[:, None, None, None]
        head_index = torch.arange(heads, device=query.device)[None, :, None, None]
        gather_started = perf_counter() if profile_timing else 0.0
        local_key = key[batch_index, head_index, indices[:, None, :, :]]
        local_value = value[batch_index, head_index, indices[:, None, :, :]]
        if profile_timing:
            if query.device.type == "cuda":
                torch.cuda.synchronize(query.device)
            sparse_gather_ms = (perf_counter() - gather_started) * 1000.0
        scores = (query.unsqueeze(-2) * local_key).sum(dim=-1) * scale
        if use_relative:
            if relative_bias.shape[-1] != k_count:
                raise ValueError("Sparse relative bias must already be gathered to NxK")
            scores = scores + (
                relative_bias if relative_bias.ndim == 4 else relative_bias.unsqueeze(0)
            )
        if use_metric:
            points = query_points()
            key_points = positions.reshape(batch, key_count, 3)
            batch_index = torch.arange(batch, device=query.device)[:, None, None]
            selected_positions = key_points[batch_index, indices]
            query_positions = points[:, :, None, :]
            squared = (query_positions - selected_positions).square().sum(dim=-1)
            if position_validity is not None:
                supported = position_validity.reshape(batch, key_count) > 0
                selected_validity = torch.gather(supported, 1, indices.reshape(batch, -1)).reshape(
                    batch, query_count, k_count
                )
                pair_valid = query_position_support()[:, :, None] & selected_validity
                squared = torch.where(pair_valid, squared, torch.zeros_like(squared))
            scores = scores - metric_weight * squared[:, None] / (metric_scale_m**2)
        if use_imf_dissimilarity:
            batch_ids = torch.arange(batch, device=query.device)[:, None]
            qdesc = (
                imf_descriptors[batch_ids, query_indices]
                if query_indices is not None
                else imf_descriptors
            )
            qvalid = (
                imf_validity[batch_ids, query_indices]
                if query_indices is not None
                else imf_validity
            )
            qrel = imf_reliability
            if query_indices is not None and qrel is not None:
                qrel = qrel[batch_ids, query_indices]
            token_batch_index = torch.arange(batch, device=query.device)[:, None, None]
            selected_desc = imf_descriptors[token_batch_index, indices]
            selected_valid = imf_validity[token_batch_index, indices]
            selected_reliability = (
                imf_reliability.gather(1, indices.reshape(batch, -1)).reshape(
                    batch, query_count, k_count
                )
                if imf_reliability is not None
                else None
            )
            # The pair helper accepts BxQxD and BxQxKxD here, so form the
            # selected-pair expression directly without an all-pairs tensor.
            scales = torch.as_tensor(imf_scales, dtype=qdesc.dtype, device=qdesc.device)
            alphas = torch.as_tensor(imf_weights, dtype=qdesc.dtype, device=qdesc.device)
            diff = (qdesc[:, :, None, :] - selected_desc) / scales
            shared = (qvalid[:, :, None, :] > 0) & (selected_valid > 0)
            support = shared.to(diff.dtype) * alphas
            support_sum = support.sum(-1)
            dissimilarity_sq = (diff.square() * support).sum(-1) / support_sum.clamp_min(1e-12)
            strength = support_sum / alphas.sum().clamp_min(1e-12)
            if qrel is not None and selected_reliability is not None:
                strength = (
                    strength
                    * (qrel[:, :, None].clamp(0, 1) * selected_reliability.clamp(0, 1)).sqrt()
                )
            scores = scores - float(imf_bias_weight) * (dissimilarity_sq * strength)[:, None]
        if key_validity is not None:
            gathered_validity = key_validity.gather(1, indices.reshape(batch, -1)).reshape(
                batch, query_count, k_count
            )
            scores = scores.masked_fill(gathered_validity[:, None] <= 0, -1e4)
        local_value_all = local_value
        if context_key is not None and context_value is not None:
            if profile_timing and query.device.type == "cuda":
                torch.cuda.synchronize(query.device)
            context_started = perf_counter() if profile_timing else 0.0
            context_scores = torch.matmul(query, context_key.transpose(-2, -1)) * scale
            if use_metric and context_positions is not None:
                points = query_points()
                map_points = context_positions.reshape(batch, -1, 3)
                squared = (points[:, :, None, :] - map_points[:, None, :, :]).square().sum(dim=-1)
                map_valid = (
                    context_validity > 0
                    if context_validity is not None
                    else torch.ones(map_points.shape[:2], dtype=torch.bool, device=scores.device)
                )
                image_valid = query_position_support()
                pair_valid = image_valid[:, :, None] & map_valid[:, None, :]
                penalty = metric_weight * squared / (metric_scale_m**2)
                context_scores = (
                    context_scores
                    - torch.where(pair_valid, penalty, torch.zeros_like(penalty))[:, None]
                )
            if context_validity is not None:
                context_scores = context_scores.masked_fill(
                    context_validity[:, None, None, :] <= 0, -1e4
                )
            scores = torch.cat((scores, context_scores), dim=-1)
            context_value_by_query = context_value[:, :, None, :, :].expand(
                batch, heads, query_count, context_value.shape[2], context_value.shape[-1]
            )
            local_value_all = torch.cat((local_value, context_value_by_query), dim=3)
            context_pairs = query_count * context_key.shape[2]
            if profile_timing:
                if query.device.type == "cuda":
                    torch.cuda.synchronize(query.device)
                context_attention_ms = (perf_counter() - context_started) * 1000.0
        else:
            context_pairs = 0
        weights = scores.softmax(dim=-1)
        if attention_debug is not None:
            attention_debug["weights"] = weights.detach().float().cpu()
        result = (weights.unsqueeze(-1) * local_value_all).sum(dim=-2)
        pair_count = (
            batch
            * heads
            * query_count
            * (k_count + (context_key.shape[2] if context_key is not None else 0))
        )
    if query_validity is not None:
        effective_queries = int(query_validity.sum().item())
    else:
        effective_queries = query_count
    dense_reference = (
        batch
        * heads
        * key_count
        * (key_count + (context_key.shape[2] if context_key is not None else 0))
    )
    if profile_timing:
        if query.device.type == "cuda":
            torch.cuda.synchronize(query.device)
        attention_ms = (perf_counter() - attention_started) * 1000.0
    else:
        attention_ms = 0.0
    return result, {
        "tokens": query_count,
        "key_tokens": key_count,
        "active_queries": effective_queries,
        "imf_dissimilarity_pairs": (
            batch
            * heads
            * query_count
            * (key_count if mode == "dense" else neighbor_indices.shape[-1])
            if use_imf_dissimilarity
            else 0
        ),
        "attention_pairs": pair_count,
        "dense_reference_pairs": dense_reference,
        "context_tokens": context_key.shape[2] if context_key is not None else 0,
        "neighbors_per_query": neighbor_indices.shape[-1]
        if neighbor_indices is not None
        else key_count,
        "attention_ms": attention_ms,
        "sparse_gather_ms": sparse_gather_ms,
        "context_attention_ms": context_attention_ms,
    }


def topology_smoothness_loss(
    embeddings: torch.Tensor, affinity: torch.Tensor, validity: torch.Tensor | None = None
) -> torch.Tensor:
    """Optional graph smoothness loss; caller controls the default zero weight."""
    if embeddings.ndim != 3 or affinity.shape != (
        embeddings.shape[0],
        embeddings.shape[1],
        embeddings.shape[1],
    ):
        raise ValueError("Expected BxNxD embeddings and BxNxN affinity")
    symmetric = (affinity + affinity.transpose(1, 2)) * 0.5
    if validity is not None:
        pair = validity[:, :, None] * validity[:, None, :]
        symmetric = symmetric * pair
    differences = embeddings[:, :, None, :] - embeddings[:, None, :, :]
    return (symmetric[..., None] * differences.square()).sum() / symmetric.sum().clamp_min(1.0)
