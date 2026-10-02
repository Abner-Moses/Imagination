"""Machine-readable contract for the C++ IMF feature tensor."""

from __future__ import annotations

import json
from pathlib import Path

from common.registry import ANALYTICAL_CHANNELS

FEATURE_SPEC_VERSION = "imagination-feature-normalization-v1"

FUSION_FAMILIES = {
    "appearance": tuple(range(0, 20)),
    "motion": (20, 21),
    "geometry": tuple(range(22, 28)),
}

# Bounds and units describe the deterministic, clipped values written by
# VisualFeatureExtractor::Impl::normalizeChannel. Config-scaled quantities use
# the defaults in data/configs/pre_extraction.yaml.
_CHANNEL_RULES = {
    "Y": ("appearance", "unitless_luma", [0.0, 1.0], "natural_[0,1]", "all pixels"),
    "Cb": ("appearance", "unitless_chroma", [0.0, 1.0], "natural_[0,1]", "all pixels"),
    "Cr": ("appearance", "unitless_chroma", [0.0, 1.0], "natural_[0,1]", "all pixels"),
    "Gx": (
        "appearance",
        "normalized_intensity_per_working_pixel",
        [-1.0, 1.0],
        "fixed_2x_sobel_then_clip",
        "all pixels",
    ),
    "Gy": (
        "appearance",
        "normalized_intensity_per_working_pixel",
        [-1.0, 1.0],
        "fixed_2x_sobel_then_clip",
        "all pixels",
    ),
    "GradientMagnitude": (
        "appearance",
        "normalized_gradient_magnitude",
        [0.0, 1.0],
        "fixed_sqrt2_scale_then_clip",
        "all pixels",
    ),
    "HarrisResponse": (
        "appearance",
        "fixed_scaled_signed_sqrt_response",
        [-1.0, 1.0],
        "configured_harris_scale_then_clip",
        "all pixels",
    ),
    "CannyEdge": (
        "appearance",
        "binary",
        [0.0, 1.0],
        "binary_0_or_1",
        "all pixels; non-edge is measured zero",
    ),
    "ContourMap": (
        "appearance",
        "binary",
        [0.0, 1.0],
        "binary_0_or_1",
        "all pixels; non-contour is measured zero",
    ),
    "ChromaGradientCb": (
        "appearance",
        "normalized_chroma_gradient",
        [0.0, 1.0],
        "fixed_sqrt2_scale_then_clip",
        "all pixels",
    ),
    "ChromaGradientCr": (
        "appearance",
        "normalized_chroma_gradient",
        [0.0, 1.0],
        "fixed_sqrt2_scale_then_clip",
        "all pixels",
    ),
    "OpticalFlowU": (
        "motion",
        "working_pixels_per_frame",
        [-1.0, 1.0],
        "divide_by_flow_scale_px_then_clip",
        "valid only where dense-flow support exists",
    ),
    "OpticalFlowV": (
        "motion",
        "working_pixels_per_frame",
        [-1.0, 1.0],
        "divide_by_flow_scale_px_then_clip",
        "valid only where dense-flow support exists",
    ),
    "Depth": (
        "geometry",
        "metres",
        [0.0, 1.0],
        "divide_by_depth_scale_m_then_clip",
        "valid reconstructed current-view geometry only",
    ),
    "DepthGradientX": (
        "geometry",
        "dimensionless_relative_depth_gradient",
        [-1.0, 1.0],
        "sobel_depth_times_focal_over_depth_then_fixed_scale_and_clip",
        "valid 3x3 reconstructed current-view support only",
    ),
    "DepthGradientY": (
        "geometry",
        "dimensionless_relative_depth_gradient",
        [-1.0, 1.0],
        "sobel_depth_times_focal_over_depth_then_fixed_scale_and_clip",
        "valid 3x3 reconstructed current-view support only",
    ),
    "Slope": (
        "geometry",
        "radians",
        [0.0, 1.0],
        "multiply_by_2_over_pi_then_clip",
        "valid reconstructed surface only",
    ),
    "Roughness": (
        "geometry",
        "metres",
        [0.0, 1.0],
        "divide_by_roughness_scale_m_then_clip",
        "valid reconstructed surface only",
    ),
    "GeometryConfidence": (
        "geometry",
        "support_score_not_calibrated_probability",
        [0.0, 1.0],
        "natural_bounded_score",
        "zero is valid only when extraction completed and found no support",
    ),
}


def feature_specification() -> dict:
    """Return the exact ordered channel contract, including generated HOG bins."""
    channels = []
    for index, name in enumerate(ANALYTICAL_CHANNELS):
        family, units, expected_range, normalization, validity = _CHANNEL_RULES.get(
            name,
            (
                "appearance",
                "unitless_orientation_energy",
                [0.0, 1.0],
                "per_cell_l2_normalized_histogram",
                "all pixels",
            ),
        )
        channels.append(
            {
                "index": index,
                "name": name,
                "family": family,
                "units": units,
                "expected_range": expected_range,
                "normalization": normalization,
                "validity_semantics": validity,
            }
        )
    return {
        "version": FEATURE_SPEC_VERSION,
        "shape": [28, 32, 32],
        "storage": "channel-major float32; validity is same-shape binary",
        "source": "IMF_HTransformer/imf/cpp/vision.cpp::normalizeChannel",
        "fusion_families": {
            name: [ANALYTICAL_CHANNELS[index] for index in indices]
            for name, indices in FUSION_FAMILIES.items()
        },
        "channels": channels,
    }


def write_feature_spec(path: str | Path) -> None:
    Path(path).write_text(json.dumps(feature_specification(), indent=2) + "\n", encoding="utf-8")


def validate_feature_batch(features, validity, atol: float = 1e-5) -> None:
    """Raise if a batch violates fixed ranges or binary validity semantics."""
    if features.shape != validity.shape or tuple(features.shape[1:]) != (28, 32, 32):
        raise ValueError("Expected matching Bx28x32x32 feature and validity tensors")
    if not features.is_floating_point():
        raise TypeError("IMF features must be floating point")
    if not validity.is_floating_point():
        raise TypeError("IMF validity must be floating point")
    if not bool(((validity - validity.round()).abs() <= atol).all()):
        raise ValueError("IMF validity must be binary")
    for channel in feature_specification()["channels"]:
        index = channel["index"]
        low, high = channel["expected_range"]
        valid_values = features[:, index][validity[:, index] > 0]
        if valid_values.numel() and (
            not bool(valid_values.isfinite().all())
            or float(valid_values.min()) < low - atol
            or float(valid_values.max()) > high + atol
        ):
            raise ValueError(f"IMF channel {channel['name']} violates [{low}, {high}]")
