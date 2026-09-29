## Coordinate conventions

- Distances and elevations are in metres; legacy landing slope is in degrees.
  Analytical surface slope is in radians, then normalized by π/2 in the tensor.
- World `+Z` points upward. Grid columns increase toward `+X`; rows toward `-Y`.
- Grid validity is `255` for measured, `127` for interpolated, and `0` for unknown.
- OpenCV camera coordinates use `+X` right, `+Y` down, and `+Z` forward.
- The stereo path assumes nadir captures and uses camera 1 as the world origin.
- An unknown or incomplete landing footprint is a hazard, not a safe default.

These conventions are shared across modules; changing one requires checking
reconstruction, mapping, landing analysis, and exports together.

Analytical mode uses an exact configurable working rectangle (default 256×256)
and output grid (32×32). Resizing updates both focal lengths and principal points
using pixel-center offsets. Its tensor is channel-major and aligned to the
**current rectified camera image**; it is never a resized top-down DEM.
With configured distortion, `pixel_to_source` describes the affine mapping to the
rectified source-sized pinhole image, not the nonlinear mapping to raw distorted
sensor pixels. The source calibration and distortion define that nonlinear map.

Dense Farneback estimates forward scene displacement (+U right, +V down) in
working pixels; values are scattered to current endpoints before pooling.
Depth is camera optical-axis Z, in metres before fixed normalization. Depth
X/Y gradients use local image-plane m/m units (`Sobel(depth)*fx_or_fy/depth`).
World-terrain slope is measured on the metric XY grid and projected separately.
Essential geometry uses `X_camera2 = R21*X_camera1 + t_direction`; relative points
have unit-baseline units and never enter metre-valued terrain without valid scale.
