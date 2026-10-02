# Motion measurements and implementation record

Measured 2026-09-26 on an Apple M3 Pro, macOS arm64, AppleClang 21,
OpenCV 5.0.0 (Homebrew), CMake Release `-O3 -DNDEBUG`, one OpenCV thread.
These are host measurements, not Arduino UNO Q / SBC measurements. No electrical
power measurements were available and no watts or battery savings are claimed.

## Reproduce

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DBUILD_MOTION_CAMERA=ON
cmake --build build -j2
./artifacts/build/motion_benchmark --frames 1000 --debug /tmp/imagination-motion-demo
./artifacts/build/motion_benchmark --frames 1000 --levels 0
./artifacts/build/motion_benchmark --frames 1000 --forward-backward
./artifacts/build/motion_benchmark --frames 1000 --triangulate
```

Each run uses a deterministic 320x240 grayscale texture with smooth translation
and yaw, `fx=240`, `fy=230`, camera height 2 m, synthetic frame interval 0.1 s,
and a maximum of 100 features. Time includes `processFrame`, excludes image
generation and debug PNG writing, and excludes 20 warmup frames from the 1,000
reported frames. Cold startup is reported separately. Synthetic pose error
checks are not estimates of real-world accuracy.

## Final measurements

All times below are milliseconds per processed frame.

| Configuration | Mean | Median | p95 | Worst | Process CPU mean | Valid frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default: base + half-resolution LK | 0.2131 | 0.2153 | 0.2370 | 0.2629 | 0.2135 | 1000/1000 |
| Base only (`--levels 0`) | 0.1199 | 0.1191 | 0.1324 | 0.1539 | 0.1204 | 1000/1000 |
| Default + backward consistency | 0.3709 | 0.3760 | 0.4159 | 0.5512 | 0.3712 | 1000/1000 |
| Default + selective triangulation | 0.2127 | 0.2145 | 0.2385 | 0.2712 | 0.2132 | 1000/1000 |

The tiny triangulation-run difference is timing noise, not a speedup.
The default run averaged 74.499 attempted tracks and 74.477 geometric inliers;
zero replenishments were needed during measured frames (one initial detection
in warmup). Mean displacement error was 0.0000470 m on this synthetic sequence.
The triangulation run emitted 78 landmarks over time with a cache cap of 64.

Default mean stage timings:

| Stage | ms/frame |
| --- | ---: |
| Grayscale/input preparation | 0.0018 |
| Feature detection | <0.0001 (no measured-frame detections) |
| Pyramids, gradients, and LK | 0.2048 |
| LK status/error/border rejection | 0.0003 |
| Robust statistics and geometry | 0.0059 |
| Pose bookkeeping, triangulation disabled | <0.0001 |

The first run's cold initialization took 4.5554 ms, including 3.5229 ms for
detection. Later process runs in this batch had initial detection around
1.3–1.8 ms. Startup/cache variation matters; do not substitute the tiny
amortized detection number for the cost of an actual detection event.
Stage clocks introduce a small overhead and their rounded sum need not equal
the total. Reported processing capacity (~4,693 frames/s here) excludes camera
I/O and is not capture FPS or an achievable SBC rate.

## Why keep one extra pyramid level?

The comparison sequence scales the sinusoidal motion rate with `--speed`.
Its texture and warp are deliberately identical between configurations.

| Sequence (500 measured frames) | maxLevel | Mean ms | Valid frames | Detection runs including warmup |
| --- | ---: | ---: | ---: | ---: |
| `--speed 4`, roughly 3 px/frame | 0 | 0.1570 | 500/500 | 15 |
| Same | 1 | 0.2204 | 500/500 | 1 |
| `--speed 6`, roughly 5 px/frame | 0 | 0.3665 | 164/500 | 100 |
| Same | 1 | 0.1956 | 500/500 | 1 |
| `--speed 12`, roughly 10 px/frame | 0 | 0.3003 | 13/500 | 104 |
| Same | 1 | 0.3870 | 64/500 | 104 |
| Same | 2 | 0.2944 | 500/500 | 3 |

These are not universal LK convergence limits. They show why a lower per-track
cost can lose its advantage through tracking failures and repeated detection.
Use level 0 only after verifying sufficiently small inter-frame displacement
on the actual camera/texture. The default one extra level avoids the observed
5-pixel failures. Level 2 is exposed for faster motion at added compute cost.
At roughly 10 pixels/frame the default intentionally returns many invalid
results; it does not claim reliable high-speed tracking. Failed configurations
return a nonzero benchmark exit code by design.

## Review and verification

The compute review retained previous pyramids/derivatives, reserved bounded
track/model buffers, kept detection out of healthy frames, removed a redundant
copy on grayscale downscaling, and avoided backward buffers when that check
is disabled. Model fitting dominates neither the measured time nor storage;
skipping current yaw estimation would save little and bias moving-camera
geometry. No descriptor, dense flow, depth image, or generic pose solver runs.

The sanitizer run exposed an ABI boundary between instrumented reserved STL
vectors and the uninstrumented packaged OpenCV library. Corner output now
uses a Mat, and LK/pyramid vector outputs are sized on the caller side before
the OpenCV call. AddressSanitizer container checks remain enabled.

- Full build includes RGB-D, stereo, motion benchmark, and optional webcam runner.
- 30 regression cases passed (22 existing and 8 new motion cases).
- Both CLI checks passed: synthetic motion and existing demo JSON round-trip.
- Address/undefined-behavior/float-cast-overflow sanitizer build passed all checks.
- Motion without stereo and RGB-D executables configured, built, and passed tests.
- RGB-D with both stereo and motion disabled configured, built, and passed tests.
- The original benchmark did not modify stereo, terrain or landing algorithms.
- Optional debug PNG was generated and visually inspected.
- Webcam capture and physical sensor acquisition were not exercised; the webcam
  executable was compiled. Real sensor acquisition remains caller-owned.
