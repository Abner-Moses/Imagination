# Optical flow with IMU and ultrasonic input

The calibrated motion tracker now accepts both sensors. This is a cheap constrained
geometric estimator, not a general inertial navigation filter. The uncalibrated
`PixelFlowTracker` continues to return only image-space measurements.

```text
Camera frames → sparse LK correspondences
                        +
IMU yaw change → fixed-rotation geometric consensus
                        +
Ultrasonic camera height → metric planar displacement → velocity

IMU roll/pitch, timestamps and image residuals → validity checks
```

## Camera command

### Arduino UNO Q, MPU-6050 and HC-SR04

Run Imagination/OpenCV on the UNO Q's Linux processor. The recommended acquisition
path is an STM32-side sketch reading the MPU-6050 over I²C and timing the HC-SR04
echo, with timestamped results forwarded through Arduino Bridge to Linux. This
matches the board's [Linux/MCU architecture](https://docs.arduino.cc/hardware/uno-q/).
It is a proposed hardware connection, not an included or hardware-tested driver.
The current implementation starts at the snapshot/direct API boundary below;
the acquisition sketch, attitude filter and Bridge receiver remain to be connected.

The [MPU-6050 specification](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet.pdf)
describes an accelerometer and gyroscope, with support for an **external**
magnetometer. There is no magnetometer inside the MPU-6050 itself. A breakout may
have a separate chip; its model must be identified before using magnetic heading.
Absolute heading is not required here: calibrated gyro-derived yaw increments
can supply the inter-frame rotation, but bias causes drift. Roll/pitch still need
an upstream attitude estimate. Do not pass raw gyro rates as angles.

Keep sensor acquisition independent of the 10 Hz camera loop so gyro samples are
not discarded between frames. Map MCU timestamps into the host monotonic clock;
Bridge receipt time alone is not measurement time. Sensor wiring, voltage levels,
mounting orientation and echo timeout handling belong in the acquisition adapter.

### Running the estimator

```sh
cmake -S . -B build -DBUILD_MOTION=ON -DBUILD_MOTION_CAMERA=ON
cmake --build build -j2
./build/imagination optical_flow 0 300 output/camera_flow \
  --camera configs/motion_camera.yaml \
  --imu /tmp/imu.txt --ultrasonic /tmp/range.txt
```

Fill the camera YAML with real calibration at the requested capture size. The
calibrated path rejects a delivered size mismatch and assumes rectified frames.
For existing callers this is equivalent to:

```sh
./build/imagination motion_camera configs/motion_camera.yaml 0 300 \
  --imu /tmp/imu.txt --ultrasonic /tmp/range.txt
```

`--imu` makes a fresh IMU pair mandatory for metric output; a missing file never
silently switches the command back to visual-only metric velocity. Without
`--imu`, the previous visual-yaw + optional-altitude behavior is retained.
The metric CSV appends `imu_used` and `imu_status` to its existing columns.
`valid` describes visual consensus; metric fields can still be blank. Check the
metric fields/status before using a velocity in metres/second.

## Snapshot formats and clocks

Each snapshot contains exactly one whitespace-separated record:

```text
# imu.txt (four numbers, no comment line in the actual file)
timestamp_s roll_rad pitch_rad yaw_rad

# range.txt (three numbers)
timestamp_s vertical_range_m sensor_offset_world_z_m
```

The sensor acquisition program should write a temporary file and atomically rename
it over the snapshot. Partial records, extra fields and missing files are treated
as unavailable. Sample values must come from the sensors, not constant example
files. The IMU timestamp is the measurement time, not the time the file is written.

Use the same host monotonic seconds clock as the camera runner's `steady_clock`
timestamp. Convert device boot counters/ticks into that clock in the acquisition
adapter. The runner timestamps frame delivery. For accurate rotation compensation,
the acquisition/flight-controller layer should interpolate attitude to actual
capture/exposure times and feed the direct API. The lightweight file adapter reads
the latest available sample; it does not buffer/interpolate high-rate IMU packets.

The default IMU age limit is **20 ms** and range age limit **150 ms**. Future-dated,
nonfinite, reused or missing IMU samples cannot provide a yaw change. Two fresh
attitudes are required after startup, reset or loss of usable attitude history.
Range must be fresh at both image endpoints for metric scale.

## Axes and sensor mounting

`ImuSample` contains attitude already estimated by the IMU/flight controller, in
**radians**, transformed for the camera mount:

- Roll and pitch zero mean the camera faces downward.
- Yaw increases counterclockwise about world **+Z upward**, matching `NadirPose`.
- A fixed yaw offset cancels between frames; roll/pitch mounting offsets do not.
- Convert NED/body-frame attitude into this convention upstream. Do not pass a
  controller's raw yaw convention without checking axes and signs.

Raw accelerometer counts or angular-rate samples are not attitude. This stage does
not integrate acceleration or implement a gyro/accelerometer orientation filter.
For a raw-output IMU, its driver or flight controller must estimate attitude first.

Ultrasonic input is the sensor's **vertical distance to locally horizontal ground**.
The offset is the sensor position relative to the camera center, in world +Z axes:

`camera_height = vertical_range - sensor_offset_world_z`.

If the beam is tilted, the acquisition adapter must project slant range into the
vertical direction and rotate the mounting offset into world axes. A horizontal
lever arm is harmless only under the locally horizontal-plane assumption. Existing
`altitudeFromUltrasonic()` performs the camera-height conversion and validation.

## Direct C++ input

```cpp
metric_mapping::OpticalFlowSettings settings;
settings.require_imu = true;
metric_mapping::SparseFlowTracker tracker(camera, settings);

metric_mapping::ImuSample attitude{roll_rad, pitch_rad, yaw_rad, imu_timestamp_s};
auto altitude = metric_mapping::altitudeFromUltrasonic(range_measurement, range_timestamp_s);
auto motion = tracker.processFrame(frame, frame_timestamp_s, altitude, attitude);
if (motion.valid && motion.imu_used && motion.planar_velocity_mps) {
    // Use the local z-up planar velocity, subject to the documented assumptions.
}
```

The stateful visual-feature API accepts the same fourth argument:
`extractor.extract(bgr, frame_timestamp_s, altitude, attitude)`.
Set `visual_settings.motion.require_imu = true` for strict sensor-assisted motion.
The typed API uses measurements directly and has no sensor-file I/O in the hot path.

## Mathematics and limits

Normalize pixels with calibration: `q = ((u-cx)/fx, (v-cy)/fy)`.
For two accepted, frame-aligned attitudes:

`theta = remainder(yaw_current - yaw_previous, 2*pi)`.

The robust image model becomes `q2 = s*R(theta)*q1 + t`. Rotation is supplied by
the IMU; only scale and translation are fitted. For centered inlier observations
`P` and `Q`:

`s = sum((R(theta)*P) dot Q) / sum(P dot P)`

`t = mean(q2) - s*R(theta)*mean(q1)`.

The existing bounded consensus fallback and residual/coverage checks still reject
moving objects and inconsistent sensor/image geometry. With current camera height
`h2` and `D = diag(1,-1)` converting camera-down Y to local z-up Y:

`planar_displacement = -h2 * D * transpose(R(theta)) * t`

`planar_velocity = planar_displacement / frame_dt`.

The observed scale must also agree with `previous_height/current_height` within
the existing 5% tolerance. A good IMU yaw does not make an incorrect range valid.

The estimator deliberately retains the near-nadir model. The defaults gate
`hypot(roll,pitch)` at **5 degrees**, and inter-frame roll/pitch change at
**0.3 degrees**. Excessive tilt/change suppresses metric output and interrupts
the local metric map chain; it is not misreported as horizontal translation.
This does not compensate general roll/pitch motion. Static small tilt, IMU bias,
timestamp error, nonplanar terrain and moving-majority scenes remain limitations.
There is no dead-reckoning translation from accelerometer integration.

Sensor-assisted fitting adds one yaw sine/cosine pair and scalar scale projection
to the existing sparse geometry pass. No neural model, Kalman-filter framework,
extra image pyramid, GPU or new dependency is introduced. Sensor drivers are
hardware-specific and stay outside these ten C++ files until the hardware protocol
is established; the snapshot/direct interfaces are ready for those drivers.

To profile the sensor-assisted path using synthetic, perfectly synchronized
attitude and height (not physical sensor readings):

```sh
./build/imagination motion_benchmark --frames 500 --imu
```

Measured on the development Apple M3 Pro, Release/OpenCV 5.0.0, one OpenCV
thread, 320×240, 500 measured frames after 20 warmup frames: mean **0.1431 ms**,
median **0.1415 ms**, p95 **0.1703 ms**, worst **0.1957 ms** per frame. LK consumed
0.1376 ms and geometry 0.0040 ms on average. Mean tracked/inlier counts were
76.018/75.976; all 500 estimates were valid. Detection ran once including warmup.
The ideal synthetic displacement error averaged 0.0000341 m. This excludes sensor
acquisition, camera capture and image generation; it is neither an UNO Q benchmark
nor an energy or real-flight accuracy measurement.

Validation includes 50 regression cases plus synthetic motion, analytical still,
analytical sequence and output CTests.
Sensor cases exercise yaw wrap, pure rotation, altitude scaling, moving outliers,
stale/future/missing/repeated readings, tilt gates, recovery and snapshot parsing.

## Analytical tensor hook

The same `extract(frame,timestamp,altitude,imu)` API now supports the analytical
profile from `analyticalFeatureSettings()`. Sensor validity rules are unchanged.
`pre_extract --camera --calibration camera.yaml --imu FILE --ultrasonic FILE`
reads the same snapshots. Offline sequence manifests instead supply explicit
per-frame sample timestamps; live snapshots are rejected for offline inputs to
avoid mixing clocks. Dense camera flow remains a 2-D displacement feature and
does not use range to manufacture depth. Range supports only the documented
constrained metric-pose/triangulation branch. See [pre-extraction](pre_extraction.md).
