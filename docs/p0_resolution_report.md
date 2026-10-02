# Prior research-audit P0 resolution

This note tracks the P0 software issues from the pre-restructure audit. Historical audit files are preserved under [archive](archive/README.md); their paths describe the earlier repository layout.

| Issue | Resolution | Current evidence |
|---|---|---|
| Existing 20k dataset adapter | CLOSED | `data/preprocessing/prepare.py`, `data/adapter/dataset.py` read original episode layout; raw frames are not copied |
| Three-class landing semantics | CLOSED | `data/preprocessing/targets.py`; unsafe+caution hazard, safe-only landing; source masks remain unchanged |
| Missing waypoint target | REMOVED FROM PRIMARY MODEL | Shared outputs contain no waypoint head or waypoint loss |
| Ultrasonic vs vertical clearance | CLOSED IN DATA CONTRACT | 13-state includes sensor-axis ultrasonic range; vertical simulator clearance is separately named as geometry oracle |
| Magnetometer omitted | CLOSED IN REGISTRY | Three magnetic-field axes are included in the 13-state registry and validity mask |
| Camera calibration | CLOSED FOR CURRENT SYNTHETIC DATA | `data/calibration/blender_camera.json/yaml`; validation checks all episodes use the same resolution/FOV |
| Episode split leakage | CLOSED IN FROZEN MANIFEST | `data/manifests/manifest_lock.json`; validator rejects shared episode/trajectory/environment IDs |
| Baseline families | IMPLEMENTED | Four active model families share output heads/losses; external exact CoHAtNet is not claimed |
| Test-set protocol | CLOSED IN RUNNER DEFAULT | Primary/Full suites evaluate validation only; no final test evaluation is triggered automatically |
| Confidence-refined title wording | DOCUMENTED LIMITATION | GeometryConfidence is a support input, not a calibrated confidence-refinement mechanism |
| Neural-policy wording | REMOVED FROM PRIMARY CLAIM | Current network is perception/context output feeding a local map, not an actuator policy |

## Predicted-map training status

Training and validation now roll episodes forward using the model's prior-frame predictions, without opening labels during map construction. Candidate verdict targets are derived afterward from independently loaded simulator landing classes, and are valid only when pose, projection and current depth agree. This closes the static-zero-context software gap; it does not establish learned map benefit until trained and evaluated. The current code has not been trained on the full dataset by this restructuring pass.
