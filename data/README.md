# Data

The repository expects the existing 20,000-frame synthetic source at `data/dataset/episode_*`. This directory is intentionally ignored by Git and must not be regenerated to run the current adapter.

```text
data/
  dataset/          existing RGB, depth, semantic/landing masks, sensor CSVs and poses
  generator/        Blender episode generator
  adapter/          PyTorch dataset reader for the actual episode format
  preprocessing/    frozen manifests, labels and C++ IMF cache adapter
  calibration/      canonical Blender pinhole calibration
  manifests/        grouped train/validation/test JSONL plus hash lock
  cache/             derived IMF NPZ frames; safe to rebuild
  map/               data-side map package boundary
  configs/           extraction, calibration and semantic rules
  assets/            lightweight or licensed generator assets
```

Start with the root [README](../README.md). `python run.py` reports the current data/cache status. `python run.py --validate` verifies the dataset report, frozen manifests, Python tests and C++ build. `python run.py --preextract --limit-episodes 1` generates derived IMF cache for one existing episode; it does not edit source frames.

The raw mask IDs and sensor meanings are documented in [the experiment protocol](../docs/experiment_protocol.md). `ultrasonic_range_m` is the measurement along the sensor axis. `ground_clearance_m` is separate simulator vertical geometry used as an explicitly marked synthetic scale source. The former is never relabeled as altitude.

The previous detailed generator setup notes are retained in `docs/archive/dataset_generator_legacy.md`. Dataset rendering was not rerun as part of the model/repository restructuring.
