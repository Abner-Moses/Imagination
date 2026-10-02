# Reproducible figures

Run `python run.py --figures` to choose the first available manifested sample with analytical cache and write under `artifacts/paper_figures/`. To select a frame directly, use `python -m tools.generate_figures --sample episode_000001:000000`. The tool reads one saved RGB image and its cached tensor; it never reruns the renderer or changes the dataset.

The pipeline panel and each extracted plane use fixed channel registry names, fixed scales and the saved validity mask. Flow is shown only if `OpticalFlowU/V` validity indicates a real temporal estimate. Sparse correspondences and 3-D points are shown only when the cache exporter includes valid observations. No still-image difference is labeled as motion.

With a map export, pass `--map-json <predicted-map.json>` to create a local-map diagnostic. Valid map covariance is shown as a 95% horizontal uncertainty ellipse. Oracle and predicted maps must be labeled separately.

For a trained HTransformer streaming diagnostic, run `python -m tools.stream_map --attention-debug-output <file.npz>` and then pass `--attention-npz <file.npz>` to `python -m tools.generate_figures`. The optional capture records the final frame's sparse neighbor graph, attention weights, metric positions and map-token validity. Capture is disabled during ordinary inference because it copies diagnostic tensors to CPU. `--bayesian-demo` produces a fixed synthetic illustration of the map's evidence update; it is explicitly labeled as synthetic and is not a model or dataset result.

The generator creates IMF preprocessing panels, map covariance and attention diagnostics. Benchmark-chart generation from result tables is not implemented yet. Raster panels are written as PNG, and publication panels are also written as SVG and PDF. Figure generation requires Pillow and OpenCV already used by the data pipeline.
