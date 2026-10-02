# Cross-platform development

The canonical entry point is Python: `python run.py`. It uses `pathlib` paths relative to the repository and passes subprocess arguments as arrays. No-argument execution reports status; it does not begin a long training suite.

## Python

Use Python 3.10 or newer and install `common/requirements.txt`. PyTorch should match the host accelerator. Device selection is CUDA, then Apple MPS, then CPU; set `training.device` in YAML to override. CPU-only smoke/unit checks do not require the dataset.

## C++

Use CMake 3.21+, a C++17 compiler and OpenCV development libraries. Configure with `cmake --preset default`, build with `cmake --build --preset default --parallel 2`, and run `ctest --preset default`. OpenCV 4 module names and OpenCV 5 renamed targets are handled in the top-level `CMakeLists.txt`. Webcam support is optional and requires `BUILD_MOTION_CAMERA=ON`.

`CMakePresets.json` includes the default host preset and a Visual Studio 2022 MSVC preset. The Python CI matrix covers Ubuntu, macOS and Windows. Physical Windows execution and Uno Q timing should only be claimed after those machines are actually tested; CI success is not hardware validation.

## Generated files and data safety

Manifests, caches, builds, figures, model weights and results live under `data/manifests/`, `data/cache/` or `artifacts/` and are ignored by Git. The 20,000-frame source is never copied by the adapter. The BlenderProc virtual environment under `data/.venv/` is also ignored. `python run.py --preextract --limit-episodes 1` is a derived-cache operation; it does not modify the rendered source frames.
