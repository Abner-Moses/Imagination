"""Canonical cross-platform entry point for the frozen primary sequence."""

try:
    from common.training.sequence import cli
except ModuleNotFoundError as error:
    if error.name in {"torch", "numpy", "cv2", "PIL", "yaml"}:
        raise SystemExit(
            f"Missing training dependency: {error.name}. Create the dedicated training "
            "environment and install the root requirements:\n\n"
            "  python3 -m venv .venv-training\n"
            "  source .venv-training/bin/activate\n"
            "  python -m pip install -r requirements.txt\n\n"
            "Do not use data/.venv; it is reserved for dataset generation."
        ) from error
    raise


if __name__ == "__main__":
    raise SystemExit(cli())
