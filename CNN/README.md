# CNN

Small RGB-only convolutional perception baseline. The model receives the current 256×256 RGB image plus the common separately masked state/map interfaces and returns the shared hazard, landing, semantic, POI and verdict outputs. Shared training, losses and checkpoints are in `common/`.

Run `python run.py --train cnn --profile research` or `python -m CNN.train --seed 7`. No training starts by importing this package.
