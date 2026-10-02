# CNN–HTransformer

RGB learned convolutional features feed the CoHAtNet-inspired MBConv-Value HTransformer. This separates the attention design from IMF's analytical front end. Q/K are learned projections; V originates from spatial MBConv output.

Run `python run.py --train cnn_htransformer --profile research` or `python -m CNN_HTransformer.train --seed 7`.
