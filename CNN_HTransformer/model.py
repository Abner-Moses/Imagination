"""RGB learned front end followed by MBConv-value HTransformer stages."""

from common.models.frontends import RGBPerceptionModel
from common.models.htransformer import HTransformerBackbone


class CNNHTransformer(RGBPerceptionModel):
    family = "cnn_htransformer"

    @staticmethod
    def build_backbone(widths, _overrides):
        config = {
            "embedding": widths["embedding"],
            "stage3_channels": widths["stage3"],
            "stage4_channels": widths["stage4"],
            "stage3_heads": widths["heads3"],
            "stage4_heads": widths["heads4"],
            "stage3_depth": widths["depth3"],
            "stage4_depth": widths["depth4"],
            "mbconv_expansion": 2.0,
            "stochastic_depth": 0.05,
        }
        return HTransformerBackbone(config, widths["embedding"])
