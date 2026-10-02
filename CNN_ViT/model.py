"""RGB convolutional front end followed by conventional ViT attention."""

from common.models.frontends import RGBPerceptionModel
from common.models.vit import ViTBackbone


class CNNViTModel(RGBPerceptionModel):
    family = "cnn_vit"

    @staticmethod
    def build_backbone(widths, _overrides):
        return ViTBackbone(
            widths["embedding"],
            widths["stage3"],
            widths["stage4"],
            widths["heads3"],
            widths["heads4"],
            widths["depth3"],
            widths["depth4"],
        )
