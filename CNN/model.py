"""Small raw-RGB convolution-only baseline."""

from common.models.frontends import ConvolutionalBackbone, RGBPerceptionModel


class CNNModel(RGBPerceptionModel):
    family = "cnn"

    @staticmethod
    def build_backbone(widths, _overrides):
        return ConvolutionalBackbone(widths["embedding"], widths["stage3"], widths["stage4"])
