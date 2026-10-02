"""Shared RGB input path and the convolution-only backbone."""

from __future__ import annotations

from torch import nn

from .blocks import MBConv
from .heads import PerceptionHeads
from .profiles import profile_config
from common.registry import POI_CLASSES, SEMANTIC_CLASSES, STATE_DIM


class RGBFrontEnd(nn.Module):
    """256x256 RGB to a 32x32 learned local representation."""

    def __init__(self, embedding: int, stem: int, input_channels: int = 3, expansion: float = 2.0):
        super().__init__()
        middle = max(stem, embedding // 2)
        self.input_channels = input_channels
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, stem, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem),
            nn.SiLU(),
            MBConv(stem, middle, expansion=expansion, stride=2),
            MBConv(middle, embedding, expansion=expansion, stride=2),
        )

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != self.input_channels or rgb.shape[-2:] != (256, 256):
            raise ValueError(f"RGB input must be Bx{self.input_channels}x256x256")
        return self.layers(rgb)


class ConvolutionalBackbone(nn.Module):
    """No attention: local depthwise convolutions plus bounded global receptive field."""

    def __init__(self, embedding: int, stage3: int, stage4: int, depth3: int = 2, depth4: int = 2):
        super().__init__()
        self.stage3 = nn.Sequential(
            MBConv(embedding, stage3, stride=2),
            *[MBConv(stage3, stage3) for _ in range(depth3 - 1)],
        )
        self.stage4 = nn.Sequential(
            MBConv(stage3, stage4, stride=2), *[MBConv(stage4, stage4) for _ in range(depth4 - 1)]
        )
        self.output_channels = stage4

    def forward(self, value):
        return self.stage4(self.stage3(value))


class RGBPerceptionModel(nn.Module):
    """Common RGB input, conditioning, outputs, and configuration contract."""

    family = ""

    def __init__(
        self,
        profile: str = "research",
        use_state: bool = True,
        overrides: dict | None = None,
        poi_classes: int = len(POI_CLASSES),
    ):
        super().__init__()
        widths = profile_config(profile, overrides)
        self.profile = profile
        self.frontend = RGBFrontEnd(widths["embedding"], widths["stem"])
        self.backbone = self.build_backbone(widths, overrides or {})
        self.heads = PerceptionHeads(
            widths["stage4"],
            widths["decoder"],
            STATE_DIM,
            poi_classes,
            len(SEMANTIC_CLASSES),
            use_state,
        )

    @staticmethod
    def build_backbone(widths: dict, overrides: dict) -> nn.Module:
        raise NotImplementedError

    def forward(
        self,
        rgb,
        state_values=None,
        state_validity=None,
        relational=None,
        relational_validity=None,
        candidates=None,
        candidate_validity=None,
        candidate_feature_validity=None,
    ):
        features = self.backbone(self.frontend(rgb))
        return self.heads(
            features,
            state_values,
            state_validity,
            relational,
            relational_validity,
            candidates,
            candidate_validity,
            candidate_feature_validity,
        )

    @classmethod
    def from_config(cls, config: dict):
        model = config.get("model", {})
        return cls(
            profile=model.get("profile", "research"),
            use_state=model.get("state_conditioning", True),
            overrides=model,
            poi_classes=len(config.get("poi", {}).get("classes", POI_CLASSES)),
        )
