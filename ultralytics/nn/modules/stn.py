# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial transformer modules.

A Spatial Transformer Network (STN) learns an explicit geometric warp of a feature map, which gives a
detector invariance to rotation, scale, and partial occlusion -- the failure modes YOLO exhibits on
cluttered, occluded, or low-contrast scenes. This block adapts the STN-YOLO contribution
(arXiv:2407.21652) into a single channel-preserving layer that can be dropped into any YOLO backbone or
neck like any other block.

Examples:
    >>> import torch
    >>> from ultralytics.nn.modules.stn import STN
    >>> m = STN(64)
    >>> x = torch.randn(1, 64, 16, 16)
    >>> assert m(x).shape == x.shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """Spatial Transformer Network block: learns a 2D affine warp of the input feature map.

    A lightweight localization head regresses the six affine parameters (theta) from pooled feature
    context, a sampling grid is generated, and the input is resampled with differentiable bilinear
    sampling. The localization head is initialized to the identity affine so the block is approximately a
    no-op until it learns a useful warp, which makes it safe to insert into a pretrained backbone.

    This adapts "Spatial Transformer Network YOLO Model for Agricultural Object Detection"
    (arXiv:2407.21652). The paper's core mechanism -- localization network -> affine grid -> differentiable
    sampler -- is preserved at full fidelity. The paper's agricultural benchmark and fixed backbone
    insertion points are out of scope here: placement is configured through the model YAML, and PyTorch's
    native ``affine_grid``/``grid_sample`` stand in for a hand-rolled sampler.

    Attributes:
        c1 (int): Input channels (== output channels; the warp is channel-preserving).
        loc (nn.Sequential): Localization network mapping the feature map to a (2, 3) affine matrix.

    Methods:
        forward: Apply the learned affine warp to the input feature map.

    Examples:
        >>> stn = STN(128)
        >>> x = torch.randn(1, 128, 32, 32)
        >>> assert stn(x).shape == x.shape
    """

    def __init__(self, c1: int, reduction: int = 8):
        """Initialize the STN block.

        Args:
            c1 (int): Input channels; output channels are equal (channel-preserving warp).
            reduction (int): Channel reduction ratio for the localization network.
        """
        super().__init__()
        self.c1 = c1
        hidden = max(c1 // reduction, 4)
        # Localization network: global feature context -> six affine parameters (a 2x3 matrix).
        self.loc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        # Bias to the identity affine and zero the weights so the warp starts as identity.
        self.loc[-1].weight.data.zero_()
        self.loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned affine transformation to the input feature map.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            (torch.Tensor): Affine-warped tensor with the same shape as the input.
        """
        theta = self.loc(x).view(x.shape[0], 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
