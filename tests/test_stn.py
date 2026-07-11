# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy

import torch

from ultralytics.nn.modules import STN
from ultralytics.nn.tasks import parse_model


def test_stn_block_preserves_shape_and_identity_init() -> None:
    """STN keeps the feature-map shape and its localization head starts as the identity affine."""
    stn = STN(64)
    # The final affine regressor is initialized to the identity transform so the block is a no-op at start.
    assert torch.allclose(stn.loc[-1].bias.data, torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    assert torch.allclose(stn.loc[-1].weight.data, torch.zeros_like(stn.loc[-1].weight.data))

    x = torch.randn(2, 64, 16, 16)
    y = stn(x)
    assert y.shape == x.shape  # the warp is channel- and spatial-size-preserving
    assert torch.isfinite(y).all()

    # The warp is parameterized, so it must be differentiable end-to-end.
    y.sum().backward()
    assert stn.loc[-1].weight.grad is not None


def test_stn_is_buildable_from_yaml_spec() -> None:
    """STN is registered in parse_model and builds like any other YOLO block from a model spec."""
    cfg = {
        "nc": 1,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "backbone": [
            [-1, 1, "Conv", [8, 3, 2]],  # 0: downsample 8x8 -> 4x4, 3 -> 8 channels
            [-1, 1, "STN", []],  # 1: learned affine warp, channel-preserving
        ],
        "head": [],
    }
    model, _ = parse_model(copy.deepcopy(cfg), ch=3)
    assert isinstance(model[1], STN)  # wiring resolves the "STN" name to the block class
    out = model(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 8, 4, 4)  # channels preserved, spatial size unchanged by STN
