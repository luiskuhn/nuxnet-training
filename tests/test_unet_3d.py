import torch
from torch import nn

from numorph_nuclei_segmentation.model import UNet3D, enable_mc_dropout
from numorph_nuclei_segmentation.model.unet_3d_models import ConvBlock


def test_unet_shape_logits_dropout_count_and_parameter_budget():
    model = UNet3D()
    model.eval()
    head_outputs = []
    handle = model.outc.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output))
    output = model(torch.randn(1, 1, 8, 16, 16))
    handle.remove()

    assert output.shape == (1, 2, 8, 16, 16)
    assert output is head_outputs[0]  # No probability activation follows the output head.
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout3d)]
    assert len(dropouts) == 16
    assert all(module.p == 0.10 for module in dropouts)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_500_000


def test_first_dropout_receives_learned_features_not_raw_input():
    model = UNet3D(dropout=0.10)
    seen_channels = []
    handle = model.inc.conv_block_1.dropout_1.register_forward_pre_hook(
        lambda _module, inputs: seen_channels.append(inputs[0].shape[1])
    )
    model.eval()
    model(torch.randn(1, 1, 4, 8, 8))
    handle.remove()

    assert seen_channels == [32]


def test_same_channel_residual_reduces_to_relu_identity():
    block = ConvBlock(8, 8, dropout=0.0).eval()
    nn.init.zeros_(block.conv_1.weight)
    nn.init.zeros_(block.conv_2.weight)
    value = torch.randn(2, 8, 4, 4, 4)

    assert torch.equal(block(value), torch.relu(value))


def test_channel_changing_residual_blocks_have_expected_shapes():
    for in_channels, out_channels in ((1, 32), (32, 64), (64, 128), (192, 64), (96, 32)):
        block = ConvBlock(in_channels, out_channels, dropout=0.0).eval()
        output = block(torch.randn(1, in_channels, 2, 4, 4))
        assert output.shape == (1, out_channels, 2, 4, 4)


def test_eval_is_deterministic_and_mc_dropout_preserves_batch_norm_eval():
    torch.manual_seed(4)
    model = UNet3D(dropout=0.5).eval()
    value = torch.randn(1, 1, 8, 16, 16)
    first = model(value)
    second = model(value)
    assert torch.equal(first, second)

    enable_mc_dropout(model)
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout3d)]
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm3d)]
    assert all(module.training for module in dropouts)
    assert all(not module.training for module in batch_norms)
    assert not torch.equal(model(value), model(value))
