"""Copy a loaded PyTorch module's weights into MLX arrays.

Linear weights stay ``[out, in]``, which is also MLX's layout. Convolutions
do not: MLX convs take channels-last activations, so a PyTorch ``Conv1d``
weight ``[out, in, k]`` becomes ``[out, k, in]``, and a ``ConvTranspose1d``
weight ``[in, out, k]`` becomes ``[out, k, in]``. The copied arrays are the
whole bridge; inference never calls back into the PyTorch module.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch


def as_mx(value):
    """A torch tensor or MLX array as an MLX array. Indices become int32."""
    if isinstance(value, mx.array):
        return value
    array = value.detach()
    if array.dtype == torch.bool:
        return mx.array(array.cpu().numpy())
    if not array.dtype.is_floating_point:
        return mx.array(array.cpu().numpy().astype(np.int32))
    return mx.array(np.ascontiguousarray(array.cpu().float().numpy()))


def as_torch(value):
    """Evaluate an MLX array and return a float32 torch tensor that owns its memory."""
    mx.eval(value)
    return torch.from_numpy(np.ascontiguousarray(np.array(value))).clone()


def _mx(weight):
    return mx.array(np.ascontiguousarray(weight.detach().cpu().float().numpy()))


def linear_from(module):
    layer = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
    layer.weight = _mx(module.weight)
    if module.bias is not None:
        layer.bias = _mx(module.bias)
    return layer


def layernorm_from(module):
    affine = module.elementwise_affine
    layer = nn.LayerNorm(module.normalized_shape[0], eps=module.eps, affine=affine, bias=module.bias is not None)
    if affine:
        layer.weight = _mx(module.weight)
        if module.bias is not None:
            layer.bias = _mx(module.bias)
    return layer


class NCLConv:
    """PyTorch's ``[batch, channel, time]`` conv, implemented with MLX's channels-last conv."""

    def __init__(self, conv, stride, padding, kernel):
        self.conv = conv
        self.stride = stride
        self.padding = padding
        self.kernel = kernel

    def __call__(self, x):
        return mx.swapaxes(self.conv(mx.swapaxes(x, 1, 2)), 1, 2)


def _conv1d(module):
    conv = nn.Conv1d(
        module.in_channels, module.out_channels, module.kernel_size[0],
        stride=module.stride[0], padding=module.padding[0], dilation=module.dilation[0],
        groups=module.groups, bias=module.bias is not None)
    # [out, in, k] -> [out, k, in]
    conv.weight = _mx(module.weight.permute(0, 2, 1).contiguous())
    if module.bias is not None:
        conv.bias = _mx(module.bias)
    return conv


def _conv_transpose1d(module):
    if module.groups != 1:
        raise ValueError("grouped ConvTranspose1d is not part of this decoder")
    conv = nn.ConvTranspose1d(
        module.in_channels, module.out_channels, module.kernel_size[0],
        stride=module.stride[0], padding=module.padding[0], dilation=module.dilation[0],
        bias=module.bias is not None)
    # [in, out, k] -> [out, k, in]
    conv.weight = _mx(module.weight.permute(1, 2, 0).contiguous())
    if module.bias is not None:
        conv.bias = _mx(module.bias)
    return conv


def conv1d_from(module):
    return NCLConv(_conv1d(module), module.stride[0], module.padding[0], module.kernel_size[0])


def conv_transpose1d_from(module):
    return NCLConv(_conv_transpose1d(module), module.stride[0], module.padding[0], module.kernel_size[0])
