"""EMA Lightning's decoder in MLX: 64-dim latents at 25 Hz in, 48 kHz audio out.

Same graph as ``decoder.py``. ``lengths`` holds everything past each row's real
length at zero after every layer, so a padded window matches the unpadded one.
"""
import mlx.core as mx
import mlx.nn as nn

from .mlx_weights import conv1d_from, conv_transpose1d_from


def _keep(x, length):
    if length is None:
        return x
    valid = mx.arange(x.shape[-1]) < length[:, None]
    return x * valid.astype(x.dtype)[:, None, :]


class ResBlock:
    def __init__(self, block):
        self.convs1 = [conv1d_from(conv) for conv in block.convs1]
        self.convs2 = [conv1d_from(conv) for conv in block.convs2]

    def __call__(self, x, length=None):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            y = _keep(nn.leaky_relu(c1(nn.leaky_relu(x, negative_slope=0.1)), negative_slope=0.1), length)
            x = _keep(x + c2(y), length)
        return x


class Decoder:
    def __init__(self, decoder):
        self.hop, self.nk = decoder.hop, decoder.nk
        self.pre = conv1d_from(decoder.pre)
        self.ups = [conv_transpose1d_from(up) for up in decoder.ups]
        self.blocks = [ResBlock(block) for block in decoder.blocks]
        self.post = conv1d_from(decoder.post)

    def __call__(self, z, lengths=None):
        """``z`` is ``[batch, latent, time]``, the same layout as the PyTorch decoder."""
        frames = z.shape[-1]
        length = lengths
        x = _keep(self.pre(_keep(z, length)), length)
        for i, up in enumerate(self.ups):
            x = up(nn.leaky_relu(x, negative_slope=0.1))
            if length is not None:
                length = (length - 1) * up.stride - 2 * up.padding + up.kernel
            x = _keep(x, length)
            acc = None
            for block in self.blocks[i * self.nk:(i + 1) * self.nk]:
                y = block(x, length)
                acc = y if acc is None else acc + y
            x = acc / self.nk
        # The last leaky ReLU uses PyTorch's default slope, 0.01. Every earlier one passes 0.1.
        return mx.tanh(self.post(nn.leaky_relu(x, negative_slope=0.01)))[:, 0, :frames * self.hop]
