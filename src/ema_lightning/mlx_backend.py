"""Load the MLX acoustic model and decoder, and run one engine stage.

Playhead still batches torch tensors. This runner converts a stage's inputs,
runs it, and converts the outputs back, so the scheduler does not change.
Noise stays on the torch generator: the same seed is the same noise as CPU.
"""
import mlx.core as mx

from .mlx_decoder import Decoder
from .mlx_model import Acoustic
from .mlx_weights import as_mx, as_torch


def convert(model, decoder):
    """MLX copies of a loaded PyTorch acoustic model and decoder."""
    mlx_model, mlx_decoder = Acoustic(model), Decoder(decoder)
    mx.eval(mlx_model.out_proj.weight, mlx_decoder.post.conv.weight)
    return mlx_model, mlx_decoder


class MLXRunner:
    def __init__(self, model, decoder):
        self.model, self.decoder = model, decoder

    def run(self, stage, args):
        if stage == "text":
            hidden, duration = self.model.text_stage(as_mx(args["ids"]), as_mx(args["mask"]))
            return as_torch(hidden), as_torch(duration)
        if stage == "sound":
            latents = self.model.sound_stage(
                as_mx(args["h"]), as_mx(args["dur"]), as_mx(args["mask"]), as_mx(args["cw"]),
                as_mx(args["wstart"]), as_mx(args["fw"]), as_mx(args["fp"]), as_mx(args["fmask"]),
                as_mx(args["noise"]))
            return as_torch(latents)
        z = as_mx(args["z"].transpose(1, 2))
        full = bool((args["lengths"] == args["z"].shape[1]).all())
        return as_torch(self.decoder(z, None if full else as_mx(args["lengths"])))
