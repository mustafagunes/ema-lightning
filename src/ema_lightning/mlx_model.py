"""EMA Lightning's acoustic model in MLX.

Same four-step latent generator as ``model.py``: letters to durations, durations
to a frame timeline, then the aligner and the distilled steps. Weights are copied
from a loaded PyTorch ``Acoustic``; this module only runs the forward pass.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from .mlx_weights import _mx, conv1d_from, layernorm_from, linear_from


def rope(pos, dim):
    pos = pos.astype(mx.float32)
    half = dim // 2
    inv = 1.0 / (10000.0 ** (mx.arange(half, dtype=mx.float32) / half))
    ang = pos[..., None] * inv
    ang = mx.concatenate([ang, ang], axis=-1)
    return mx.cos(ang), mx.sin(ang)


def apply_rope(x, cos, sin):
    a, b = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-b, a], axis=-1) * sin


def _heads(t, heads):
    batch, time, dim = t.shape
    return mx.swapaxes(mx.reshape(t, (batch, time, heads, dim // heads)), 1, 2)


def _merge(t):
    batch, heads, time, dh = t.shape
    return mx.reshape(mx.swapaxes(t, 1, 2), (batch, time, heads * dh))


def _scatter_add(base, index, src):
    rows = mx.broadcast_to(mx.arange(index.shape[0])[:, None], index.shape)
    return base.at[rows, index].add(src)


class SwiGLU:
    def __init__(self, block):
        self.w1, self.w3, self.w2 = linear_from(block.w1), linear_from(block.w3), linear_from(block.w2)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention:
    def __init__(self, block):
        self.h, self.dh = block.h, block.dh
        self.qkv, self.proj = linear_from(block.qkv), linear_from(block.proj)

    def __call__(self, x, cos, sin, mask):
        q, k, v = (_heads(part, self.h) for part in mx.split(self.qkv(x), 3, axis=-1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        logits = (q @ mx.swapaxes(k, -2, -1)) / math.sqrt(self.dh)
        logits = mx.where(mask[:, None, None, :], logits, mx.array(float("-inf"), dtype=logits.dtype))
        out = mx.softmax(logits, axis=-1) @ v
        return self.proj(_merge(out))


class ConvNeXtBlock:
    def __init__(self, block):
        self.dw = conv1d_from(block.dw).conv  # channels-last, same layout as x
        self.norm = layernorm_from(block.norm)
        self.pw1, self.pw2 = linear_from(block.pw1), linear_from(block.pw2)
        self.gamma = _mx(block.gamma)

    def __call__(self, x, mask):
        # Padding positions are multiplied by zero before the depthwise conv, so they read as zeros.
        y = self.norm(self.dw(x * mask.astype(mx.float32)[..., None]))
        return x + self.gamma * self.pw2(nn.gelu(self.pw1(y)))


class TextEncoder:
    def __init__(self, encoder):
        self.dh = encoder.dh
        self.emb = nn.Embedding(encoder.emb.num_embeddings, encoder.emb.embedding_dim)
        self.emb.weight = _mx(encoder.emb.weight)
        self.conv = [ConvNeXtBlock(block) for block in encoder.conv]
        self.attn = [TextBlock(block) for block in encoder.attn]
        self.norm = layernorm_from(encoder.norm)

    def __call__(self, ids, mask):
        x = self.emb(ids)
        for block in self.conv:
            x = block(x, mask)
        cos, sin = rope(mx.arange(ids.shape[1]), self.dh)
        for block in self.attn:
            x = block(x, cos, sin, mask)
        return self.norm(x)


class TextBlock:
    def __init__(self, block):
        self.n1, self.n2 = layernorm_from(block.n1), layernorm_from(block.n2)
        self.attn, self.ff = Attention(block.attn), SwiGLU(block.ff)

    def __call__(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ff(self.n2(x))


def _masked_group_norm(x, norm, mask):
    """GroupNorm over real positions only. ``x`` and ``mask`` are ``[batch, channel, time]`` / ``[batch, 1, time]``."""
    count = mx.sum(mask, axis=(1, 2), keepdims=True) * x.shape[1]
    mean = mx.sum(x * mask, axis=(1, 2), keepdims=True) / count
    var = mx.sum(((x - mean) * mask) ** 2, axis=(1, 2), keepdims=True) / count
    return (x - mean) / mx.sqrt(var + norm.eps) * norm.weight[None, :, None] + norm.bias[None, :, None]


class _GroupNorm:
    def __init__(self, norm):
        self.eps = norm.eps
        self.weight, self.bias = _mx(norm.weight), _mx(norm.bias)


class Duration:
    def __init__(self, duration):
        self.conv0 = conv1d_from(duration.net[0])
        self.norm0 = _GroupNorm(duration.net[2])
        self.conv1 = conv1d_from(duration.net[4])
        self.norm1 = _GroupNorm(duration.net[6])
        self.out = linear_from(duration.out)

    def __call__(self, h, mask):
        m = mask.astype(mx.float32)[:, None, :]
        x = mx.swapaxes(h, 1, 2) * m
        x = _masked_group_norm(nn.silu(self.conv0(x)), self.norm0, m) * m
        x = _masked_group_norm(nn.silu(self.conv1(x)), self.norm1, m)
        log_d = mx.squeeze(self.out(mx.swapaxes(x, 1, 2)), axis=-1)
        log_d = mx.where(mask, log_d, mx.array(0.0, dtype=log_d.dtype))
        return mx.maximum(mx.expm1(mx.minimum(log_d, mx.array(6.0))), mx.array(1e-3)) * mask.astype(mx.float32)


class Aligner:
    def __init__(self, aligner):
        self.h, self.dh = aligner.h, aligner.dh
        self.pos_scale = float(aligner.pos_scale)
        self.lookback, self.lookahead = int(aligner.lookback), int(aligner.lookahead)
        self.letter_pos = bool(aligner.letter_pos)
        self.q, self.k, self.v, self.o = (linear_from(layer) for layer in (aligner.q, aligner.k, aligner.v, aligner.o))
        self.frame_q = _mx(aligner.frame_q)
        self.log_sigma, self.bias_w, self.log_temp = _mx(aligner.log_sigma), _mx(aligner.bias_w), _mx(aligner.log_temp)

    def __call__(self, h, cw, cp, fw, fp, mask, n_words):
        batch, letters, dim = h.shape
        frames = fw.shape[1]
        if self.letter_pos:
            c, f = mx.maximum(cw, 0), mx.maximum(fw, 0)
            valid = (cw >= 0).astype(mx.float32)
            wlen = mx.maximum(_scatter_add(mx.zeros((batch, n_words)), c, valid), mx.array(1.0))
            woff = mx.cumsum(wlen, axis=-1) - wlen
            cg = mx.take_along_axis(woff, c, axis=1) + cp * mx.take_along_axis(wlen, c, axis=1)
            fg = mx.take_along_axis(woff, f, axis=1) + fp * mx.take_along_axis(wlen, f, axis=1)
        else:
            cg = mx.maximum(cw, 0).astype(mx.float32) + cp
            fg = fw.astype(mx.float32) + fp
        q = _heads(self.q(mx.broadcast_to(self.frame_q, (batch, frames, dim))), self.h)
        k, v = _heads(self.k(h), self.h), _heads(self.v(h), self.h)
        q_cos, q_sin = rope(fg * self.pos_scale, self.dh)
        k_cos, k_sin = rope(cg * self.pos_scale, self.dh)
        q, k = apply_rope(q, q_cos[:, None], q_sin[:, None]), apply_rope(k, k_cos[:, None], k_sin[:, None])
        logits = (q @ mx.swapaxes(k, -2, -1)) / math.sqrt(self.dh) * mx.exp(self.log_temp)
        sig2 = mx.reshape(mx.exp(self.log_sigma) ** 2, (1, self.h, 1, 1))
        dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
        logits = logits - mx.reshape(self.bias_w, (1, self.h, 1, 1)) * dist2 / (2 * sig2)
        rel = cw[:, None, :] - fw[:, :, None]
        allow = (rel >= -self.lookback) & (rel <= self.lookahead) & mask[:, None, :] & (cw[:, None, :] >= 0)
        weights = mx.softmax(mx.where(allow[:, None], logits, mx.array(-1e4, dtype=logits.dtype)), axis=-1)
        weights = weights * allow.astype(mx.float32)[:, None]
        return self.o(_merge(weights @ v))


class _SiluLinear:
    def __init__(self, sequential):
        self.linear = linear_from(sequential[1])

    def __call__(self, x):
        return self.linear(nn.silu(x))


class TimestepEmbed:
    def __init__(self, embed):
        self.freq = embed.freq
        self.lin1, self.lin2 = linear_from(embed.mlp[0]), linear_from(embed.mlp[2])

    def __call__(self, t):
        half = self.freq // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        angles = t[:, None] * 1000.0 * freqs[None]
        return self.lin2(nn.silu(self.lin1(mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1))))


class DiTBlock:
    def __init__(self, block):
        self.n1, self.n2 = layernorm_from(block.n1), layernorm_from(block.n2)
        self.attn, self.ff = Attention(block.attn), SwiGLU(block.ff)
        self.shared = hasattr(block, "ada_offset")
        if self.shared:
            self.ada_offset = _mx(block.ada_offset)
        else:
            self.ada = _SiluLinear(block.ada)

    def __call__(self, x, c, cos, sin, mask):
        parts = [c[:, i] + self.ada_offset[i] for i in range(6)] if self.shared else mx.split(self.ada(c), 6, axis=-1)
        sa, ga, aa, sf, gf, af = (part[:, None, :] for part in parts)
        x = x + aa * self.attn(self.n1(x) * (1 + ga) + sa, cos, sin, mask)
        return x + af * self.ff(self.n2(x) * (1 + gf) + sf)


class Acoustic:
    def __init__(self, model):
        self.d, self.dh = model.d, model.dh
        self.latent_dim, self.shared = model.latent_dim, model.shared
        self.vocab, self.stoi = list(model.vocab), dict(model.stoi)
        self.times = [float(t) for t in model.times]
        self.text = TextEncoder(model.text)
        self.chardur = Duration(model.chardur)
        self.aligner = Aligner(model.aligner)
        self.in_proj = linear_from(model.in_proj)
        self.t_embed = TimestepEmbed(model.t_embed)
        self.ada_shared = _SiluLinear(model.ada_shared) if model.shared else None
        self.blocks = [DiTBlock(block) for block in model.blocks]
        self.norm_out = layernorm_from(model.norm_out)
        self.ada_out = _SiluLinear(model.ada_out)
        self.out_proj = linear_from(model.out_proj)

    def backbone(self, x, cond, t, mask, cos, sin):
        c = self.t_embed(t)
        bc = mx.reshape(self.ada_shared(c), (c.shape[0], 6, self.d)) if self.shared else c
        x = self.in_proj(x) + cond
        for block in self.blocks:
            x = block(x, bc, cos, sin, mask)
        s, g = mx.split(self.ada_out(c), 2, axis=-1)
        return self.out_proj(self.norm_out(x) * (1 + g[:, None, :]) + s[:, None, :])

    def text_stage(self, ids, mask):
        h = self.text(ids, mask)
        return h, self.chardur(h, mask)

    def sound_stage(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        spoken = mask.astype(mx.float32)
        c = mx.maximum(dur, mx.array(1e-4)) * spoken
        done = mx.cumsum(c, axis=-1)
        before = done - c
        word = mx.maximum(cw, 0)
        total = mx.take_along_axis(_scatter_add(mx.zeros_like(c), word, c), word, axis=1)
        start = mx.take_along_axis(before, wstart, axis=1)
        cp = mx.clip((done - start - 0.5 * c) / mx.maximum(total, mx.array(1e-8)), 0.0, 1.0) * spoken
        cond = self.aligner(h, cw, cp, fw, fp, mask, cw.shape[1])
        cos, sin = rope(mx.arange(fw.shape[1]), self.dh)
        x = noise[:, 0]
        x1 = x
        for k, t in enumerate(self.times):
            step = mx.full((x.shape[0],), t)
            x1 = x + (1.0 - t) * self.backbone(x, cond, step, fmask, cos, sin)
            if k + 1 < len(self.times):
                x = (1.0 - self.times[k + 1]) * noise[:, k + 1] + self.times[k + 1] * x1
        return x1
