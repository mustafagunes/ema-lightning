"""MLX stages match the CPU PyTorch path on the tiny model, including a padded batch."""
import pytest

pytest.importorskip("mlx.core")

from conftest import same_speech  # noqa: E402

from ema_lightning import EMA  # noqa: E402
from ema_lightning.engine import Engine  # noqa: E402
from ema_lightning.mlx_backend import MLXRunner, convert  # noqa: E402


def _close(a, b, tol, name):
    err = float((a - b).abs().max())
    assert err <= tol, f"{name} max abs {err} > {tol}"


def test_mlx_stages_match_cpu(model, decoder):
    cpu = Engine(model, decoder, "cpu")
    mlx_model, mlx_decoder = convert(model, decoder)
    mlx = Engine(mlx_model, mlx_decoder, "cpu", MLXRunner(mlx_model, mlx_decoder))
    texts = [("merhaba dunya.", 3), ("ab", 9)]
    cpu_pieces = [cpu.piece(text, 0.0, seed) for text, seed in texts]
    mlx_pieces = [mlx.piece(text, 0.0, seed) for text, seed in texts]
    cpu.plan(cpu_pieces, 1.0)
    mlx.plan(mlx_pieces, 1.0)
    for left, right in zip(cpu_pieces, mlx_pieces, strict=True):
        _close(left.h, right.h, 1e-4, "text")
        _close(left.dur, right.dur, 1e-4, "duration")
        assert left.frames == right.frames
    cpu.think(cpu_pieces)
    mlx.think(mlx_pieces)
    for left, right in zip(cpu_pieces, mlx_pieces, strict=True):
        _close(left.latents, right.latents, 1e-3, "latents")
        span = (0, left.frames)
        audio = cpu.decode([(left, span)])[0]
        other = mlx.decode([(right, span)])[0]
        _close(audio, other, 1e-3, "audio")


def test_mlx_say_matches_cpu(model, decoder):
    from ema_lightning.frontend import Frontend

    frontend = Frontend(list(model.vocab))
    cpu = EMA._from_parts(model, decoder, frontend, "cpu")
    mlx = EMA._from_parts(model, decoder, frontend, "mlx")
    cpu._batch_size = mlx._batch_size = 2
    spoken = cpu.say("merhaba.", seed=1)
    other = mlx.say("merhaba.", seed=1)
    same_speech(spoken.audio, other.audio, db=40)
    assert spoken.sample_rate == other.sample_rate == 48000


def test_mlx_rejects_lightning(model, decoder):
    from ema_lightning.frontend import Frontend

    mlx = EMA._from_parts(model, decoder, Frontend(list(model.vocab)), "mlx")
    with pytest.warns(UserWarning, match="CUDA path"):
        assert mlx.lightning() is mlx
    assert mlx._engine.graphs is None
