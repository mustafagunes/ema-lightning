"""Check MLX audio against CPU, then time CPU, MLX and MPS on one sentence.

    python benches/mlx_bench.py

Writes cpu.wav and mlx.wav next to each other and exits non-zero if they
disagree by more than the project's 40 dB margin.
"""
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ema_lightning import EMA
from ema_lightning.api import PROBE

OUT = Path("/tmp/ema-mlx")
SEED = 0


def quieter(a, b):
    """How many dB the difference sits below the speech. None when the lengths differ."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return None
    diff = float(np.sum((a - b) ** 2))
    if diff == 0.0:
        return float("inf")
    speech = float(np.sum(a ** 2))
    return 10 * np.log10(speech / diff) if speech > 0 else float("-inf")


def machine():
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        brand = platform.processor()
    return brand or platform.machine()


def timed(tts):
    """Median say() and first-chunk time, after the batch-size search has already run."""
    says, firsts, speech = [], [], None
    for _ in range(5):
        start = time.perf_counter()
        speech = tts.say(PROBE, seed=SEED)
        says.append(time.perf_counter() - start)
    for _ in range(5):
        start = time.perf_counter()
        next(iter(tts.stream(PROBE, seed=SEED)))
        firsts.append(time.perf_counter() - start)
    return speech, statistics.median(says), statistics.median(firsts)


def report(name, speech, say_s, first_s, load_s):
    factor = speech.duration / say_s
    print(f"{name:4}  load {load_s:6.2f}s   say {say_s * 1000:8.1f} ms   "
          f"first chunk {first_s * 1000:7.1f} ms   {speech.duration:.2f}s audio   {factor:.1f}x real time")


def main():
    print(machine())
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading CPU...")
    start = time.perf_counter()
    cpu = EMA(device="cpu")
    cpu_load = time.perf_counter() - start
    reference = cpu.say(PROBE, seed=SEED, path=OUT / "cpu.wav")

    print("loading MLX...")
    start = time.perf_counter()
    mlx = EMA(device="mlx")
    mlx_load = time.perf_counter() - start
    other = mlx.say(PROBE, seed=SEED, path=OUT / "mlx.wav")
    db = quieter(reference.audio, other.audio)
    if db is None:
        print(f"length mismatch: CPU {reference.audio.shape} MLX {other.audio.shape}")
        return 1
    print(f"CPU vs MLX: {db:.1f} dB quieter than the speech (need at least 40)")
    if db < 40:
        return 1

    print("timing...")
    _, cpu_say, cpu_first = timed(cpu)
    _, mlx_say, mlx_first = timed(mlx)
    report("cpu", reference, cpu_say, cpu_first, cpu_load)
    report("mlx", other, mlx_say, mlx_first, mlx_load)
    del cpu, mlx

    if torch.backends.mps.is_available():
        print("loading MPS...")
        start = time.perf_counter()
        mps = EMA(device="mps")
        mps_load = time.perf_counter() - start
        speech, mps_say, mps_first = timed(mps)
        report("mps", speech, mps_say, mps_first, mps_load)
    print(f"wavs: {OUT / 'cpu.wav'}  {OUT / 'mlx.wav'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
