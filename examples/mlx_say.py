"""Speak one sentence with MLX and write a WAV.

    python examples/mlx_say.py "Merhaba, size nasıl yardımcı olabilirim?"
    python examples/mlx_say.py "Yarın saat 14:30'da görüşürüz." --out randevu.wav --seed 0 --speed 0.83

The first run measures a batch size and caches it. The printed time is only
the sentence after that.
"""
import argparse
import time

from ema_lightning import EMA


def main():
    parser = argparse.ArgumentParser(description="Speak one sentence with EMA on MLX.")
    parser.add_argument("text", help="the sentence to speak")
    parser.add_argument("--out", default="mlx.wav", help="WAV path (default: mlx.wav)")
    parser.add_argument("--seed", type=int, default=0, help="same seed, same audio (default: 0)")
    parser.add_argument("--speed", type=float, default=1.0, help="pace from 0.25 to 4 (default: 1)")
    args = parser.parse_args()

    started = time.perf_counter()
    tts = EMA(device="mlx")
    tts.best_batch_size()
    ready = time.perf_counter() - started

    started = time.perf_counter()
    speech = tts.say(args.text, speed=args.speed, seed=args.seed, path=args.out)
    took = time.perf_counter() - started
    print(f"ready in {ready:.1f}s, then {speech.duration:.2f}s of audio in {took * 1000:.0f} ms "
          f"({speech.duration / took:.0f}x real time)")
    print(f"wrote {args.out}  seed {speech.seed}  speed {args.speed}")


if __name__ == "__main__":
    main()
