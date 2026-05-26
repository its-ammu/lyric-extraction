"""Pre-download the faster-whisper model so it's already on disk at startup.

Run this once during your AMI / Docker image build, or right after deploy,
so the first /transcribe request doesn't trigger a multi-gigabyte download
from Hugging Face.

Honors the same env vars as the Flask app:
    WHISPER_MODEL       (default: large-v3)
    WHISPER_MODEL_DIR   (default: HF cache, ~/.cache/huggingface/hub)

This script intentionally uses device="cpu" so it works even on machines
without a GPU / CUDA libraries (e.g. AMI build environments). The downloaded
weights are device-independent — the runtime app will still use CUDA.
"""
from __future__ import annotations

import os
import sys
import time

from faster_whisper import WhisperModel


def main() -> int:
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    download_root = os.environ.get("WHISPER_MODEL_DIR")  # None => HF default cache

    print(f"[prefetch] model={model_size} download_root={download_root or '<HF default cache>'}")
    t0 = time.time()
    WhisperModel(model_size, device="cpu", compute_type="int8", download_root=download_root)
    print(f"[prefetch] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
