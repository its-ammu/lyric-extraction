# Lyrics Transcriber (Flask + faster-whisper)

A minimal Flask service that accepts a vocal-music audio file and returns
the transcribed lyrics using
[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Designed to run on an AWS EC2 instance with an NVIDIA GPU (CUDA 12 + cuDNN 9).

## Features

- `POST /transcribe` — multipart upload returns JSON with full text + per-segment timestamps
- Optional word-level timestamps and VAD (silence) filtering
- Optional language hint or auto-detection
- Optional initial prompt (helpful for songs, artist names, jargon)
- Browser UI at `/` for ad-hoc testing
- `GET /healthz` health check
- Model loaded once at startup; batched inference pipeline for throughput

## EC2 setup (recommended: `g4dn.xlarge` / `g5.xlarge` or larger)

Use the **Deep Learning AMI (Ubuntu 22.04)** — it already ships with CUDA 12
and the NVIDIA driver. Otherwise use a vanilla Ubuntu 22.04 AMI and install
the NVIDIA driver yourself.

```bash
sudo apt-get update && sudo apt-get install -y python3-venv ffmpeg git

git clone <your-repo-url> lyrics && cd lyrics
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# cuBLAS and cuDNN come from the nvidia-* pip wheels in requirements.txt.
# Tell the dynamic linker where to find them:
export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

Verify the GPU is visible:

```bash
nvidia-smi
```

## Pre-download the model (recommended)

The model auto-downloads from Hugging Face on first use (~3 GB for
`large-v3`). To avoid making the first request wait several minutes — and
to bake the weights into your AMI / Docker image — pre-fetch them:

```bash
# Optional: pin where the weights live (otherwise ~/.cache/huggingface/hub)
export WHISPER_MODEL_DIR=/opt/whisper-models

python prefetch.py
```

This is GPU-free, so it works fine during AMI builds before CUDA libs are
configured. The runtime app will then load instantly from this cache.

## Run

Dev server (loads the model on startup via `app.py`'s `__main__` block):

```bash
python app.py
```

Production with gunicorn — uses `gunicorn_conf.py`, which warms the model in
the worker via a `post_worker_init` hook so the first request is fast:

```bash
gunicorn -c gunicorn_conf.py app:app
```

Use **one worker** — Whisper holds the GPU. Tune throughput with threads via
`GUNICORN_THREADS` (default 4). Do not pass `--preload`: a CUDA context
initialized in the master process cannot be shared with forked workers.

Open `http://<ec2-public-ip>:8000/` (make sure the security group lets port
8000 through, or put it behind nginx / an ALB).

## Configuration (env vars)

| Variable                | Default      | Notes                                                          |
| ----------------------- | ------------ | -------------------------------------------------------------- |
| `WHISPER_MODEL`         | `large-v3`   | e.g. `large-v3`, `medium`, `distil-large-v3`, `turbo`          |
| `WHISPER_DEVICE`        | `cuda`       | use `cpu` to test without a GPU                                |
| `WHISPER_COMPUTE_TYPE`  | `float16`    | try `int8_float16` to save VRAM, `int8` on CPU                 |
| `WHISPER_BEAM_SIZE`     | `5`          | larger = slower, sometimes better                              |
| `WHISPER_BATCH_SIZE`    | `8`          | used by `BatchedInferencePipeline`                             |
| `WHISPER_BATCHED`       | `1`          | set to `0` to use the plain (sequential) pipeline              |
| `WHISPER_MODEL_DIR`     | (HF cache)   | local path to cache CT2 weights                                |
| `MAX_UPLOAD_MB`         | `200`        | max upload size                                                |
| `PORT`                  | `8000`       | server port                                                    |
| `GUNICORN_WORKERS`      | `1`          | keep at 1 unless you have VRAM for multiple model copies       |
| `GUNICORN_THREADS`      | `4`          | concurrent requests per worker                                 |
| `GUNICORN_TIMEOUT`      | `600`        | seconds; long enough for big files                             |

## API

### `POST /transcribe`

multipart/form-data fields:

| Field             | Required | Description                                       |
| ----------------- | -------- | ------------------------------------------------- |
| `audio`           | yes      | audio file (mp3, wav, flac, m4a, ogg, opus, ...)  |
| `language`        | no       | language hint, e.g. `en`. Omit for auto-detect    |
| `word_timestamps` | no       | `true` / `false` (default `false`)                |
| `vad_filter`      | no       | `true` / `false` (default `true`)                 |
| `initial_prompt`  | no       | free-text prompt to bias decoding                 |
| `segments`        | no       | JSON of structure segments; transcribe only these ranges (overrides VAD) |
| `skip_labels`     | no       | comma-separated labels to drop from `segments` (default `silence`) |

#### Structure-segment mode (instead of VAD)

If you already have a music-structure analysis (e.g. from
[allin1](https://github.com/mir-aidata/all-in-one)), pass it as `segments` to
transcribe each labeled section directly and skip the parts you don't want
(`silence` by default). This is usually more reliable than VAD for music: no
sung phrases get dropped by a speech-tuned detector. Accepted shapes: a raw
list, an object with a `segments` key, or any nested JSON containing such a
list. Returned timestamps are absolute (offset back into the original audio),
and adjacent kept ranges are merged so Whisper keeps context.

```bash
curl -F "audio=@song.mp3" -F "language=en" \
     -F 'segments={"segments":[{"start":0.0,"end":127.08,"label":"verse"},{"start":216.0,"end":220.4,"label":"silence"}]}' \
     -F "skip_labels=silence" \
     http://<ec2-public-ip>:8000/transcribe
```

Example:

```bash
curl -F "audio=@song.mp3" \
     -F "language=en" \
     -F "word_timestamps=true" \
     http://<ec2-public-ip>:8000/transcribe
```

Response:

```json
{
  "language": "en",
  "language_probability": 0.99,
  "duration": 213.4,
  "text": "...",
  "segments": [
    { "id": 0, "start": 0.0, "end": 5.2, "text": "...", "words": [ ... ] }
  ]
}
```

## Notes for music transcription

- Whisper is trained primarily on speech, so dense instrumentation and
  layered harmonies can hurt accuracy. For best results, run the audio
  through a vocal-isolation tool (e.g. Demucs) first.
- `vad_filter=true` (default) skips silent/instrumental gaps.
- For best control, pass pre-computed `segments` (see Structure-segment mode)
  to transcribe specific sections and skip `silence` — avoids VAD dropping
  sung phrases.
- Increase `WHISPER_BEAM_SIZE` (e.g. `8`) for slightly better quality at a
  speed cost.
- Set `WHISPER_COMPUTE_TYPE=int8_float16` to roughly halve VRAM usage on
  small GPUs (e.g. T4 16 GB).
