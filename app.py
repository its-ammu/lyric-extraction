"""Flask app that transcribes vocal music audio into lyrics using faster-whisper.

Designed to run on an EC2 GPU instance (CUDA 12 + cuDNN 9).
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from faster_whisper import BatchedInferencePipeline, WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lyrics-api")


# ---------------------------------------------------------------------------
# Configuration (via environment variables)
# ---------------------------------------------------------------------------
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
USE_BATCHED = os.environ.get("WHISPER_BATCHED", "1") == "1"
DOWNLOAD_ROOT = os.environ.get("WHISPER_MODEL_DIR")  # optional cache dir

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
ALLOWED_EXTENSIONS = {
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "webm", "mp4", "wma",
}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "lyrics_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model loading (singleton, thread-safe)
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_model: Optional[WhisperModel] = None
_batched_model: Optional[BatchedInferencePipeline] = None
# faster-whisper's transcribe is not guaranteed thread-safe; serialize requests.
_transcribe_lock = threading.Lock()


def get_model() -> tuple[WhisperModel, Optional[BatchedInferencePipeline]]:
    global _model, _batched_model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info(
                    "Loading faster-whisper model=%s device=%s compute_type=%s",
                    MODEL_SIZE, DEVICE, COMPUTE_TYPE,
                )
                _model = WhisperModel(
                    MODEL_SIZE,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=DOWNLOAD_ROOT,
                )
                if USE_BATCHED:
                    _batched_model = BatchedInferencePipeline(model=_model)
                logger.info("Model loaded.")
    return _model, _batched_model


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", model=MODEL_SIZE, device=DEVICE)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file part in request."}), 400

    audio = request.files["audio"]
    if not audio.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed(audio.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        }), 400

    language = request.form.get("language") or None  # auto-detect when None
    word_timestamps = request.form.get("word_timestamps", "false").lower() == "true"
    vad_filter = request.form.get("vad_filter", "true").lower() == "true"
    initial_prompt = request.form.get("initial_prompt") or None

    safe_name = secure_filename(audio.filename)
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    audio.save(tmp_path)

    try:
        model, batched = get_model()
        logger.info(
            "Transcribing %s (lang=%s, word_ts=%s, vad=%s)",
            tmp_path.name, language, word_timestamps, vad_filter,
        )

        with _transcribe_lock:
            if batched is not None:
                segments_iter, info = batched.transcribe(
                    str(tmp_path),
                    batch_size=BATCH_SIZE,
                    beam_size=BEAM_SIZE,
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                    initial_prompt=initial_prompt,
                )
            else:
                segments_iter, info = model.transcribe(
                    str(tmp_path),
                    beam_size=BEAM_SIZE,
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                    initial_prompt=initial_prompt,
                )

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                entry = {
                    "id": seg.id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }
                if word_timestamps and seg.words:
                    entry["words"] = [
                        {
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "word": w.word,
                            "probability": round(w.probability, 4),
                        }
                        for w in seg.words
                    ]
                segments.append(entry)
                full_text_parts.append(seg.text)

        return jsonify({
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "text": "".join(full_text_parts).strip(),
            "segments": segments,
        })

    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not remove temp file %s", tmp_path)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File too large. Limit is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    # Pre-load model so the first request isn't slow.
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
