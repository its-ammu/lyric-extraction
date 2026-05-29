"""Flask app that transcribes vocal music audio into lyrics using faster-whisper.

Designed to run on an EC2 GPU instance (CUDA 12 + cuDNN 9).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

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

# faster-whisper resamples audio to this rate; we slice raw arrays at it too.
SAMPLE_RATE = 16000
# Structure labels treated as non-vocal and skipped when segment ranges given.
DEFAULT_SKIP_LABELS = {"silence"}

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
# Structure-segment helpers (transcribe explicit ranges instead of using VAD)
# ---------------------------------------------------------------------------
def _extract_segments(data) -> Optional[list]:
    """Find a list of {start, end, label} dicts anywhere in parsed JSON.

    Accepts the raw list, an object with a "segments" key (e.g. allin1
    output), or any nested structure containing such a list.
    """
    def _looks_like_segments(value) -> bool:
        return (
            isinstance(value, list)
            and len(value) > 0
            and all(
                isinstance(x, dict) and "start" in x and "end" in x for x in value
            )
        )

    if _looks_like_segments(data):
        return data
    if isinstance(data, dict):
        if _looks_like_segments(data.get("segments")):
            return data["segments"]
        for value in data.values():
            found = _extract_segments(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _extract_segments(item)
            if found:
                return found
    return None


def _build_ranges(segments, skip_labels) -> list[tuple[float, float]]:
    """Turn structure segments into merged (start, end) ranges to transcribe.

    Segments whose label is in `skip_labels` are dropped. Touching/overlapping
    ranges are merged so Whisper sees continuous vocals with proper context.
    """
    keep: list[tuple[float, float]] = []
    for seg in segments:
        label = str(seg.get("label", "")).strip().lower()
        if label in skip_labels:
            continue
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            keep.append((start, end))

    keep.sort()
    merged: list[list[float]] = []
    for start, end in keep:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _seg_to_entry(seg, seg_id: int, offset: float, word_timestamps: bool) -> dict:
    """Serialize a faster-whisper segment, shifting times by `offset` seconds."""
    entry = {
        "id": seg_id,
        "start": round(seg.start + offset, 3),
        "end": round(seg.end + offset, 3),
        "text": seg.text.strip(),
    }
    if word_timestamps and seg.words:
        entry["words"] = [
            {
                "start": round(w.start + offset, 3),
                "end": round(w.end + offset, 3),
                "word": w.word,
                "probability": round(w.probability, 4),
            }
            for w in seg.words
        ]
    return entry


def _transcribe_ranges(model, batched, audio_array, ranges, *,
                       language, word_timestamps, initial_prompt):
    """Transcribe each (start, end) range and stitch results with time offsets."""
    segments: list[dict] = []
    full_text_parts: list[str] = []
    info = None
    detected_language = language
    next_id = 0

    for start, end in ranges:
        s = int(round(start * SAMPLE_RATE))
        e = int(round(end * SAMPLE_RATE))
        clip = audio_array[s:e]
        if clip.shape[0] == 0:
            continue

        kwargs = dict(
            beam_size=BEAM_SIZE,
            language=detected_language,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
        )
        if batched is not None:
            seg_iter, clip_info = batched.transcribe(
                clip, batch_size=BATCH_SIZE, **kwargs
            )
        else:
            seg_iter, clip_info = model.transcribe(clip, **kwargs)

        if info is None:
            info = clip_info
            # Reuse the first detected language so the whole song stays consistent.
            if detected_language is None:
                detected_language = clip_info.language

        for seg in seg_iter:
            segments.append(_seg_to_entry(seg, next_id, start, word_timestamps))
            full_text_parts.append(seg.text)
            next_id += 1

    return segments, full_text_parts, info


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

    # Optional structure segments (e.g. allin1 output). When supplied, we
    # transcribe only the non-skipped time ranges instead of relying on VAD.
    segments_raw = request.form.get("segments") or None
    skip_labels = {
        s.strip().lower()
        for s in request.form.get("skip_labels", "silence").split(",")
        if s.strip()
    } or DEFAULT_SKIP_LABELS

    structure_segments = None
    if segments_raw:
        try:
            parsed = json.loads(segments_raw)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid 'segments' JSON: {e}"}), 400
        structure_segments = _extract_segments(parsed)
        if not structure_segments:
            return jsonify({
                "error": "Could not find a list of {start, end, label} objects "
                         "in 'segments'.",
            }), 400

    safe_name = secure_filename(audio.filename)
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    audio.save(tmp_path)

    try:
        model, batched = get_model()

        # --- Structure-segment mode: transcribe explicit ranges ------------
        if structure_segments is not None:
            ranges = _build_ranges(structure_segments, skip_labels)
            if not ranges:
                return jsonify({
                    "error": "No transcribable ranges left after skipping labels "
                             f"{sorted(skip_labels)}.",
                }), 400

            logger.info(
                "Transcribing %s in structure-segment mode "
                "(%d range(s), lang=%s, word_ts=%s, skip=%s)",
                tmp_path.name, len(ranges), language, word_timestamps,
                sorted(skip_labels),
            )

            audio_array = decode_audio(str(tmp_path), sampling_rate=SAMPLE_RATE)
            with _transcribe_lock:
                segments, full_text_parts, info = _transcribe_ranges(
                    model, batched, audio_array, ranges,
                    language=language,
                    word_timestamps=word_timestamps,
                    initial_prompt=initial_prompt,
                )

            return jsonify({
                "mode": "structure-segments",
                "language": info.language if info else (language or "unknown"),
                "language_probability": (
                    round(info.language_probability, 4) if info else None
                ),
                "duration": round(len(audio_array) / SAMPLE_RATE, 3),
                "text": "".join(full_text_parts).strip(),
                "ranges": [
                    {"start": round(a, 3), "end": round(b, 3)} for a, b in ranges
                ],
                "segments": segments,
            })

        # --- Default mode: whole file, optional VAD ------------------------
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
                segments.append(_seg_to_entry(seg, seg.id, 0.0, word_timestamps))
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
