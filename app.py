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

# Amazon Bedrock (Nova) settings for lyric correction.
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get(
    "AWS_DEFAULT_REGION", "us-east-1"
)
NOVA_MODEL_ID = os.environ.get("NOVA_MODEL_ID", "amazon.nova-pro-v1:0")
NOVA_MAX_TOKENS = int(os.environ.get("NOVA_MAX_TOKENS", "4096"))
NOVA_TEMPERATURE = float(os.environ.get("NOVA_TEMPERATURE", "0.2"))

# Web search tool (lets Nova look up accurate lyrics while correcting).
# Provider auto-detects from available keys unless WEB_SEARCH_PROVIDER is set:
#   tavily (TAVILY_API_KEY) -> serper (SERPER_API_KEY) -> duckduckgo (no key).
WEB_SEARCH_ENABLED = os.environ.get("LYRICS_WEB_SEARCH", "1") == "1"
WEB_SEARCH_PROVIDER = os.environ.get("WEB_SEARCH_PROVIDER", "").strip().lower()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "15"))
# Max tool-use round-trips before we force Nova to answer.
NOVA_MAX_TOOL_ITERS = int(os.environ.get("NOVA_MAX_TOOL_ITERS", "4"))

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

_bedrock_lock = threading.Lock()
_bedrock_client = None  # boto3 bedrock-runtime client, created lazily


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


def get_bedrock():
    """Lazily create a thread-safe Bedrock runtime client (boto3).

    Uses the standard AWS credential chain (env vars, shared config, or the
    EC2 instance role), so no secrets live in this app.
    """
    global _bedrock_client
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:
                import boto3  # imported lazily so transcription works without it

                logger.info(
                    "Creating Bedrock client (region=%s, model=%s)",
                    AWS_REGION, NOVA_MODEL_ID,
                )
                _bedrock_client = boto3.client(
                    "bedrock-runtime", region_name=AWS_REGION
                )
    return _bedrock_client


# ---------------------------------------------------------------------------
# Web search tool (grounding for lyric correction)
# ---------------------------------------------------------------------------
def _active_search_provider() -> Optional[str]:
    """Resolve which search provider to use, or None if web search is off."""
    if not WEB_SEARCH_ENABLED:
        return None
    if WEB_SEARCH_PROVIDER:
        return WEB_SEARCH_PROVIDER
    if TAVILY_API_KEY:
        return "tavily"
    if SERPER_API_KEY:
        return "serper"
    return "duckduckgo"


def _format_results(results: list[dict]) -> str:
    """Render search results (title/url/content dicts) as text for the model."""
    if not results:
        return "No results found."
    blocks = []
    for r in results[:WEB_SEARCH_MAX_RESULTS]:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        if len(content) > 1500:
            content = content[:1500] + "..."
        blocks.append(f"Title: {title}\nURL: {url}\nContent: {content}")
    return "\n\n---\n\n".join(blocks)


def _search_tavily(query: str) -> list[dict]:
    import requests

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": WEB_SEARCH_MAX_RESULTS,
        },
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
        for r in data.get("results", [])
    ]


def _search_serper(query: str) -> list[dict]:
    import requests

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": WEB_SEARCH_MAX_RESULTS},
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title"), "url": r.get("link"), "content": r.get("snippet")}
        for r in data.get("organic", [])
    ]


def _search_duckduckgo(query: str) -> list[dict]:
    """Keyless fallback using DuckDuckGo's HTML endpoint."""
    import html
    import re

    import requests

    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; lyrics-bot/1.0)"},
        timeout=WEB_SEARCH_TIMEOUT,
    )
    resp.raise_for_status()

    results: list[dict] = []
    # Each result: <a class="result__a" href="URL">TITLE</a> ... snippet.
    link_re = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.S,
    )
    snippet_re = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S
    )
    tag_re = re.compile(r"<[^>]+>")

    titles = link_re.findall(resp.text)
    snippets = snippet_re.findall(resp.text)
    for i, (url, title) in enumerate(titles):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({
            "title": html.unescape(tag_re.sub("", title)).strip(),
            "url": html.unescape(url).strip(),
            "content": html.unescape(tag_re.sub("", snippet)).strip(),
        })
        if len(results) >= WEB_SEARCH_MAX_RESULTS:
            break
    return results


def web_search(query: str) -> str:
    """Run a web search with the active provider and return text for the model."""
    provider = _active_search_provider()
    logger.info("web_search(provider=%s, query=%r)", provider, query)
    try:
        if provider == "tavily":
            results = _search_tavily(query)
        elif provider == "serper":
            results = _search_serper(query)
        else:
            results = _search_duckduckgo(query)
    except Exception as e:  # surface error to the model instead of crashing
        logger.warning("web_search failed: %s", e)
        return f"Web search failed: {e}"
    return _format_results(results)


WEB_SEARCH_TOOL = {
    "toolSpec": {
        "name": "web_search",
        "description": (
            "Search the web for the official, accurate lyrics of a song. Use "
            "this to verify words, fill gaps, and confirm section structure. "
            "Prefer queries like '<song> <artist> lyrics'."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'Golden Brown The Stranglers lyrics'.",
                    }
                },
                "required": ["query"],
            }
        },
    }
}


CORRECTION_SYSTEM_PROMPT = (
    "You are an expert music lyrics editor. You will be given the raw output of "
    "an automatic speech-to-text system applied to a song. That transcription "
    "often contains misheard words, run-on lines, missing words, and "
    "duplicated phrases. Your job is to produce the correct, clean lyrics.\n\n"
    "Rules:\n"
    "- When a web_search tool is available, USE IT to look up the official "
    "lyrics of the named song before finalizing. Search '<song> <artist> "
    "lyrics', then reconcile the transcription against the real lyrics. Trust "
    "reputable lyrics sources over the noisy transcription.\n"
    "- Use your knowledge of the named song (and artist, if given) to fix "
    "misheard words and restore the real lyrics.\n"
    "- Keep the original language; do not translate.\n"
    "- Organize the lyrics into sections. Put a section header on its own line "
    "in square brackets, then the lines of that section, then a blank line "
    "before the next section.\n"
    "- Use standard section labels: [Intro], [Verse 1], [Verse 2], "
    "[Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Refrain], [Outro], etc. "
    "Number repeated verses ([Verse 1], [Verse 2], ...). Include [Intro] and "
    "[Outro] headers even when those parts are instrumental (leave them with no "
    "lyric lines underneath).\n"
    "- Fix line breaks so each line is a natural lyric line.\n"
    "- Keep backing vocals, echoes, and ad-libs in parentheses on the same line, "
    "e.g. \"Never a frown (never a frown)\".\n"
    "- Remove obvious ASR artifacts and accidental repetitions.\n"
    "- Do not invent whole sections that are not supported by the transcription "
    "unless you are confident they are the song's actual lyrics.\n"
    "- Return ONLY the formatted lyrics as plain text, starting with the first "
    "section header. No preamble, no explanations, no markdown fences.\n\n"
    "Example of the required format:\n"
    "[Intro]\n\n"
    "[Verse 1]\n"
    "Golden brown, texture like sun\n"
    "Lays me down, with my mind she runs\n\n"
    "[Chorus]\n"
    "Never a frown (never a frown)\n"
    "(Never a frown) with golden brown (with golden brown)\n\n"
    "[Outro]"
)


def _extract_text(message: dict) -> str:
    """Join the text blocks of a Converse message."""
    return "".join(
        b["text"] for b in message.get("content", []) if "text" in b
    ).strip()


def correct_lyrics_with_nova(song_name: str, lyrics: str) -> tuple[str, list[str]]:
    """Correct lyrics with a Bedrock Nova model, optionally using web search.

    Returns (corrected_text, search_queries_used).
    """
    client = get_bedrock()

    song_line = (
        f'Song: "{song_name}".' if song_name else "Song title: (unknown)."
    )
    user_text = (
        f"{song_line}\n\n"
        "Here is the raw speech-to-text transcription of the lyrics. "
        "Correct it per your instructions and return only the corrected "
        "lyrics:\n\n"
        f"{lyrics}"
    )

    messages = [{"role": "user", "content": [{"text": user_text}]}]
    inference_config = {
        "maxTokens": NOVA_MAX_TOKENS,
        "temperature": NOVA_TEMPERATURE,
    }
    use_tools = _active_search_provider() is not None
    searches: list[str] = []

    for _ in range(NOVA_MAX_TOOL_ITERS + 1):
        kwargs = dict(
            modelId=NOVA_MODEL_ID,
            system=[{"text": CORRECTION_SYSTEM_PROMPT}],
            messages=messages,
            inferenceConfig=inference_config,
        )
        if use_tools:
            kwargs["toolConfig"] = {"tools": [WEB_SEARCH_TOOL]}

        response = client.converse(**kwargs)
        out_msg = response["output"]["message"]
        messages.append(out_msg)

        if response.get("stopReason") != "tool_use":
            return _extract_text(out_msg), searches

        # Run every requested tool and feed the results back to the model.
        tool_results = []
        for block in out_msg.get("content", []):
            tool_use = block.get("toolUse")
            if not tool_use:
                continue
            query = (tool_use.get("input") or {}).get("query", "")
            if tool_use.get("name") == "web_search":
                searches.append(query)
                result_text = web_search(query)
            else:
                result_text = f"Unknown tool: {tool_use.get('name')}"
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use["toolUseId"],
                    "content": [{"text": result_text}],
                }
            })
        messages.append({"role": "user", "content": tool_results})

    # Tool budget exhausted: ask once more without tools for a final answer.
    response = client.converse(
        modelId=NOVA_MODEL_ID,
        system=[{"text": CORRECTION_SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig=inference_config,
    )
    return _extract_text(response["output"]["message"]), searches


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


@app.route("/correct", methods=["POST"])
def correct():
    """Correct transcribed lyrics with a Bedrock Nova model.

    Accepts JSON or form fields: `song_name` (optional) and `lyrics` (required).
    """
    data = request.get_json(silent=True) or request.form
    song_name = (data.get("song_name") or "").strip()
    lyrics = (data.get("lyrics") or "").strip()

    if not lyrics:
        return jsonify({"error": "No 'lyrics' provided to correct."}), 400

    try:
        logger.info(
            "Correcting lyrics with Nova (song=%r, chars=%d, model=%s)",
            song_name, len(lyrics), NOVA_MODEL_ID,
        )
        corrected, searches = correct_lyrics_with_nova(song_name, lyrics)
        return jsonify({
            "song_name": song_name,
            "model": NOVA_MODEL_ID,
            "corrected": corrected,
            "web_search_provider": _active_search_provider(),
            "searches": searches,
        })
    except Exception as e:
        logger.exception("Lyric correction failed")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File too large. Limit is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    # Pre-load model so the first request isn't slow.
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
