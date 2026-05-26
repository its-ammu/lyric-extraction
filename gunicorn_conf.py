"""Gunicorn configuration for the lyrics transcriber.

Run with:
    gunicorn -c gunicorn_conf.py app:app

Key behavior:
- Single worker, multi-threaded: only one Whisper model lives in GPU memory.
- `post_worker_init` warms the model after fork, so the first /transcribe
  request doesn't pay the model-load cost. We deliberately do NOT use
  `--preload`, because a CUDA context initialized in the master process
  cannot be shared with forked children and will raise
  "Cannot re-initialize CUDA in forked subprocess".
"""
from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")


def post_worker_init(worker):
    """Load the Whisper model immediately after the worker process forks.

    Runs once per worker (typically once total, since workers=1). After this
    returns, the worker is ready to serve requests with a hot model.
    """
    worker.log.info("Pre-loading faster-whisper model in worker %s...", worker.pid)
    from app import get_model  # imported here so it runs inside the worker

    get_model()
    worker.log.info("Model ready in worker %s.", worker.pid)
