"""Shared request plumbing for both audio endpoints.

The core's own proxy is JSON-only by construction — it reads `model` out of a
parsed body and forwards the original bytes — while these routes have to
understand multipart uploads and binary responses, so they do their own
forwarding rather than trying to reuse it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

# `request.form()` yields Starlette's UploadFile. fastapi.UploadFile is a
# *subclass* of it, so isinstance() against the FastAPI one silently misses
# every real upload — match the base class instead.
from starlette.datastructures import UploadFile as FormUploadFile

from arc_llama_audio.backends import Backends
from arc_llama_audio.config import AudioConfig, AudioModelConfig, load_config

log = logging.getLogger("arc_llama_audio.proxy")

# Uploaded audio is buffered in memory to be re-encoded, so the request body
# needs its own ceiling. A minute of 16 kHz mono WAV is under 2 MB.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_SPEECH_BODY_BYTES = 64 * 1024

# Qwen3-ASR's native output framing. `<asr_text>` is a real token in its
# vocabulary, and the model prefixes each transcript with a detected-language
# announcement: `language English<asr_text>the actual words`. llama.cpp
# forwards that verbatim (ggml-org/llama.cpp#26749).
_ASR_TEXT_MARKER = "<asr_text>"
_ASR_AUDIO_TOKENS = ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>")


def strip_asr_markers(text: str) -> str:
    """Remove a transcription model's output framing from *text*.

    Everything up to and including the last `<asr_text>` is the model
    announcing what it is about to do, not speech anyone said. A voice
    assistant matching intents against the raw string fails on every
    utterance.

    Splitting on the *last* marker degrades to a no-op for a model that never
    emits one, and a transcript that genuinely contained the literal token
    cannot occur — it is reserved and never produced from audio.
    """
    if _ASR_TEXT_MARKER in text:
        text = text.rsplit(_ASR_TEXT_MARKER, 1)[1]
    for token in _ASR_AUDIO_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def sanitize_transcription(content: bytes) -> bytes:
    """Apply ``strip_asr_markers`` to a `{"text": ...}` response body.

    Anything else is returned untouched: an error payload or an unexpected
    schema is the backend's to explain, and rewriting it would obscure the
    real failure.
    """
    try:
        body = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return content
    cleaned = strip_asr_markers(body["text"])
    if cleaned == body["text"]:
        return content
    body["text"] = cleaned
    return json.dumps(body).encode("utf-8")


# -- plugin state -----------------------------------------------------
#
# Both plugins share one config and one backend registry. `register` runs
# before the core's lifespan populates app.state, so these are created on
# first use rather than at import.


def audio_config(app: Any) -> AudioConfig:
    cfg = getattr(app.state, "audio_config", None)
    if cfg is None:
        cfg = load_config()
        app.state.audio_config = cfg
    return cfg


def backends(app: Any) -> Backends:
    registry = getattr(app.state, "audio_backends", None)
    if registry is None:
        registry = Backends()
        app.state.audio_backends = registry
    return registry


def require_model(app: Any, query: str, task: str, empty_detail: str) -> AudioModelConfig:
    audio = audio_config(app)
    model = audio.find_model(query, task=task)
    if model is not None:
        return model
    registered = [m.name for m in audio.audio_models if m.task == task]
    if not registered:
        raise HTTPException(status_code=501, detail=empty_detail)
    raise HTTPException(
        status_code=404,
        detail=f"Unknown {task} model: {query!r}. Registered: {', '.join(registered)}",
    )


def _log_dir(app: Any) -> Path | None:
    state_dir = getattr(app.state.cfg.paths, "state_dir", "")
    return Path(state_dir).expanduser() if state_dir else None


async def forward(
    request: Request,
    model: AudioModelConfig,
    target_path: str,
    kwargs: dict[str, Any],
    *,
    want_stream: bool = False,
    sanitize: bool = False,
):
    """Start the backend serving *model* if needed, and forward one request."""
    app = request.app
    try:
        server = await backends(app).ensure(
            model, app.state.cfg, audio_config(app), _log_dir(app)
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    url = f"{server.plan.backend_url}{target_path}"

    if want_stream:
        # Deltas are forwarded raw, so sanitising does not apply: the framing
        # arrives split across chunks and rewriting it would mean buffering
        # the stream, which defeats asking for one.
        client = httpx.AsyncClient(timeout=None)
        try:
            upstream = await client.send(
                client.build_request("POST", url, **kwargs), stream=True
            )
        except BaseException:
            await client.aclose()
            raise

        released = False

        async def _release() -> None:
            # Idempotent, and reached from both the generator's finally and
            # the background task: Starlette only runs the latter when the
            # body streamed cleanly, so an upstream that dies mid-generation
            # would otherwise leak the connection and the client pool.
            nonlocal released
            if released:
                return
            released = True
            for closer in (upstream.aclose, client.aclose):
                try:
                    await closer()
                except Exception:
                    log.debug("audio proxy: close failed", exc_info=True)

        async def body_iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await _release()

        return StreamingResponse(
            body_iter(),
            status_code=upstream.status_code,
            headers=_strip_hop_by_hop(dict(upstream.headers)),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(_release),
        )

    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(url, **kwargs)
    content = sanitize_transcription(r.content) if sanitize and r.status_code == 200 else r.content
    return Response(
        content=content,
        status_code=r.status_code,
        headers=_strip_hop_by_hop(dict(r.headers)),
        media_type=r.headers.get("content-type", "application/json"),
    )


def _strip_hop_by_hop(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding", "content-length", "connection")
    }


async def read_multipart_or_json(
    request: Request,
) -> tuple[bool, dict[str, str], tuple[str, bytes, str] | None, bytes]:
    """Parse either shape the transcription endpoint accepts.

    Returns (is_multipart, form fields, upload, raw json body).
    """
    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.lower().startswith("multipart/form-data")

    # Refuse an oversized upload from the declared length, before the body is
    # read: checking after would mean buffering the very payload the limit
    # exists to refuse.
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio upload must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None

    if not is_multipart:
        return False, {}, None, await request.body()

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid multipart body: {e}") from e
    fields: dict[str, str] = {}
    upload: tuple[str, bytes, str] | None = None
    try:
        for key, value in form.multi_items():
            if isinstance(value, FormUploadFile):
                if key != "file":
                    continue
                content = await value.read()
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio upload must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                    )
                upload = (
                    value.filename or "audio.wav",
                    content,
                    value.content_type or "application/octet-stream",
                )
            else:
                fields[key] = str(value)
    finally:
        await form.close()
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing 'file' in multipart body")
    return True, fields, upload, b""
