"""Speech-to-text plugin: OpenAI `/v1/audio/transcriptions`."""
from __future__ import annotations

import json
import logging
from typing import Any

from arc_llama.plugins import Plugin
from fastapi import FastAPI, HTTPException, Request

from arc_llama_audio import proxy

log = logging.getLogger("arc_llama_audio.asr")

_NO_MODELS = (
    "No transcription models are registered. Add one with "
    "`arc-llama-audio add <weights.gguf> --task asr`."
)


class AsrPlugin(Plugin):
    name = "audio-asr"

    def register(self, app: FastAPI) -> None:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request) -> Any:
            """Accepts both shapes clients send: a multipart upload (Home
            Assistant, Open WebUI, the OpenAI SDKs) and a JSON body naming a
            server-local path."""
            is_multipart, fields, upload, body_bytes = await proxy.read_multipart_or_json(request)
            if is_multipart:
                model_query = fields.get("model", "")
            else:
                try:
                    body = json.loads(body_bytes) if body_bytes else {}
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
                if not isinstance(body, dict):
                    raise HTTPException(status_code=400, detail="Body must be a JSON object")
                model_query = str(body.get("model", ""))

            model = proxy.require_model(request.app, model_query, "asr", _NO_MODELS)

            want_stream = str(fields.get("stream", "")).lower() == "true"
            if want_stream and model.mode != "streaming":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model {model.name!r} is configured mode='offline'; "
                        "stream=true needs a model registered with --mode streaming."
                    ),
                )

            # The backend only answers to its own id, whatever alias got us
            # here, so the model field is rewritten rather than passed through.
            if is_multipart:
                assert upload is not None
                kwargs = {"data": {**fields, "model": model.name}, "files": {"file": upload}}
            else:
                parsed = json.loads(body_bytes) if body_bytes else {}
                parsed["model"] = model.name
                kwargs = {"json": parsed}

            return await proxy.forward(
                request,
                model,
                "/v1/audio/transcriptions",
                kwargs,
                want_stream=want_stream,
                sanitize=model.strip_asr_markers,
            )

    async def shutdown(self, app: FastAPI) -> None:
        # Both plugins share one registry; stopping it twice is a no-op, and
        # whichever runs first returns the VRAM.
        await proxy.backends(app).shutdown()
