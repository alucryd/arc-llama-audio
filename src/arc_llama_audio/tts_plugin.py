"""Text-to-speech plugin: OpenAI `/v1/audio/speech`."""
from __future__ import annotations

import json
import logging
from typing import Any

from arc_llama.plugins import Plugin
from fastapi import FastAPI, HTTPException, Request

from arc_llama_audio import proxy

log = logging.getLogger("arc_llama_audio.tts")

_NO_MODELS = (
    "No speech models are registered. Add one with "
    "`arc-llama-audio add <model> --task tts --engine omnivoice`."
)


class TtsPlugin(Plugin):
    name = "audio-tts"

    def register(self, app: FastAPI) -> None:
        @app.post("/v1/audio/speech")
        async def speech(request: Request) -> Any:
            body_bytes = await request.body()
            if len(body_bytes) > proxy.MAX_SPEECH_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Speech request must be under "
                    f"{proxy.MAX_SPEECH_BODY_BYTES // 1024} KB",
                )
            try:
                body = json.loads(body_bytes) if body_bytes else {}
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Body must be a JSON object")

            text = body.get("input")
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(status_code=400, detail="'input' must be a non-empty string")

            model = proxy.require_model(request.app, str(body.get("model", "")), "tts", _NO_MODELS)

            from arc_llama_audio.tts import require_engine

            try:
                engine = require_engine(model.engine)
            except ValueError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e

            # The engine translates the OpenAI body into whatever its backend
            # wants — that translation is the whole reason engines exist, and
            # folding a second request shape into this function would undo it.
            payload = engine.build_payload(model, body)
            return await proxy.forward(
                request, model, engine.speech_path, {"json": payload}
            )

        @app.get("/v1/audio/voices")
        async def voices(request: Request) -> dict[str, Any]:
            """List registered voices, so a client can populate a picker."""
            audio = proxy.audio_config(request.app)
            return {
                "object": "list",
                "data": [
                    {
                        "id": v.name,
                        "display_name": v.display_name,
                        "aliases": list(v.aliases),
                        "language": v.language,
                        "models": list(v.models),
                    }
                    for v in audio.voices
                ],
            }

    async def shutdown(self, app: FastAPI) -> None:
        await proxy.backends(app).shutdown()
