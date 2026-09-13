"""Start, health-gate and stop the subprocess serving one audio model.

The core router manages the models in *its* registry and evicts them per its
swap policy. Speech backends are not in that registry, so this owns their
lifecycle instead — which is simpler than it sounds, because they are always
resident: start on first use, stop when arc-llama stops. There is no swap
policy to implement and nothing to drain.

``LlamaServer`` from the core does the actual work. Despite the name it is
runtime-agnostic — it drives whatever ``plan.argv`` says and gates on
``plan.health_url`` returning ``{"status": "ok"}`` — so both a llama-server
transcription backend and a Python TTS sidecar reuse it as-is.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from arc_llama.config import Config
from arc_llama.launcher import LaunchPlan, LlamaServer

from arc_llama_audio.config import AudioConfig, AudioModelConfig

log = logging.getLogger("arc_llama_audio.backends")


class Backends:
    """One ``LlamaServer`` per audio model, started on demand."""

    def __init__(self) -> None:
        self._servers: dict[str, LlamaServer] = {}
        self._lock = asyncio.Lock()
        self.launch_errors: dict[str, str] = {}

    async def ensure(
        self, model: AudioModelConfig, cfg: Config, audio: AudioConfig,
        log_dir: Path | None,
    ) -> LlamaServer:
        """Return a healthy backend for *model*, starting it if needed.

        The fast path takes no lock, so a warm backend costs one dict lookup.
        Concurrent first requests serialise on the lock and the later ones
        find the server already up rather than starting a second copy.
        """
        server = self._servers.get(model.name)
        if server is not None and server.is_running and server.ready:
            return server

        async with self._lock:
            server = self._servers.get(model.name)
            if server is not None and server.is_running and server.ready:
                return server
            plan = build_plan(cfg, audio, model)
            server = LlamaServer(plan, name=model.name)
            log.info("starting %s backend for %s", model.task, model.name)
            server.start(log_dir=log_dir)
            if not await server.wait_ready():
                tail = server.tail_log(lines=40)
                server.stop()
                detail = f"{model.name} did not become healthy"
                if tail:
                    detail += "\n\n--- last log lines ---\n" + tail
                self.launch_errors[model.name] = detail
                raise RuntimeError(detail)
            self.launch_errors.pop(model.name, None)
            self._servers[model.name] = server
            return server

    async def shutdown(self) -> None:
        async with self._lock:
            for server in self._servers.values():
                await server.astop()
            self._servers.clear()

    def loaded(self) -> list[str]:
        return [n for n, s in self._servers.items() if s.is_running and s.ready]


def build_plan(cfg: Config, audio: AudioConfig, model: AudioModelConfig) -> LaunchPlan:
    """Dispatch to the engine that serves *model*.

    Transcription is always llama-server; synthesis is whichever engine the
    entry names, which is the only place an engine name is dispatched on.
    """
    gpu = cfg.find_gpu(model.gpu_pci_slot)
    if gpu is None:
        raise RuntimeError(
            f"{model.name} is bound to GPU {model.gpu_pci_slot}, which is not in "
            "arc-llama's config. Run `arc-llama gpus` and re-register it."
        )
    if model.task == "tts":
        from arc_llama_audio.tts import require_engine

        engine = require_engine(model.engine)
        plan = engine.build_plan(cfg, audio, model, gpu, host=cfg.server.host)
        if plan.health_timeout is None:
            plan.health_timeout = engine.health_timeout
        return plan
    from arc_llama_audio.asr_plan import build_asr_plan

    return build_asr_plan(cfg, model, gpu, host=cfg.server.host)
