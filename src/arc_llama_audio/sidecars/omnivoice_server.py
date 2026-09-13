"""A minimal OpenAI `/v1/audio/speech` server in front of an OmniVoice model.

Run as a standalone script by whatever interpreter has OmniVoice installed:

    <python> .../arc_llama/tts/omnivoice_server.py --model k2-fsa/OmniVoice \
        --host 127.0.0.1 --port 8090 --device xpu --voices /path/voices.json

**Nothing here may import `arc_llama`.** OmniVoice pulls in torch, transformers
and torchaudio, which arc-llama deliberately does not depend on, so it lives in
its own virtualenv and this file is executed by that virtualenv's interpreter.
The only contract with the parent is the command line, the voices JSON, and the
two HTTP routes below — which is also what makes the engine swappable.

It sits in `sidecars/` rather than beside the engine module for a reason:
running a script puts its own directory at the front of `sys.path`, so a
neighbour named after one of the imports below would shadow it. Living next to
`omnivoice.py` meant `from omnivoice import OmniVoice` resolved to arc-llama's
engine module and failed on a machine with OmniVoice plainly installed.

Only the standard library is used for the HTTP side. The alternative (FastAPI,
matching the parent) would be a dependency the OmniVoice environment has no
reason to carry, and a TTS backend serialises on one GPU anyway, so there is
nothing for an async server to overlap.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("omnivoice_server")

# OpenAI's documented set. `wav`/`pcm` are written by hand below, the rest go
# through soundfile and then ffmpeg, so which of them actually work depends on
# the environment; the request path reports what it could not produce.
RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

# OpenAI's `pcm` is documented as 24 kHz 16-bit signed little-endian mono,
# which is exactly OmniVoice's native output rate — so `pcm` is a raw dump and
# never resamples. A model with a different rate is served correctly in every
# container format that carries a rate, and only `pcm` would mislead a client;
# that case warns at startup rather than silently retuning the audio.
OPENAI_PCM_RATE = 24000

MAX_INPUT_CHARS = 8192

#: Submodules compiled by `--compile`, in the order they are tried.
#:
#: Deliberately *not* the top-level model. `torch.compile(model)` returns a
#: wrapper whose compiled entry point is `forward`, and this server calls
#: `model.generate(...)` — which `OptimizedModule.__getattr__` forwards to the
#: original module, so `self` inside `generate` is the uncompiled model and
#: every op runs eager. Compiling the modules `generate` actually calls is the
#: only thing that reaches the hot loop. These two are also exactly what the
#: int8 path quantizes, which matters because torchao's weight-only kernels are
#: written to be fused by Inductor: quantized *and* uncompiled is the slowest
#: combination available, paying dequantization without the fusion that repays
#: it.
DEFAULT_COMPILE_TARGETS = "llm,audio_heads"

#: Encode time above which switching container is worth suggesting.
#:
#: A libsndfile mp3 encode of a short utterance measures ~17 ms, which is
#: nothing next to a 300 ms synthesis — and mp3 is both what OpenAI clients
#: default to and far smaller over the air, which a wifi speaker cares about.
#: Only the ffmpeg fallback, a process spawn per request, is worth avoiding.
_SLOW_ENCODE_S = 0.1


def quiet_inductor_noise() -> None:
    """Stop Inductor's extern-choice warning from burying everything else.

    `select_algorithm` logs "Constructing input/output tensor meta failed for
    Extern Choice" once per op whenever it cannot resolve a size into a
    concrete one, which on a transformer means hundreds of identical lines in
    a couple of seconds. It is not a failure — the compile continues, the
    choice simply carries empty metadata — but it drowns the load timings and
    the benchmark table, which are the two things anyone is reading this
    output for. Set ARC_LLAMA_TTS_LOG=DEBUG to see them again.
    """
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.ERROR)


def _inference_context() -> Any:
    """`torch.inference_mode()` when torch is importable, else a no-op.

    Guarded because this module is unit-tested against a stub model in an
    environment that has no torch, and because a missing torch here should
    surface as the model failing to load rather than as an import error from
    the request path.
    """
    try:
        import torch
    except Exception:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _device_sync(device: str) -> None:
    """Wait for queued device work, so a measured span means what it says.

    XPU and CUDA queues are asynchronous: without this, `generate` appears to
    return in milliseconds and the cost lands in whatever is timed next.
    """
    try:
        import torch
    except Exception:
        return
    backend = getattr(torch, (device or "").split(":", 1)[0], None)
    sync = getattr(backend, "synchronize", None)
    if not callable(sync):
        return
    try:
        sync()
    except Exception:
        log.debug("could not synchronize device %r", device, exc_info=True)


def dynamo_stats() -> dict[str, int]:
    """Dynamo's compile counters, or {} when torch is absent or has moved them.

    Read defensively: these live under a private module and their key names
    are not API. A missing counter should cost a diagnostic line, never a
    benchmark run.
    """
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    out: dict[str, int] = {}
    try:
        for group in ("frames", "stats", "graph_break"):
            entries = counters.get(group) or {}
            for key, value in entries.items():
                if isinstance(value, int):
                    out[f"{group}.{key}"] = value
    except Exception:
        log.debug("could not read dynamo counters", exc_info=True)
    return out


def _audio_seconds(samples: Any, rate: int) -> float:
    """Duration of a sample buffer, for the real-time factor in the log."""
    if not rate:
        return 0.0
    try:
        count = len(samples)
    except TypeError:
        numel = getattr(samples, "numel", None)
        count = int(numel()) if callable(numel) else 0
    return count / rate


class VoiceBook:
    """The voice table, reloaded from disk whenever the file changes.

    Re-reading on each request is what lets `arc-llama audio voice add` take
    effect without restarting the backend — the alternative, baking voices into
    argv at launch, would make adding a voice cost a model reload (tens of
    seconds and a VRAM round-trip) for what is a one-line config edit.

    Encoded clone prompts are cached in memory, keyed by the voice definition
    itself, so an edited voice re-encodes and an untouched one never does.
    """

    def __init__(self, path: str | None):
        self.path = Path(path).expanduser() if path else None
        self._mtime: float | None = None
        self._data: dict[str, Any] = {}
        self._prompts: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _reload_if_stale(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Keep serving the last good table: a half-written file during an
            # `arc-llama audio voice add` must not take TTS down.
            log.warning("could not reload voices from %s: %s", self.path, exc)
            return
        self._mtime = mtime
        self._data = raw if isinstance(raw, dict) else {}
        self._prompts.clear()
        log.info("loaded %d voice(s) from %s", len(self.voices), self.path)

    @property
    def voices(self) -> dict[str, Any]:
        entries = self._data.get("voices", {})
        return entries if isinstance(entries, dict) else {}

    @property
    def default_voice(self) -> str:
        return str(self._data.get("default_voice", "") or "")

    def lookup(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """Resolve a `voice` field to (canonical name, definition).

        Matching is exact, then case-insensitive, then over each voice's
        aliases. A client that hardcodes one of OpenAI's voice ids ("alloy")
        gets whatever the user registered under that alias, or falls through to
        the default voice rather than an error — an unknown voice is a much
        worse failure for a speech client than a substituted one.
        """
        with self._lock:
            self._reload_if_stale()
            entries = self.voices
            if name:
                if name in entries:
                    return name, dict(entries[name])
                lowered = {k.lower(): k for k in entries}
                if name.lower() in lowered:
                    key = lowered[name.lower()]
                    return key, dict(entries[key])
                for key, entry in entries.items():
                    aliases = entry.get("aliases") or []
                    if any(str(a).lower() == name.lower() for a in aliases):
                        return key, dict(entry)
            fallback = self.default_voice
            if fallback and fallback in entries:
                return fallback, dict(entries[fallback])
            return None

    def cached_prompt(self, key: str, definition: dict[str, Any]) -> Any | None:
        with self._lock:
            hit = self._prompts.get(key)
            if hit is not None and hit[0] == definition:
                return hit[1]
            return None

    def store_prompt(self, key: str, definition: dict[str, Any], prompt: Any) -> None:
        with self._lock:
            self._prompts[key] = (dict(definition), prompt)


class Engine:
    """The loaded OmniVoice model plus the generation lock around it."""

    def __init__(self, args: argparse.Namespace, voices: VoiceBook):
        self.args = args
        self.voices = voices
        self.model: Any = None
        self.sampling_rate = OPENAI_PCM_RATE
        self.load_error: str | None = None
        # One GPU, one model, and `generate` is not re-entrant. Serialise here
        # rather than in the HTTP layer so a threaded server stays correct.
        self.gpu_lock = threading.Lock()
        self._asr_loaded = False

    @property
    def ready(self) -> bool:
        return self.model is not None

    def warn_on_foreign_sycl_runtime(self) -> None:
        """Warn when a second Intel runtime is on the library path.

        A torch XPU build ships its own libsycl, Unified Runtime adapters and
        MKL. Putting a system oneAPI installation ahead of them loads two
        Intel runtimes into one process, and the failure is a SIGSEGV inside
        the SYCL device-code build rather than anything that names the cause —
        so say it here, where the environment is still readable.
        """
        entries = [
            part
            for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if part and "oneapi" in part.lower()
        ]
        if entries:
            log.warning(
                "LD_LIBRARY_PATH points at a system oneAPI install (%s). torch "
                "carries its own SYCL runtime, and loading both into one "
                "process crashes in the device-code build. Unset it for this "
                "service if the backend dies during model load.",
                ", ".join(entries[:2]),
            )

    def load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.warn_on_foreign_sycl_runtime()

        dtype = getattr(torch, self.args.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown torch dtype: {self.args.dtype!r}")
        log.info("loading %s onto %s (%s)", self.args.model, self.args.device, self.args.dtype)
        kwargs: dict[str, Any] = {"device_map": self.args.device, "dtype": dtype}
        if self.args.asr_model:
            kwargs["asr_model_name"] = self.args.asr_model
        if self.args.asr_device:
            kwargs["asr_device"] = self.args.asr_device
        started = time.perf_counter()
        if self.args.quantize:
            model = self._load_quantized(kwargs)
        else:
            model = OmniVoice.from_pretrained(self.args.model, **kwargs)
        weights_s = time.perf_counter() - started
        log.info("weights resident after %.1f s", weights_s)
        if self.args.compile:
            # Opt-in: the first request pays the compile, and Inductor on XPU is
            # newer than the eager path, so it is not something to impose by
            # default on a backend whose first call is a user waiting for audio.
            self._compile(model)
        # Assigned last, because `ready` is "self.model is not None" and that is
        # what /health reports. Warming up after publishing the model would put
        # the first-call cost (lazy kernel init, an Inductor compile, the memory
        # pool growing) back inside a real request — the thing the warmup exists
        # to prevent.
        self.sampling_rate = int(getattr(model, "sampling_rate", None) or OPENAI_PCM_RATE)
        warmup_s = self._warmup(model) if self.args.warmup else 0.0
        self.model = model
        if self.sampling_rate != OPENAI_PCM_RATE:
            log.warning(
                "model sampling rate is %d Hz, but OpenAI's `pcm` response format "
                "is defined as %d Hz mono s16le. Clients decoding raw pcm will play "
                "this back at the wrong speed; use response_format=wav instead.",
                self.sampling_rate, OPENAI_PCM_RATE,
            )
        # Attribute the startup cost, because "it took ten minutes" is not
        # actionable and its two halves have completely different remedies.
        # torch.compile() itself returns immediately — it only installs a
        # wrapper — so an Inductor compile is paid on the first forward, which
        # is the warmup. A large warmup number with --compile on is that
        # compile, and it is a once-per-process cost, not a per-utterance one.
        log.info(
            "ready: %s at %d Hz — %.1f s total (weights %.1f s, warmup %.1f s%s)",
            self.args.model, self.sampling_rate,
            time.perf_counter() - started, weights_s, warmup_s,
            ", including the torch.compile" if self.args.compile and warmup_s else "",
        )

    def _compile(self, model: Any) -> None:
        """Compile the submodules `generate` actually calls.

        See ``DEFAULT_COMPILE_TARGETS`` for why the top-level model is the
        wrong thing to hand to ``torch.compile``. Failures are logged and
        skipped: an uncompiled backend is slower, a backend that refuses to
        start is unusable.

        ``--compile-dynamic`` picks how shapes are treated, and the default is
        torch's own automatic mode rather than forced dynamic. Forcing it makes
        every size symbolic from the first trace, and Inductor cannot resolve a
        symbolic size into the concrete one it needs to build a benchmark
        request for a library (extern) kernel — so it logs

            Constructing input/output tensor meta failed for Extern Choice

        for every such op and carries on with empty metadata. Automatic mode
        instead specialises on the first shape and only re-traces as dynamic
        once it has actually seen a second one, which keeps the oneDNN/ATen
        matmuls in play for the common case. Force it either way if your
        utterance lengths vary enough that the recompiles cost more than the
        specialisation wins.
        """
        import torch

        dynamic = {"auto": None, "true": True, "false": False}[self.args.compile_dynamic]
        targets = [t.strip() for t in (self.args.compile_targets or "").split(",") if t.strip()]
        compiled: list[str] = []
        for attr in targets:
            sub = getattr(model, attr, None)
            if not isinstance(sub, torch.nn.Module):
                log.warning("--compile: %r is not a module on this model; skipping", attr)
                continue
            try:
                setattr(model, attr, torch.compile(sub, dynamic=dynamic))
                compiled.append(attr)
            except Exception:
                log.warning("--compile: could not compile %r", attr, exc_info=True)
        if compiled:
            log.info(
                "compiled %s with dynamic=%s (the first request pays for it)",
                ", ".join(compiled), self.args.compile_dynamic,
            )
        else:
            log.warning(
                "--compile was requested but nothing was compiled; generation "
                "will run eager. Set --compile-targets to this model's hot "
                "submodules."
            )

    def _warmup(self, model: Any) -> float:
        """Run one throwaway synthesis so the first real request is not the first.

        A cold torch backend defers a great deal to the first call: kernel
        loading and autotuning, the allocator's pool, and — when `--compile` is
        on — the entire Inductor compile, since `torch.compile` only installs a
        wrapper and traces on first use. On a voice assistant that cost lands
        on whoever speaks first after a restart, and it is the one request most
        likely to be judged.

        Returns the elapsed seconds, or 0.0 if it did not complete.
        """
        text = self.args.warmup_text or "Warming up."
        if self.args.compile:
            log.info("warming up; with --compile this is where the graph is built")
        started = time.perf_counter()
        try:
            with _inference_context():
                model.generate(text=text, num_step=self.args.num_step)
            _device_sync(self.args.device)
        except Exception:
            # Never fatal: a model whose generate() wants arguments this one
            # does not pass is still perfectly able to serve real requests,
            # which carry the voice and language the config supplies.
            log.warning("warmup synthesis failed; first request will be slower", exc_info=True)
            return 0.0
        elapsed = time.perf_counter() - started
        log.info("warmup synthesis took %.2f s", elapsed)
        return elapsed

    def _load_quantized(self, kwargs: dict[str, Any]) -> Any:
        """Load a torchao-quantized checkpoint.

        `torchao`'s `quantize_()` swaps Linear weights for tensor subclasses, so
        the saved file is a state dict over a structure that does not exist yet
        at load time. It cannot be rebuilt by `from_pretrained` alone: the base
        model has to be materialised first, quantized to create the same module
        shapes, and only then can the weights be read into it. That is why a
        quantized directory holds `quantized_state.pt` rather than the
        `model.safetensors` `from_pretrained` looks for.
        """
        # Checked before the imports and before the base model is materialised:
        # both cost real time (tens of seconds of weights onto the GPU), and
        # neither can turn a bad scheme or an absent checkpoint into a good one.
        if self.args.quantize != "int8":
            raise ValueError(
                f"unsupported quantization {self.args.quantize!r}; expected 'int8'"
            )
        state_path = Path(self.args.quantized_state).expanduser()
        if not state_path.exists():
            # Emphatically not a warning. Continuing here would serve the base
            # model's voice under the fine-tune's name — audio that sounds
            # plausible and is simply the wrong speaker, which is far harder to
            # notice than a backend that refuses to start.
            raise FileNotFoundError(
                f"quantized weights not found at {state_path}. The model is "
                "registered as int8, so refusing to start on the base weights."
            )

        import torch
        from omnivoice import OmniVoice
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        base = self.args.base_model or self.args.model
        log.info("loading base model %s to rebuild the int8 structure", base)
        model = OmniVoice.from_pretrained(base, **kwargs)

        # Same two submodules the quantization script targeted; quantizing a
        # different set would produce different keys and load nothing.
        quantize_(model.llm, Int8WeightOnlyConfig())
        quantize_(model.audio_heads, Int8WeightOnlyConfig())

        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)

        # `strict=False` is required (the audio tokenizer and feature extractor
        # are not in this file), but it also means a checkpoint that matches
        # nothing loads "successfully" and leaves the base weights in place. So
        # check that it actually applied rather than trusting the call.
        applied = len(state) - len(unexpected)
        if applied == 0:
            raise RuntimeError(
                f"{state_path} shares no parameter names with the model — "
                "nothing was loaded. Check that it was quantized from this "
                f"same base model ({base})."
            )
        log.info(
            "loaded %d/%d quantized tensors (%d unexpected, %d left at base)",
            applied, len(state), len(unexpected), len(missing),
        )
        if unexpected:
            log.warning(
                "%d tensor(s) in the checkpoint had no home in the model, e.g. %s",
                len(unexpected), ", ".join(sorted(unexpected)[:3]),
            )
        return model

    def _ensure_asr(self) -> None:
        """Load Whisper, needed only to transcribe a reference clip for us.

        Deferred because it is a second model on the GPU that is pure waste for
        the common case: a voice registered with its `ref_text` never needs it,
        and `arc-llama audio voice add` asks for that text up front.
        """
        if self._asr_loaded:
            return
        self.model.load_asr_model()
        self._asr_loaded = True

    def prompt_for(self, name: str, definition: dict[str, Any]) -> Any | None:
        """Build (or fetch) the encoded clone prompt for one voice.

        Returns None for a voice that has no reference to encode — a designed
        or auto voice — which is the common case and must not cost anything.
        """
        cached = self.voices.cached_prompt(name, definition)
        if cached is not None:
            return cached

        prompt_file = str(definition.get("prompt_file", "") or "")
        ref_audio = str(definition.get("ref_audio", "") or "")
        if not prompt_file and not ref_audio:
            return None

        from omnivoice import VoiceClonePrompt

        # A saved prompt is the encoded reference, so loading one skips both the
        # audio decode and any Whisper pass. Treat a stale/corrupt file as a
        # cache miss and re-encode rather than failing the request.
        if prompt_file and Path(prompt_file).expanduser().exists():
            try:
                prompt = VoiceClonePrompt.load(str(Path(prompt_file).expanduser()))
                self.voices.store_prompt(name, definition, prompt)
                return prompt
            except Exception:
                log.warning("ignoring unreadable voice prompt %s", prompt_file, exc_info=True)

        if not ref_audio:
            return None
        ref_path = Path(ref_audio).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(f"voice {name!r}: reference audio not found at {ref_path}")
        ref_text = str(definition.get("ref_text", "") or "") or None
        if ref_text is None:
            self._ensure_asr()
        prompt = self.model.create_voice_clone_prompt(str(ref_path), ref_text=ref_text)

        if prompt_file:
            # Persist so the next cold start of this backend skips the encode.
            try:
                target = Path(prompt_file).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                prompt.save(str(target))
            except Exception:
                log.warning("could not cache voice prompt to %s", prompt_file, exc_info=True)
        self.voices.store_prompt(name, definition, prompt)
        return prompt

    def generate_kwargs(self, body: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        """Translate one OpenAI speech body into generate() kwargs.

        Split out of ``synthesize`` so `--bench` measures the same call the
        HTTP path makes — a benchmark that skipped the voice resolution would
        be measuring a different model than the one serving requests.

        Returns (kwargs, response format, resolved voice name).
        """
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise BadRequestError("'input' must be a non-empty string")
        if len(text) > MAX_INPUT_CHARS:
            raise BadRequestError(f"'input' must be at most {MAX_INPUT_CHARS} characters")

        fmt = str(body.get("response_format") or self.args.default_response_format).lower()
        if fmt not in RESPONSE_FORMATS:
            raise BadRequestError(
                f"unsupported response_format {fmt!r}; expected one of {', '.join(RESPONSE_FORMATS)}"
            )

        kwargs: dict[str, Any] = {"text": text}
        # `instructions` is OpenAI's per-request style field and maps exactly
        # onto OmniVoice's voice-design `instruct`, so a request carrying one
        # overrides whatever style the registered voice implies.
        instruct = str(body.get("instructions") or "")
        language = str(body.get("language") or self.args.default_language or "")

        resolved = self.voices.lookup(str(body.get("voice") or ""))
        voice_name = ""
        if resolved is not None:
            voice_name, definition = resolved
            prompt = self.prompt_for(voice_name, definition)
            if prompt is not None:
                kwargs["voice_clone_prompt"] = prompt
            if not instruct:
                instruct = str(definition.get("instruct", "") or "")
            if not language:
                language = str(definition.get("language", "") or "")
        # Voice cloning wins when both are present: the reference audio already
        # fixes the speaker, and OmniVoice's own guidance is that cloning is the
        # stable mode. Design attributes only apply when there is no clone.
        if instruct and "voice_clone_prompt" not in kwargs:
            kwargs["instruct"] = instruct
        if language:
            kwargs["language"] = language

        speed = body.get("speed")
        if speed is not None:
            try:
                speed = float(speed)
            except (TypeError, ValueError):
                raise BadRequestError("'speed' must be a number") from None
            # OpenAI's documented range. OmniVoice accepts more, but a value
            # outside this band is nearly always a client bug.
            if not 0.25 <= speed <= 4.0:
                raise BadRequestError("'speed' must be between 0.25 and 4.0")
            kwargs["speed"] = speed

        for key, caster in (
            ("num_step", int),
            ("guidance_scale", float),
            ("duration", float),
            ("t_shift", float),
            ("class_temperature", float),
        ):
            if body.get(key) is not None:
                try:
                    kwargs[key] = caster(body[key])
                except (TypeError, ValueError):
                    raise BadRequestError(f"'{key}' must be a number") from None
        kwargs.setdefault("num_step", self.args.num_step)
        if self.args.normalize_text:
            kwargs.setdefault("normalize_text", True)
        return kwargs, fmt, voice_name

    def synthesize(self, body: dict[str, Any]) -> tuple[bytes, str]:
        """Turn one OpenAI speech request into encoded audio bytes."""
        kwargs, fmt, voice_name = self.generate_kwargs(body)
        text = str(kwargs.get("text", ""))

        log.info(
            "speech: %d chars, voice=%s, format=%s", len(text), voice_name or "(auto)", fmt
        )
        started = time.perf_counter()
        with self.gpu_lock:
            with _inference_context():
                audios = self.model.generate(**kwargs)
            # Inside the lock: the sync has to cover this request's work only,
            # or a second request's queue time is charged to this one.
            _device_sync(self.args.device)
        generated = time.perf_counter()
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        encoded = encode_audio(audios[0], self.sampling_rate, fmt)
        finished = time.perf_counter()

        # The line that answers "why does this feel slow?". A real-time factor
        # above 1 means synthesis is slower than playback, which is the whole
        # difference between a snappy assistant and one that pauses; and the
        # split says whether to reach for num_step or for the encoder.
        audio_s = _audio_seconds(audios[0], self.sampling_rate)
        gen_s = generated - started
        log.info(
            "speech done: %.2f s audio in %.2f s (RTF %.2fx) — generate %.2f s, "
            "encode %.2f s, num_step=%s",
            audio_s, finished - started, (gen_s / audio_s) if audio_s else 0.0,
            gen_s, finished - generated, kwargs.get("num_step"),
        )
        return encoded


class BadRequestError(Exception):
    """A client error worth reporting verbatim; anything else is a 500."""


def _to_int16(samples: Any) -> bytes:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    # OmniVoice can overshoot 1.0 slightly; clipping first keeps that from
    # wrapping around into loud noise when it becomes int16.
    clipped = np.clip(array, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _wav_bytes(samples: Any, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(_to_int16(samples))
    return buf.getvalue()


# soundfile format/subtype per requested container, for the formats libsndfile
# can write. mp3 and aac depend on the libsndfile build, so both stay in the
# table and simply fall through to ffmpeg when the write raises.
_SOUNDFILE_FORMATS = {
    "flac": ("FLAC", "PCM_16"),
    "opus": ("OGG", "OPUS"),
    "mp3": ("MP3", None),
}

_FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-b:a", "128k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac"],
}

_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


def encode_audio(samples: Any, rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode float samples into *fmt*, returning (bytes, media type).

    `wav` and `pcm` are written from the standard library so the two formats
    every client can decode never depend on an optional codec. The compressed
    formats try libsndfile first (already present, since OmniVoice uses it) and
    fall back to ffmpeg, because libsndfile's mp3/aac support is a build option
    that is off in many wheels — and mp3 is what OpenAI clients ask for by
    default, so failing there would break the common case.
    """
    if fmt == "pcm":
        return _to_int16(samples), _MEDIA_TYPES["pcm"]
    wav = _wav_bytes(samples, rate)
    if fmt == "wav":
        return wav, _MEDIA_TYPES["wav"]

    errors: list[str] = []
    spec = _SOUNDFILE_FORMATS.get(fmt)
    if spec is not None:
        try:
            import soundfile as sf

            container, subtype = spec
            buf = io.BytesIO()
            sf.write(buf, _sf_array(samples), rate, format=container, subtype=subtype)
            return buf.getvalue(), _MEDIA_TYPES[fmt]
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and fmt in _FFMPEG_ARGS:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                 *_FFMPEG_ARGS[fmt], "pipe:1"],
                input=wav, capture_output=True, check=True, timeout=120,
            )
            return proc.stdout, _MEDIA_TYPES[fmt]
        except subprocess.SubprocessError as exc:
            stderr = getattr(exc, "stderr", b"") or b""
            errors.append(f"ffmpeg: {stderr.decode('utf-8', 'replace').strip() or exc}")
    elif not ffmpeg:
        errors.append("ffmpeg: not installed")

    raise BadRequestError(
        f"cannot encode {fmt!r} in this environment ({'; '.join(errors)}). "
        "Install ffmpeg, or ask for response_format=wav."
    )


def _sf_array(samples: Any):
    import numpy as np

    return np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """A server that reports a dropped connection as one line, not a traceback.

    ``socketserver`` prints a full traceback to stderr whenever a client goes
    away mid-request. The router health-checks every 1.5 s with a 2 s timeout
    while the model loads, and every one of those timeouts is a reset
    connection — so a cold start that legitimately takes minutes would write
    hundreds of tracebacks into the backend log and bury the single message
    that explains a real failure. arc-llama shows the tail of this log when a
    backend fails to start, which is exactly when that matters.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            log.debug("client %s disconnected: %s", client_address, exc)
            return
        log.exception("error handling request from %s", client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine  # set on the server class before serving

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N803
        log.debug("%s - %s", self.address_string(), fmt % args)


    def _send(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            log.debug("client disconnected before the response was written")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            # The parent gates "safe to forward to" on this exact shape, so it
            # must stay 503 until the weights are actually resident — a 200 at
            # bind time would put the whole model load inside a user's first
            # request, past the timeout the router is enforcing.
            if self.engine.ready:
                self._send_json(200, {"status": "ok"})
            elif self.engine.load_error:
                self._send_json(500, {"status": "error", "error": self.engine.load_error})
            else:
                self._send_json(503, {"status": "loading"})
            return
        self._error(404, f"unknown path {path!r}")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/audio/speech":
            self._error(404, f"unknown path {path!r}")
            return
        if not self.engine.ready:
            self._error(503, self.engine.load_error or "model is still loading")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            self._error(400, f"invalid JSON: {exc}")
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        try:
            audio, media_type = self.engine.synthesize(body)
        except BadRequestError as exc:
            self._error(400, str(exc))
            return
        except FileNotFoundError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:
            log.exception("speech synthesis failed")
            self._error(500, f"speech synthesis failed: {exc}")
            return
        self._send(200, audio, media_type)


def run_bench(engine: Engine, args: argparse.Namespace, startup_s: float = 0.0) -> int:
    """Sweep `num_step` over one utterance and print a real-time-factor table.

    Exists because the settings that matter here — how many solver steps are
    enough, whether an int8 checkpoint is actually faster than bf16, whether
    `--compile` repays its warmup — are all properties of one machine's GPU and
    driver stack, and cannot be answered by reading the code. Run it on the
    box that serves, against the voice that serves.
    """
    steps = [int(s) for s in str(args.bench_steps).split(",") if s.strip()]
    body: dict[str, Any] = {"input": args.bench, "voice": args.bench_voice}
    base_kwargs, _fmt, voice_name = engine.generate_kwargs(body)

    if args.compile and len(steps) > 1:
        # Dynamo guards on `num_step` because it is an ordinary Python int
        # driving the solver loop, so each value in the sweep is a different
        # graph and pays a full compile. At the ~10 minutes an XPU compile
        # takes, a four-value sweep is most of an hour before the first row.
        print(
            f"WARNING: sweeping {len(steps)} step values with --compile means "
            f"{len(steps)} separate compiles.\n"
            "         num_step is a Python int the tracer specialises on, so "
            "changing it invalidates\n"
            "         the graph. Measure eager first (--no-compile), or pin one "
            "value (--bench-steps 32)\n"
            "         to time the compiled path.\n"
        )

    print(f"model      : {args.model}")
    print(f"device     : {args.device}  dtype={args.dtype}  quantize={args.quantize or 'none'}")
    print(f"compile    : {args.compile_targets if args.compile else 'off'}")
    print(f"voice      : {voice_name or '(auto)'}")
    print(f"text       : {args.bench!r}")
    print(f"runs       : {args.bench_runs} timed (plus one discarded warmup)\n")
    print(f"{'num_step':>9}  {'audio s':>8}  {'best s':>8}  {'mean s':>8}  {'RTF':>6}")

    last_audio: Any = None
    before = dynamo_stats() if args.compile else {}
    for step in steps:
        kwargs = dict(base_kwargs, num_step=step)
        times: list[float] = []
        audio_s = 0.0
        # The first run of each configuration is discarded: it carries the
        # shape-specialised compile and any first-touch allocation, neither of
        # which a warm server pays per request.
        for run in range(args.bench_runs + 1):
            started = time.perf_counter()
            try:
                with _inference_context():
                    audios = engine.model.generate(**kwargs)
                _device_sync(args.device)
            except Exception as exc:
                print(f"{step:>9}  failed: {type(exc).__name__}: {exc}")
                break
            elapsed = time.perf_counter() - started
            if run:
                times.append(elapsed)
                audio_s = _audio_seconds(audios[0], engine.sampling_rate)
                last_audio = audios[0]
        if not times:
            continue
        best, mean = min(times), sum(times) / len(times)
        rtf = (best / audio_s) if audio_s else 0.0
        print(f"{step:>9}  {audio_s:>8.2f}  {best:>8.2f}  {mean:>8.2f}  {rtf:>6.2f}")

        if args.compile:
            # A compiled module that is *slower* than eager is nearly always
            # re-tracing rather than running a bad kernel. The decoder is
            # called once per solver step, so a value that changes each step —
            # the flow-matching timestep, as a Python float — makes every call
            # a fresh graph, and Dynamo's tracing lands in the measurement.
            # Compiles growing with the step count is that, visibly.
            after = dynamo_stats()
            grew = {
                key: after[key] - before.get(key, 0)
                for key in after
                if after[key] - before.get(key, 0) > 0
            }
            if grew:
                summary = ", ".join(f"{k}={v}" for k, v in sorted(grew.items())[:4])
                print(f"{'':>9}  dynamo during this step: {summary}")
            before = after

    # The rows above time generation only, but a client waits for encoded
    # bytes. mp3 is OpenAI's default and may shell out to ffmpeg per request,
    # which is a process spawn no amount of num_step tuning will recover.
    if last_audio is not None:
        started = time.perf_counter()
        try:
            payload, _media = encode_audio(last_audio, engine.sampling_rate, _fmt)
            encode_s = time.perf_counter() - started
            # Only nudge toward wav when the encode is actually worth
            # avoiding. libsndfile encodes mp3 in-process in tens of
            # milliseconds; it is the ffmpeg fallback that costs a process
            # spawn. Suggesting wav against a 17 ms encode trades a smaller
            # payload — which matters to a wifi speaker — for nothing.
            hint = ""
            if _fmt not in ("wav", "pcm") and encode_s >= _SLOW_ENCODE_S:
                hint = "  — slow enough to matter; try response_format=wav"
            print(
                f"\nencode to {_fmt}: {encode_s:.3f} s for {len(payload) / 1024:.0f} KiB{hint}"
            )
        except Exception as exc:
            print(f"\nencode to {_fmt} failed: {type(exc).__name__}: {exc}")

    print(
        "\nRTF is generate-time / audio-length: below 1.0 is faster than real time.\n"
        "Pick the lowest num_step that still sounds right — quality falls off a\n"
        "cliff rather than degrading smoothly, so listen, don't just read."
    )
    if args.compile:
        print(
            "\nCompiled and slower than eager? The decoder runs once per solver\n"
            "step, so anything the tracer specialises on that changes per step —\n"
            "the flow-matching timestep as a plain Python float is the usual one —\n"
            "makes every call a new graph, and the re-tracing lands in the\n"
            "measurement. `TORCH_LOGS=recompiles` names the guard that failed."
        )
    if startup_s:
        # Said explicitly because the wall-clock of this command is dominated
        # by startup, and startup is paid once per server, not per utterance.
        # Reading the total as "how slow synthesis is" is the wrong conclusion.
        print(
            f"\nStartup (loading, and any compile) took {startup_s:.0f} s and is not in\n"
            "the table above: a served backend pays it once, at launch, not per\n"
            "request. The table is the steady state Home Assistant would see."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="HF repo id or local model directory.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", default="xpu", help="torch device_map, e.g. xpu, cuda:0, cpu.")
    p.add_argument("--dtype", default="float16", help="torch dtype name.")
    p.add_argument("--voices", default="", help="Path to the voices JSON written by arc-llama.")
    p.add_argument("--num-step", type=int, default=32, help="Default diffusion steps.")
    p.add_argument("--default-language", default="", help="Language used when none is given.")
    p.add_argument(
        "--default-response-format", default="mp3",
        help="Format used when the request omits response_format.",
    )
    p.add_argument("--asr-model", default="", help="Whisper model for reference auto-transcription.")
    p.add_argument("--asr-device", default="", help="Device for the Whisper model.")
    p.add_argument(
        "--normalize-text", action="store_true",
        help="Expand numbers and dates to their spoken form before synthesis.",
    )
    p.add_argument(
        "--quantize", default="",
        help="Quantization scheme of the checkpoint ('int8'). Empty for an "
        "ordinary unquantized model.",
    )
    p.add_argument(
        "--quantized-state", default="",
        help="Path to the quantized state dict (quantized_state.pt).",
    )
    p.add_argument(
        "--base-model", default="",
        help="Model the quantized checkpoint was derived from, whose structure "
        "is rebuilt before the weights are read in.",
    )
    p.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False,
        help="torch.compile the hot submodules after loading. Needed to get "
        "torchao's int8 kernels fused; without it a quantized model pays "
        "dequantization for nothing. `--no-compile` later in the command line "
        "wins, which is how the bench overrides a model's configured default.",
    )
    p.add_argument(
        "--compile-dynamic", default="auto", choices=("auto", "true", "false"),
        help="How compiled shapes are treated. `auto` (torch's default) "
        "specialises on the first shape and re-traces as dynamic once a second "
        "appears; `true` forces symbolic shapes from the start, which makes "
        "Inductor log 'Constructing input/output tensor meta failed for Extern "
        "Choice' because a symbolic size cannot be turned into the concrete one "
        "a library-kernel benchmark needs.",
    )
    p.add_argument(
        "--compile-targets", default=DEFAULT_COMPILE_TARGETS,
        help="Comma-separated submodules to compile. The top-level model is "
        "deliberately not one of them: generate() is not forward(), so "
        "compiling the wrapper has no effect.",
    )
    p.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="Run one throwaway synthesis before reporting healthy, so the "
        "first real request does not pay for lazy kernel init.",
    )
    p.add_argument(
        "--warmup-text", default="Warming up.",
        help="Text used for the warmup synthesis.",
    )
    p.add_argument(
        "--bench", default="",
        help="Benchmark this text instead of serving, then exit.",
    )
    p.add_argument(
        "--bench-steps", default="8,16,24,32",
        help="Comma-separated num_step values to sweep in --bench.",
    )
    p.add_argument("--bench-runs", type=int, default=3, help="Timed runs per --bench step.")
    p.add_argument("--bench-voice", default="", help="Voice to use for --bench.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = os.environ.get("ARC_LLAMA_TTS_LOG", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if level != "DEBUG":
        quiet_inductor_noise()
    engine = Engine(args, VoiceBook(args.voices or None))

    if args.bench:
        # Warm up at the first value the sweep will use, not the configured
        # default. The warmup is what pays for the compile, and a compile of
        # num_step=32 is worth nothing to a sweep that starts at 8 — it just
        # buys a second full compile on the first measured run.
        first_step = next(
            (int(s) for s in str(args.bench_steps).split(",") if s.strip()), args.num_step
        )
        args.num_step = first_step
        # Load in the foreground: there is no health endpoint to answer and
        # nothing to measure until the weights are resident.
        started = time.perf_counter()
        engine.load()
        return run_bench(engine, args, startup_s=time.perf_counter() - started)

    class BoundHandler(Handler):
        pass

    BoundHandler.engine = engine
    httpd = QuietThreadingHTTPServer((args.host, args.port), BoundHandler)

    # Bind before loading so /health can answer 503 "loading" instead of
    # refusing the connection: the router distinguishes a backend that is slow
    # from one that never came up, and only the former is worth waiting on.
    def _load() -> None:
        try:
            engine.load()
        except Exception as exc:
            log.exception("model load failed")
            engine.load_error = f"{type(exc).__name__}: {exc}"

    loader = threading.Thread(target=_load, name="omnivoice-load", daemon=True)
    loader.start()

    log.info("listening on http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
