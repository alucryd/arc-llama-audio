"""Configuration for the audio plugin.

Kept in its own file, ``audio.toml`` beside arc-llama's ``config.toml``,
rather than as extra tables in the core config. The core builds its ``Config``
with a field filter that drops unknown top-level keys and logs a warning for
each, so ``[[audio_models]]`` living there would be discarded *and* noisy. A
plugin owning its own file also means its schema can move without waiting on
a core release.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib

    _toml_load = tomllib.load
else:
    import tomli

    _toml_load = tomli.load

log = logging.getLogger("arc_llama_audio.config")

AUDIO_ENGINE_LLAMACPP = "llamacpp"

ASR_ENGINES = (AUDIO_ENGINE_LLAMACPP,)
"""Runtimes that can serve `/v1/audio/transcriptions`.

Only llama-server. It is the binary an Arc box already has, it is the only one
with a SYCL build, and it inherits the arch env profiles and the device
selector. TTS engines are not listed here: they are discovered from
``arc_llama.tts``, which config cannot import without a cycle and does not need
to — nothing in this module dispatches on an engine name.
"""

DEFAULT_ASR_CTX = 4096
"""Context length for a transcription model when the recipe doesn't say.

This must be set explicitly, because llama-server's `-c` default is `0`,
meaning "whatever the GGUF was trained for" — and Qwen3-ASR advertises 65536.
That sizes a ~7 GB KV cache for a 1.7B model whose weights are under 2 GB,
which looks like a runaway leak and is really just an unasked-for context.
A single utterance is a few hundred audio tokens plus its transcript, so 4096
covers minutes of speech; raise it in the recipe for long-form dictation.
"""


_AUDIO_RECIPE_KEYS = (
    "mmproj",
    "ctx",
    "n_gpu_layers",
    "cache_type_k",
    "cache_type_v",
    "threads",
    "extra_flags",
    "python",
    "device",
    "dtype",
    "default_voice",
    "default_language",
    "default_response_format",
    "options",
)



@dataclass
class AudioRecipe:
    """Resolved launch knobs for one audio model.

    Engine-specific fields are inert for the other engine rather than an
    error: swapping `engine` on an existing entry should not mean rewriting
    the recipe from scratch.
    """

    mmproj: str = ""
    ctx: int = DEFAULT_ASR_CTX
    n_gpu_layers: int = 999
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    threads: int = 1
    extra_flags: list[str] = field(default_factory=list)

    # -- TTS --------------------------------------------------------
    python: str = ""
    """Interpreter for a Python TTS backend, overriding `paths.tts_python`."""
    device: str = ""
    """Compute device as the engine names it (`xpu`, `cuda:0`, `cpu`).

    Empty lets the engine choose from the GPU's configured backend, which is
    `xpu` for the SYCL cards this project exists for.
    """
    dtype: str = ""
    """Weight dtype for a torch-based engine (e.g. `float16`, `bfloat16`).

    Empty lets the engine pick, which it needs to do: a quantized checkpoint
    dictates the dtype of the base model it was derived from, and getting that
    wrong is a dtype mismatch on the first matmul rather than a slow path.
    """
    default_voice: str = ""
    """Voice used when a request's `voice` field matches nothing registered."""
    default_language: str = ""
    """Language used when a request does not say (e.g. `English`)."""
    default_response_format: str = "mp3"
    """Encoding used when a request omits `response_format`, matching OpenAI."""
    options: dict[str, Any] = field(default_factory=dict)
    """Engine-specific knobs, passed through untouched.

    Anything only one engine understands lives here rather than becoming a
    field: OmniVoice's `num_step` and `normalize_text` mean nothing to a
    `llama-tts` backend, and a shared dataclass that grows a field per engine
    stops being a shared dataclass. Adding an engine should not need an edit
    to this file.
    """


@dataclass
class AudioModelConfig:
    """One audio model, served by its own backend subprocess.

    Covers both directions, because they share everything except the endpoint
    they answer on: the same registry, ports, GPU binding, launch/health/evict
    lifecycle and VRAM accounting.

      * **`task = "asr"`** — speech to text on `/v1/audio/transcriptions`,
        always `engine = "llamacpp"`: `llama-server -m model.gguf --mmproj
        proj.gguf`. That is the binary an Arc box already has, it is the only
        transcription runtime with a **SYCL** build, and it inherits the arch
        env profiles and the device selector.
      * **`task = "tts"`** — text to speech on `/v1/audio/speech`, served by
        whichever engine is named in ``engine``. Those come from
        ``arc_llama.tts``, which owns both how the backend is launched and how
        an OpenAI request is translated for it; `omnivoice` is the one shipped
        today.

    Deliberately not a ``ModelConfig``: the tuner's whole surface (KV-cache
    sweeps, ctx-vs-VRAM recipes, benchmark prompts) is meaningless for a
    transcription model, and giving audio models fields the tuner reads would
    let a sweep pick up something it cannot benchmark. Keeping them in their
    own table also stops `arc-llama scan` from ever treating one as an LLM.
    """

    name: str  # short id, also URL-safe (e.g. "qwen3-asr")
    path: str  # model directory or .gguf file
    port: int  # backend port for this model's backend process
    gpu_pci_slot: str  # which detected GPU to bind to
    engine: str = "llamacpp"
    """Which runtime serves this model.

    `llamacpp` for `task = "asr"`; for `task = "tts"` it is the name of a
    registered TTS engine (see ``arc_llama.tts``), e.g. `omnivoice`.
    """
    task: str = "asr"
    """`asr` or `tts`. Decides which OpenAI endpoint routes here."""
    mode: str = "offline"
    """`offline` or `streaming`. Streaming is required for `stream=true`
    transcriptions, and only for models whose backend can produce incremental
    deltas."""
    recipe: dict[str, Any] = field(default_factory=dict)
    """How to launch this model, in the same place `[models.recipe]` keeps it.

    Everything that shapes the process goes here — `mmproj`, `ctx`,
    `n_gpu_layers`, `cache_type_k`/`cache_type_v`, `extra_flags` for ASR;
    `python`, `device`, `dtype`, the `default_*` request fallbacks and an
    `options` bag for TTS. The body above stays identity and routing policy,
    so the two model tables read the same way. See ``audio_recipe`` for the
    defaults.
    """
    strip_asr_markers: bool = True
    """Strip Qwen3-ASR's native output framing from the transcript.

    Qwen3-ASR emits `language English<asr_text>the actual words`, and
    llama.cpp forwards it verbatim (ggml-org/llama.cpp#26749, still open).
    A Home Assistant voice pipeline then tries to match that prefix as part
    of the command and fails. Stripping is on by default and is a no-op for
    any model that does not emit the marker.
    """
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    """Extra strings that should match this model in the OpenAI `model` field.
    Register `whisper-1` here if a client hardcodes OpenAI's STT model id."""
    # Speech backends are always resident: the plugin owns them, and the core
    # router only ever evicts models from its own registry. That is what you
    # want anyway — an ASR model is well under a gigabyte and is used in short
    # bursts between LLM turns, so evicting it would make each utterance cost
    # two cold starts. The cost is that the core's VRAM fit guard cannot see
    # this footprint when admitting an LLM; leave headroom for it.

    def audio_recipe(self) -> AudioRecipe:
        """The launch recipe with defaults filled in."""
        r = self.recipe or {}
        return AudioRecipe(
            mmproj=str(r.get("mmproj", "")),
            ctx=int(r.get("ctx", DEFAULT_ASR_CTX)),
            n_gpu_layers=int(r.get("n_gpu_layers", 999)),
            cache_type_k=str(r.get("cache_type_k", "f16")),
            cache_type_v=str(r.get("cache_type_v", "f16")),
            threads=int(r.get("threads", 1)),
            extra_flags=list(r.get("extra_flags", [])),
            python=str(r.get("python", "")),
            device=str(r.get("device", "")),
            dtype=str(r.get("dtype", "")),
            default_voice=str(r.get("default_voice", "")),
            default_language=str(r.get("default_language", "")),
            default_response_format=str(r.get("default_response_format", "mp3")),
            options=dict(r.get("options", {})),
        )


@dataclass
class VoiceConfig:
    """A named voice, resolvable from the OpenAI `voice` request field.

    Kept in its own top-level table rather than nested under a model, because a
    voice is a property of the speaker and not of the runtime: the same
    reference clip should still name the same voice after switching engines,
    and a client that says `voice = "glados"` should not have to know which
    backend is loaded. ``models`` narrows it when that is not true.

    Which fields are set decides the synthesis mode. `ref_audio` (with or
    without `ref_text`) clones; `instruct` alone designs a voice from
    attributes; neither lets the model pick one itself.
    """

    name: str
    ref_audio: str = ""
    """Reference clip to clone, 3–10 s of clean speech."""
    ref_text: str = ""
    """Transcript of `ref_audio`. Empty makes the engine transcribe it with
    Whisper on first use, which costs a second model on the GPU — so supplying
    it is worth the typing."""
    instruct: str = ""
    """Voice-design attributes, e.g. `female, low pitch, british accent`.
    Ignored when `ref_audio` is set: cloning already fixes the speaker."""
    language: str = ""
    """Language this voice is meant to speak, e.g. `English`."""
    prompt_file: str = ""
    """Where the engine caches the encoded reference.

    Encoding a reference clip is not free and its result never changes, so the
    first use writes it here and later starts load it back instead of decoding
    (and possibly re-transcribing) the audio again. Empty means the engine
    picks a path under the state dir.
    """
    models: list[str] = field(default_factory=list)
    """TTS model names this voice applies to. Empty means all of them."""
    display_name: str = ""
    aliases: list[str] = field(default_factory=list)
    """Extra strings that resolve to this voice. Register `alloy` here for
    clients that hardcode one of OpenAI's voice ids."""




@dataclass
class AudioConfig:
    """Everything in ``audio.toml``."""

    version: int = 1
    tts_python: str = ""
    """Interpreter that runs a TTS sidecar, when the model's recipe does not
    name one. Empty falls back to the interpreter running arc-llama, which is
    right only when they share an environment."""
    audio_models: list[AudioModelConfig] = field(default_factory=list)
    voices: list[VoiceConfig] = field(default_factory=list)

    # -- lookup ------------------------------------------------------

    def find_model(self, query: str, task: str = "") -> AudioModelConfig | None:
        """Resolve a client's `model` field, optionally narrowed to a task.

        Exact name, then exact alias, then case-insensitive substring — the
        same order the core uses for LLMs, so an id that works on
        /v1/chat/completions needs no different spelling here. With no match
        and exactly one model registered for the task, that one wins: clients
        hardcode `whisper-1` or `tts-1`, and a single-model box has no
        ambiguity worth failing over.
        """
        pool = [m for m in self.audio_models if not task or m.task == task]
        if query:
            for m in pool:
                if m.name == query:
                    return m
            for m in pool:
                if query in m.aliases:
                    return m
            ql = query.lower()
            for m in pool:
                haystacks = [m.name.lower(), m.display_name.lower(), *(a.lower() for a in m.aliases)]
                if any(ql in h for h in haystacks):
                    return m
        return pool[0] if len(pool) == 1 else None

    def voices_for(self, model_name: str) -> list[VoiceConfig]:
        """Voices that apply to *model_name* — unrestricted ones plus its own."""
        return [v for v in self.voices if not v.models or model_name in v.models]

    def find_voice(self, name: str) -> VoiceConfig | None:
        for v in self.voices:
            if v.name == name or name in v.aliases:
                return v
        return None

    # -- persistence -------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Write atomically: a truncated TOML file still parses, and would
        silently come back with no models registered."""
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "tts_python": self.tts_python,
            "audio_models": [_strip_none(asdict(m)) for m in self.audio_models],
            "voices": [_strip_none(asdict(v)) for v in self.voices],
        }
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as f:
            tomli_w.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path


def _strip_none(obj: Any) -> Any:
    """TOML has no null, so None values would crash the writer."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys *cls* knows, so a newer file loads on an older plugin."""
    known = {f.name for f in fields(cls)}
    out = {}
    for k, v in raw.items():
        if k in known:
            out[k] = v
        else:
            log.warning("ignoring unknown key %r in %s", k, cls.__name__)
    return out


def _xdg_config_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def default_config_path() -> Path:
    """Beside arc-llama's own config, so one directory holds the whole setup."""
    return _xdg_config_home() / "arc-llama" / "audio.toml"


def load_config(path: Path | None = None) -> AudioConfig:
    path = path or default_config_path()
    if not path.exists():
        return AudioConfig()
    with open(path, "rb") as f:
        raw = _toml_load(f)
    # Launch knobs live under [audio_models.recipe], as they do for core
    # models. Lift any that were written flat by an early version.
    for entry in raw.get("audio_models", []):
        if isinstance(entry, dict):
            recipe = entry.setdefault("recipe", {})
            for key in _AUDIO_RECIPE_KEYS:
                if key in entry:
                    recipe.setdefault(key, entry.pop(key))
    return AudioConfig(
        version=int(raw.get("version", 1)),
        tts_python=str(raw.get("tts_python", "")),
        audio_models=[
            AudioModelConfig(**_filter_fields(AudioModelConfig, m))
            for m in raw.get("audio_models", [])
        ],
        voices=[VoiceConfig(**_filter_fields(VoiceConfig, v)) for v in raw.get("voices", [])],
    )
