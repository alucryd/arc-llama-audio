"""OmniVoice as an arc-llama TTS engine.

OmniVoice (k2-fsa) is a zero-shot multilingual TTS model shipped as a Python
library — a `model.generate(...)` call, no server and no binary. So this engine
launches :mod:`arc_llama.tts.omnivoice_server`, a small script that wraps one
loaded model in the `/v1/audio/speech` route, under whatever interpreter has
OmniVoice installed.

Running it as a subprocess rather than importing it here buys three things that
matter more than the extra hop: torch never enters arc-llama's environment,
stopping the model actually returns its VRAM (the router's existing evict path
just works), and a synthesis that wedges the GPU takes down a child process
instead of the router.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from arc_llama.arch import Arch, Backend, profile_for
from arc_llama.config import Config, GPUConfig
from arc_llama.launcher import LaunchPlan

from arc_llama_audio.config import AudioConfig, AudioModelConfig, VoiceConfig
from arc_llama_audio.tts.base import TTSEngine, register_engine

log = logging.getLogger("arc_llama.tts.omnivoice")

TTS_ENGINE_OMNIVOICE = "omnivoice"

DEFAULT_OMNIVOICE_REPO = "k2-fsa/OmniVoice"

SERVER_SCRIPT = Path(__file__).parent.parent / "sidecars" / "omnivoice_server.py"

QUANTIZED_STATE_NAME = "quantized_state.pt"
"""Filename torchao quantization scripts write the weights to.

`torchao`'s `quantize_()` replaces Linear weights with tensor subclasses, which
`save_pretrained` cannot serialise — so the established practice is a plain
`torch.save` of the state dict beside an otherwise ordinary model directory
(config, tokenizer, audio_tokenizer/, but no `model.safetensors`). Its presence
is what marks a directory as quantized.
"""


def quantized_state_path(model_path: str) -> Path | None:
    """The quantized weights inside *model_path*, or None if it is an ordinary model.

    Detection rather than configuration: a directory holding
    `quantized_state.pt` cannot be loaded by `from_pretrained` at all, so
    treating it as unquantized has no valid interpretation — there is nothing
    for the user to choose here.
    """
    if not model_path:
        return None
    # `is_dir()` is also what rules out a Hugging Face repo id, which is not a
    # path on this machine and so can never be a quantized directory.
    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        return None
    candidate = directory / QUANTIZED_STATE_NAME
    return candidate if candidate.is_file() else None


def tts_state_dir(cfg: Config) -> Path:
    """Where generated voice tables and cached voice prompts live."""
    base = Path(cfg.paths.state_dir).expanduser() if cfg.paths.state_dir else Path(".")
    return base / "tts"


def resolve_python(audio: AudioConfig, model: AudioModelConfig) -> str:
    """The interpreter that runs the sidecar, most specific first.

    Falls back to the one running arc-llama, which is right only when the two
    share an environment — common in a container image built for both, wrong
    for the usual `pip install arc-llama` plus a separate OmniVoice checkout.
    The failure is loud either way: the child exits with ModuleNotFoundError
    and the router's log-tail hint says what to set.
    """
    recipe = model.audio_recipe()
    for candidate in (recipe.python, audio.tts_python):
        if candidate:
            return str(Path(candidate).expanduser())
    return sys.executable


def _hf_cache_dir(repo_id: str) -> Path | None:
    """Local cache directory for an HF repo id, if it has been downloaded.

    Resolves to the *snapshot* for the checked-out revision rather than the
    repo root, because only that revision is ever loaded. A cache holding two
    revisions has two sets of blobs, and measuring the root would charge the
    fit guard for weights that will never reach the GPU.
    """
    # `org/name` and nothing else. The obvious spelling of this test —
    # "contains os.sep, so it is a path" — is silently always true on POSIX,
    # where os.sep *is* the separator a repo id uses, so it rejected every repo
    # id and the estimate came back unknown for exactly the models it exists to
    # measure.
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts) or repo_id[0] in "./~\\":
        return None
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        base = Path(hub).expanduser()
    else:
        home = os.environ.get("HF_HOME")
        root = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
        base = root / "hub"
    candidate = base / ("models--" + repo_id.replace("/", "--"))
    if not candidate.is_dir():
        return None
    snapshot = _current_snapshot(candidate)
    return snapshot if snapshot is not None else candidate


def _current_snapshot(repo_dir: Path) -> Path | None:
    """The snapshot directory `refs/main` points at, if it can be resolved."""
    try:
        revision = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        # No refs (a manually assembled cache, or a revision pinned by hash).
        # A single snapshot is unambiguous; more than one is not, so fall back
        # to the repo root and let the inode de-duplication bound the damage.
        snapshots = sorted((repo_dir / "snapshots").glob("*")) if (
            repo_dir / "snapshots"
        ).is_dir() else []
        return snapshots[0] if len(snapshots) == 1 else None
    snapshot = repo_dir / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


class OmniVoiceEngine(TTSEngine):
    name = TTS_ENGINE_OMNIVOICE
    description = "k2-fsa OmniVoice — zero-shot multilingual TTS with voice cloning"
    speech_path = "/v1/audio/speech"
    accepts_remote_path = True

    # -- registration -------------------------------------------------

    def validate(self, model: AudioModelConfig) -> None:
        recipe = model.audio_recipe()
        if recipe.dtype and not recipe.dtype.replace("_", "").isalnum():
            raise ValueError(f"invalid dtype {recipe.dtype!r}")
        path = Path(model.path).expanduser()
        if not path.exists() and "/" not in model.path:
            raise FileNotFoundError(
                f"{model.path!r} is neither a local directory nor a Hugging Face "
                f"repo id. Use a path, or the repo id {DEFAULT_OMNIVOICE_REPO!r}."
            )

    # -- lifecycle ----------------------------------------------------

    def build_plan(
        self, cfg: Config, audio: AudioConfig, model: AudioModelConfig,
        gpu: GPUConfig, host: str = "127.0.0.1",
    ) -> LaunchPlan:
        recipe = model.audio_recipe()
        python = resolve_python(audio, model)
        if not SERVER_SCRIPT.exists():  # pragma: no cover - broken install
            raise RuntimeError(f"the OmniVoice sidecar is missing from {SERVER_SCRIPT}")

        device = recipe.device or default_device(gpu)
        voices_path = write_voices_file(cfg, audio, model)
        # torchao int8 checkpoints are produced from a bf16 base, and the
        # quantized tensors carry that as their compute dtype. Loading the base
        # as fp16 instead gives a dtype mismatch on the first matmul, so the
        # default follows the checkpoint rather than the engine.
        dtype = recipe.dtype or (
            "bfloat16" if quantized_state_path(model.path) is not None else "float16"
        )

        argv: list[str] = [
            python,
            str(SERVER_SCRIPT),
            "--model", str(Path(model.path).expanduser()) if Path(model.path).expanduser().exists()
            else model.path,
            "--host", host,
            "--port", str(model.port),
            "--device", device,
            "--dtype", dtype,
            "--voices", str(voices_path),
            "--default-response-format", recipe.default_response_format or "mp3",
        ]
        if recipe.default_language:
            argv += ["--default-language", recipe.default_language]

        options = recipe.options or {}
        if options.get("num_step"):
            argv += ["--num-step", str(int(options["num_step"]))]
        if options.get("asr_model"):
            argv += ["--asr-model", str(options["asr_model"])]
        if options.get("asr_device"):
            argv += ["--asr-device", str(options["asr_device"])]
        if options.get("normalize_text"):
            argv.append("--normalize-text")
        if options.get("compile"):
            argv.append("--compile")
            if options.get("compile_targets"):
                argv += ["--compile-targets", str(options["compile_targets"])]
            if options.get("compile_dynamic"):
                argv += ["--compile-dynamic", str(options["compile_dynamic"])]
        # Warmup is on by default in the sidecar: the first synthesis after a
        # load is far slower than the rest, and on a voice assistant that cost
        # lands on whoever speaks first. Opt out for a machine where the extra
        # startup time matters more than the first request's latency.
        if options.get("warmup") is False:
            argv.append("--no-warmup")
        if options.get("warmup_text"):
            argv += ["--warmup-text", str(options["warmup_text"])]

        state_path = quantized_state_path(model.path)
        if state_path is not None:
            # The quantized directory has no weights `from_pretrained` can read,
            # so the sidecar rebuilds the structure from the base model and then
            # loads these tensors into it.
            argv += [
                "--quantize", str(options.get("quantization", "int8")),
                "--quantized-state", str(state_path),
                "--base-model", str(options.get("base_model") or DEFAULT_OMNIVOICE_REPO),
            ]
        argv.extend(recipe.extra_flags)

        backend_url = f"http://{host}:{model.port}"
        return LaunchPlan(
            argv=argv,
            env=build_env_for(cfg, gpu, device),
            backend_url=backend_url,
            health_url=f"{backend_url}/health",
        )

    # -- requests -----------------------------------------------------

    def build_payload(self, model: AudioModelConfig, body: dict[str, Any]) -> dict[str, Any]:
        """Fill in the model's request defaults; the sidecar is OpenAI-native.

        The defaults are applied here rather than in the sidecar so they follow
        the config: editing `recipe.default_voice` takes effect on the next
        request instead of the next model reload.
        """
        recipe = model.audio_recipe()
        payload = dict(body)
        payload.pop("model", None)
        if not payload.get("voice") and recipe.default_voice:
            payload["voice"] = recipe.default_voice
        if not payload.get("response_format") and recipe.default_response_format:
            payload["response_format"] = recipe.default_response_format
        if not payload.get("language") and recipe.default_language:
            payload["language"] = recipe.default_language
        for key, value in (recipe.options or {}).items():
            # Per-request wins; these are the model's defaults for knobs the
            # sidecar accepts (num_step, guidance_scale, ...).
            if key in _REQUEST_OPTIONS and payload.get(key) is None:
                payload[key] = value
        return payload

    # -- diagnostics --------------------------------------------------

    def preflight(self, cfg: Config, audio: AudioConfig, model: AudioModelConfig) -> list[str]:
        problems: list[str] = []
        python = resolve_python(audio, model)
        resolved = shutil.which(python) if os.sep not in python else (
            python if Path(python).exists() else None
        )
        if resolved is None:
            problems.append(
                f"TTS interpreter not found: {python}. Set it with "
                "`arc-llama audio set-python /path/to/omnivoice/venv/bin/python`."
            )
            return problems
        if not _can_import(resolved, "omnivoice"):
            problems.append(
                f"{resolved} cannot import `omnivoice`. Install OmniVoice into "
                "that environment, or point `arc-llama audio set-python` at the "
                "virtualenv that has it."
            )
        recipe = model.audio_recipe()
        fmt = recipe.default_response_format or "mp3"
        if fmt not in ("wav", "pcm") and shutil.which("ffmpeg") is None:
            problems.append(
                f"default_response_format is {fmt!r} but ffmpeg is not installed. "
                "The backend will try libsndfile first and fall back to an error; "
                "install ffmpeg or set the default to wav."
            )
        return problems


_REQUEST_OPTIONS = frozenset({
    "num_step", "guidance_scale", "duration", "t_shift", "class_temperature",
})


def _with_path(model: AudioModelConfig, path: str) -> AudioModelConfig:
    """A shallow copy of *model* pointing at *path*, for size measurement."""
    import dataclasses

    return dataclasses.replace(model, path=path)


def default_device(gpu: GPUConfig) -> str:
    """The torch device string for a GPU with no explicit `recipe.device`.

    `xpu` for the Intel cards this project targets — torch's XPU backend goes
    through the same Level Zero runtime as the SYCL llama.cpp build, so it
    honours the `ONEAPI_DEVICE_SELECTOR` pin set below and index 0 inside the
    process is the card the model was bound to.
    """
    backend = Backend(gpu.backend) if gpu.backend else Backend.SYCL
    return "xpu" if backend == Backend.SYCL else "cpu"


# Variables that only mean something to a SYCL llama.cpp build, and that a
# stale value of can misdirect a torch process onto the wrong device.
_SYCL_ONLY = (
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_CACHE_PERSISTENT",
    "SYCL_CACHE_DIR",
    "SYCL_DEVICE_FILTER",
    "SYCL_DEVICE_ALLOWLIST",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "GGML_SYCL_DISABLE_OPT",
)


def build_env_for(cfg: Config, gpu: GPUConfig, device: str) -> dict[str, str]:
    """Environment for the sidecar, pinned to this model's GPU.

    Built here rather than through the core's ``build_env`` for one decisive
    reason: that function sources oneAPI's setvars.sh when it judges the
    runtime libraries missing, which is right for llama-server and wrong for
    torch. A torch XPU build ships its own libsycl, Unified Runtime adapters
    and MKL; prepending a system oneAPI's lib directory puts a second,
    differently versioned copy of all of it ahead of them, and the process
    loads both and dies inside the SYCL device-code build — visible in the
    core dump as two libccl versions mapped at once. It bites under systemd
    and not in a shell, because a service environment is bare enough for that
    "are the libraries missing?" heuristic to answer yes.

    What is worth taking from the core is the arch profile: the per-generation
    SYCL settings (Battlemage wants ``SYCL_CACHE_PERSISTENT=0``) and the list
    of inherited variables known to break things. A non-Intel device gets the
    ambient environment, since neither selector means anything to it.
    """
    if not device.startswith("xpu"):
        return os.environ.copy()
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    profile = profile_for(arch)
    env = os.environ.copy()
    for key in (*_SYCL_ONLY, *profile.sycl_env_remove):
        env.pop(key, None)
    env.update(profile.sycl_env)
    # torch reaches an Arc card through Level Zero, so it wants the SYCL
    # selector whatever backend the GPU is configured for.
    env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpu.sycl_index}"
    return env


def voice_entry(cfg: Config, model: AudioModelConfig, voice: VoiceConfig) -> dict[str, Any]:
    """One voice, as the sidecar's JSON expects it."""
    prompt_file = voice.prompt_file
    if not prompt_file and voice.ref_audio:
        # Cache the encoded reference per model: the encoding is produced by
        # that model's audio tokenizer, so it is not portable between models
        # and a shared filename would hand one model another's prompt.
        prompt_file = str(tts_state_dir(cfg) / "prompts" / f"{model.name}-{voice.name}.pt")
    return {
        "ref_audio": str(Path(voice.ref_audio).expanduser()) if voice.ref_audio else "",
        "ref_text": voice.ref_text,
        "instruct": voice.instruct,
        "language": voice.language,
        "prompt_file": prompt_file,
        "aliases": list(voice.aliases),
    }


def write_voices_file(cfg: Config, audio: AudioConfig, model: AudioModelConfig) -> Path:
    """Write this model's voice table and return its path.

    Rewritten on every plan build, and re-read by the sidecar whenever its
    mtime changes, so `arc-llama audio voice add` reaches a running backend
    without a restart.
    """
    recipe = model.audio_recipe()
    payload = {
        "default_voice": recipe.default_voice,
        "voices": {
            v.name: voice_entry(cfg, model, v) for v in audio.voices_for(model.name)
        },
    }
    directory = tts_state_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model.name}.voices.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _can_import(python: str, module: str) -> bool:
    """Whether *python* can import *module*, without paying for the import.

    `find_spec` only resolves the module on the path; actually importing
    omnivoice would drag in torch and take tens of seconds, which is far too
    slow for `arc-llama doctor`.
    """
    try:
        proc = subprocess.run(
            [python, "-c",
             f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


register_engine(OmniVoiceEngine())
