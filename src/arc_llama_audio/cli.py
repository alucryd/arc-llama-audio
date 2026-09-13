"""`arc-llama-audio` — register and inspect speech models.

A plugin cannot add commands to arc-llama's own CLI: the contract is FastAPI
routes and lifecycle hooks, nothing else. So this is a separate console
script over the plugin's own `audio.toml`.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
from arc_llama.config import load_config as load_core_config

from arc_llama_audio.config import (
    AUDIO_ENGINE_LLAMACPP,
    AudioConfig,
    AudioModelConfig,
    VoiceConfig,
    default_config_path,
    load_config,
)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _load(path: Path | None) -> tuple[AudioConfig, Path]:
    target = path or default_config_path()
    return load_config(target), target


def _next_port(audio: AudioConfig, core_ports: set[int]) -> int:
    used = {m.port for m in audio.audio_models} | core_ports
    port = 18090
    while port in used:
        port += 1
    return port


@click.group()
@click.option(
    "--config", "config_path", default=None, type=click.Path(path_type=Path),
    help="Path to audio.toml (default: beside arc-llama's config.toml).",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """Speech models for arc-llama."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command("add")
@click.argument("path")
@click.option("--name", default=None, help="Short id (default: derived from PATH).")
@click.option("--task", type=click.Choice(["asr", "tts"]), default="asr")
@click.option("--engine", default=None, help="TTS engine name (default: omnivoice for tts).")
@click.option("--mmproj", default=None, help="ASR: audio projector GGUF. Auto-detected if beside the weights.")
@click.option("--gpu", "gpu_pci_slot", default=None, help="PCI slot (default: first enabled GPU).")
@click.option("--port", type=int, default=None)
@click.option("--alias", "aliases", multiple=True, help="Extra ids clients may send (e.g. whisper-1).")
@click.option("--mode", type=click.Choice(["offline", "streaming"]), default="offline")
@click.option("--ctx", "ctx_len", type=int, default=0, help="ASR: context length (-c).")
@click.option("--option", "options", multiple=True, help="Engine option, key=value. Repeatable.")
@click.option("--no-strip-markers", is_flag=True, help="Keep Qwen3-ASR's raw output framing.")
@click.pass_context
def add(ctx, path, name, task, engine, mmproj, gpu_pci_slot, port, aliases, mode,
        ctx_len, options, no_strip_markers):
    """Register a speech model. PATH is a .gguf, a directory, or an HF repo id."""
    audio, cfg_path = _load(ctx.obj["config_path"])
    core = load_core_config()
    if not core.gpus:
        raise click.ClickException("No GPUs in arc-llama's config — run `arc-llama init` first.")
    if gpu_pci_slot is None:
        enabled = [g for g in core.gpus if g.enabled] or core.gpus
        gpu_pci_slot = enabled[0].pci_slot

    engine = engine or ("omnivoice" if task == "tts" else AUDIO_ENGINE_LLAMACPP)
    local = Path(path).expanduser()
    derived = name or re.sub(r"[^a-z0-9._-]+", "-", (local.stem if local.is_file() else local.name).lower()).strip("-")
    if not NAME_RE.match(derived):
        raise click.ClickException(f"Bad model name {derived!r}; pass --name.")
    if any(m.name == derived for m in audio.audio_models):
        raise click.ClickException(f"{derived!r} is already registered.")

    recipe: dict = {}
    if task == "asr":
        if mmproj is None and local.is_file():
            # The projector normally sits beside the weights under a
            # predictable name; finding it saves a flag the user would
            # otherwise discover from an error message.
            sibling = local.parent / f"mmproj-{local.name}"
            if sibling.exists():
                mmproj = str(sibling)
                click.echo(f"Found projector beside the weights: {sibling.name}")
        if not mmproj:
            raise click.ClickException(
                "ASR needs --mmproj, the audio projector GGUF published beside the "
                "weights. Without it llama-server loads a plain text LLM and "
                "transcription returns confident nonsense rather than failing."
            )
        if not Path(mmproj).expanduser().exists():
            raise click.ClickException(f"mmproj not found: {mmproj}")
        recipe["mmproj"] = str(Path(mmproj).expanduser().resolve())
        if ctx_len:
            recipe["ctx"] = ctx_len
    if options:
        parsed: dict = {}
        for item in options:
            key, _, value = item.partition("=")
            if not _:
                raise click.ClickException(f"--option needs key=value, got {item!r}")
            parsed[key] = _coerce(value)
        recipe["options"] = parsed

    entry = AudioModelConfig(
        name=derived,
        path=str(local.resolve()) if local.exists() else path,
        port=port or _next_port(audio, {m.port for m in core.models}),
        gpu_pci_slot=gpu_pci_slot,
        engine=engine,
        task=task,
        mode=mode,
        recipe=recipe,
        display_name=derived,
        aliases=list(aliases),
        strip_asr_markers=not no_strip_markers,
    )
    audio.audio_models.append(entry)
    audio.save(cfg_path)
    click.echo(f"Registered {entry.name} ({entry.task}, {entry.engine}) on port {entry.port}")


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


@cli.command("list")
@click.pass_context
def list_models(ctx):
    """List registered speech models."""
    audio, _ = _load(ctx.obj["config_path"])
    if not audio.audio_models:
        click.echo("No speech models registered.")
        return
    for m in audio.audio_models:
        detail = m.audio_recipe().mmproj if m.task == "asr" else m.engine
        click.echo(f"{m.name:20} {m.task:4} {m.mode:9} port={m.port:<6} {Path(detail).name if detail else ''}")


@cli.command("rm")
@click.argument("name")
@click.pass_context
def rm(ctx, name):
    """Unregister a speech model."""
    audio, cfg_path = _load(ctx.obj["config_path"])
    before = len(audio.audio_models)
    audio.audio_models = [m for m in audio.audio_models if m.name != name]
    if len(audio.audio_models) == before:
        raise click.ClickException(f"Unknown model: {name}")
    audio.save(cfg_path)
    click.echo(f"Removed {name}")


@cli.command("set-python")
@click.argument("path")
@click.pass_context
def set_python(ctx, path):
    """Point TTS sidecars at the interpreter that has their model installed."""
    audio, cfg_path = _load(ctx.obj["config_path"])
    resolved = shutil.which(path) if "/" not in path else (path if Path(path).exists() else None)
    if resolved is None:
        raise click.ClickException(f"Not found: {path}")
    audio.tts_python = str(Path(resolved).resolve())
    audio.save(cfg_path)
    click.echo(f"tts_python = {audio.tts_python}")


@cli.group("voice")
def voice() -> None:
    """Named voices resolvable from the OpenAI `voice` field."""


@voice.command("add")
@click.argument("name")
@click.option("--ref-audio", default="", help="Clip to clone: 3–10 s of clean speech.")
@click.option("--ref-text", default="", help="Transcript of --ref-audio. Omitting it costs a Whisper pass on first use.")
@click.option("--instruct", default="", help="Design a voice from attributes instead of cloning.")
@click.option("--language", default="")
@click.option("--alias", "aliases", multiple=True, help="e.g. alloy, for clients that hardcode OpenAI ids.")
@click.option("--model", "models", multiple=True, help="Restrict to these TTS models (default: all).")
@click.pass_context
def voice_add(ctx, name, ref_audio, ref_text, instruct, language, aliases, models):
    """Register a voice. With neither --ref-audio nor --instruct the model uses its own."""
    audio, cfg_path = _load(ctx.obj["config_path"])
    if any(v.name == name for v in audio.voices):
        raise click.ClickException(f"Voice {name!r} already exists.")
    if ref_audio and not Path(ref_audio).expanduser().exists():
        raise click.ClickException(f"Reference audio not found: {ref_audio}")
    audio.voices.append(VoiceConfig(
        name=name,
        ref_audio=str(Path(ref_audio).expanduser().resolve()) if ref_audio else "",
        ref_text=ref_text,
        instruct=instruct,
        language=language,
        aliases=list(aliases),
        models=list(models),
    ))
    audio.save(cfg_path)
    click.echo(f"Registered voice {name}")


@voice.command("list")
@click.pass_context
def voice_list(ctx):
    audio, _ = _load(ctx.obj["config_path"])
    for v in audio.voices:
        kind = "clone" if v.ref_audio else ("design" if v.instruct else "model's own")
        click.echo(f"{v.name:20} {kind:12} {','.join(v.aliases)}")


@voice.command("rm")
@click.argument("name")
@click.pass_context
def voice_rm(ctx, name):
    audio, cfg_path = _load(ctx.obj["config_path"])
    before = len(audio.voices)
    audio.voices = [v for v in audio.voices if v.name != name]
    if len(audio.voices) == before:
        raise click.ClickException(f"Unknown voice: {name}")
    audio.save(cfg_path)
    click.echo(f"Removed voice {name}")


@cli.command("bench")
@click.argument("name")
@click.option("--text", default="Turn off the kitchen lights.")
@click.option("--steps", default="8,16,24,32", help="num_step values to sweep.")
@click.option("--runs", type=int, default=3)
@click.option("--voice", "voice_name", default="")
@click.option("--compile/--no-compile", "compile_override", default=None,
              help="Override the model's compile setting. --no-compile first: an "
                   "Inductor compile can peg the GPU far longer than the synthesis "
                   "it speeds up.")
@click.option("--compile-targets", default="")
@click.option("--timeout", type=int, default=900, help="Give up after N seconds. 0 waits forever.")
@click.pass_context
def bench(ctx, name, text, steps, runs, voice_name, compile_override, compile_targets, timeout):
    """Measure a TTS model's latency across solver-step counts.

    Loads it exactly as serving would — same device, dtype, quantized weights
    and voices — so the latency/quality trade is measured on the card that
    serves. Starts a second copy of the model.
    """
    audio, _ = _load(ctx.obj["config_path"])
    model = audio.find_model(name, task="tts")
    if model is None:
        raise click.ClickException(f"Unknown TTS model: {name}")
    core = load_core_config()
    gpu = core.find_gpu(model.gpu_pci_slot)
    if gpu is None:
        raise click.ClickException(f"{name} is bound to unknown GPU {model.gpu_pci_slot}")

    from arc_llama_audio.tts import require_engine

    plan = require_engine(model.engine).build_plan(core, audio, model, gpu, host=core.server.host)
    argv = [*plan.argv, "--bench", text, "--bench-steps", steps, "--bench-runs", str(runs)]
    if voice_name:
        argv += ["--bench-voice", voice_name]
    if compile_override is not None:
        argv.append("--compile" if compile_override else "--no-compile")
    if compile_targets:
        argv += ["--compile-targets", compile_targets]
    click.echo(" ".join(argv) + "\n")
    try:
        completed = subprocess.run(argv, env=plan.env, timeout=timeout or None)
    except subprocess.TimeoutExpired:
        raise click.ClickException(
            f"No result after {timeout}s. A bench that pegs the GPU and prints "
            "nothing is usually an Inductor compile, not slow synthesis. Try "
            "--no-compile, then --compile-targets audio_heads."
        ) from None
    sys.exit(completed.returncode)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
