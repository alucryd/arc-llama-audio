"""How to launch llama-server as a transcription backend."""
from __future__ import annotations

from pathlib import Path

from arc_llama.arch import Arch, profile_for
from arc_llama.config import Config, GPUConfig
from arc_llama.launcher import LaunchPlan, build_env

from arc_llama_audio.binary import resolve_binary, supports_mmproj
from arc_llama_audio.config import AudioModelConfig


def build_asr_plan(
    cfg: Config, model: AudioModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
) -> LaunchPlan:
    """An ordinary llama-server invocation with an audio projector attached.

    It goes through the core's ``build_env``, so it inherits the arch SYCL
    profile, the stripped known-bad variables and the device selector — the
    reason transcription runs on llama-server at all is that this path already
    exists and has a SYCL build.

    Launched without a router config: llama.cpp's own multi-model mode is
    reported to 500 on this endpoint, and arc-llama is the router anyway.
    """
    recipe = model.audio_recipe()
    if not recipe.mmproj:
        raise RuntimeError(
            f"audio model {model.name!r} has no 'mmproj' in its recipe. "
            "llama.cpp keeps the audio projector in a separate GGUF "
            "(mmproj-*.gguf, published beside the weights); without it "
            "llama-server loads the model as a plain text LLM and "
            "transcription returns confident nonsense."
        )
    mmproj_path = Path(recipe.mmproj).expanduser()
    if not mmproj_path.exists():
        raise RuntimeError(f"audio model {model.name!r}: mmproj not found at {mmproj_path}")
    binary = resolve_binary(cfg.paths.llama_server)
    if binary is None:
        raise RuntimeError(
            f"llama-server not found at {cfg.paths.llama_server!r}. Run "
            "`arc-llama install-runtime` or set paths.llama_server."
        )

    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    env = build_env(
        profile_for(arch),
        gpu,
        llama_server=binary,
        oneapi_setvars=getattr(cfg.paths, "oneapi_setvars", None),
    )

    if supports_mmproj(binary, env) is False:
        raise RuntimeError(
            f"{binary} has no --mmproj, so it was built without multimodal "
            "(mtmd) support and cannot serve ASR. Install a newer build with "
            "`arc-llama install-runtime`."
        )

    # -c is not optional here. llama-server's default is 0, meaning "whatever
    # the GGUF was trained for", and Qwen3-ASR advertises 65536 — a ~7 GB KV
    # cache in front of 2 GB of weights, held for the life of the process.
    # -np 1 keeps that budget from being split into auto-chosen slots as well.
    argv: list[str] = [
        binary,
        "-m", str(Path(model.path).expanduser()),
        "--mmproj", str(mmproj_path),
        "--host", host,
        "--port", str(model.port),
        "-ngl", str(recipe.n_gpu_layers),
        "-c", str(recipe.ctx),
        "-np", "1",
        "-ctk", recipe.cache_type_k,
        "-ctv", recipe.cache_type_v,
    ]
    argv.extend(recipe.extra_flags)

    backend_url = f"http://{host}:{model.port}"
    return LaunchPlan(
        argv=argv, env=env, backend_url=backend_url, health_url=f"{backend_url}/health"
    )
