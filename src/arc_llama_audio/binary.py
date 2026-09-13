"""Finding and vetting the llama-server binary.

Kept here rather than imported from the core: these are small, and depending
on core internals that may not exist in the installed version turns a missing
attribute into a 500 at request time. Everything below works against any
arc-llama that has a `paths.llama_server`.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("arc_llama_audio.binary")

_HELP_TIMEOUT_S = 20


def resolve_binary(path_or_name: str) -> str | None:
    """Resolve a configured binary to a runnable path, or None if absent.

    A bare name goes through PATH the way the shell would; anything with a
    separator is taken literally. None rather than the unresolved string, so
    callers can say "not installed" instead of leaving the user to decode an
    ENOENT from a subprocess that never started.
    """
    if not path_or_name:
        return None
    candidate = Path(path_or_name).expanduser()
    if os.sep in path_or_name or (os.altsep and os.altsep in path_or_name):
        return str(candidate) if candidate.exists() else None
    return shutil.which(str(candidate)) or None


def supports_mmproj(binary: str, env: dict[str, str] | None = None) -> bool | None:
    """Whether *binary* was built with multimodal support.

    True/False from its own `--help`, or None when the probe could not run —
    which is not the same answer, and is why this is tri-state. A SYCL build
    probed without its Intel runtime exits 127, and treating that as "no
    multimodal" would refuse a perfectly good binary.

    ``--help`` is used because it touches no GPU and returns immediately.
    """
    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=_HELP_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("could not probe %s: %s", binary, e)
        return None
    if proc.returncode:
        log.debug("%s --help exited %s", binary, proc.returncode)
        return None
    return "--mmproj" in (proc.stdout + proc.stderr)
