"""The `arc-llama-audio` console script.

A plugin cannot add subcommands to arc-llama's CLI, so registration and
benchmarking live here instead.
"""
from __future__ import annotations

import subprocess

import pytest
from click.testing import CliRunner

import arc_llama_audio.cli as cli_mod
from arc_llama_audio.config import load_config


@pytest.fixture
def core_config(tmp_path, monkeypatch):
    """A core config with one GPU, so `add` has something to bind to."""
    from arc_llama.config import Config, GPUConfig, PathsConfig

    cfg = Config(
        paths=PathsConfig(state_dir=str(tmp_path), llama_server="/usr/bin/llama-server"),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage",
                        vram_mb=16384, enabled=True, backend="sycl")],
    )
    monkeypatch.setattr(cli_mod, "load_core_config", lambda: cfg)
    return cfg


def _weights(tmp_path):
    w = tmp_path / "Qwen3-ASR-0.6B-Q8_0.gguf"
    w.write_bytes(b"GGUF")
    (tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf").write_bytes(b"GGUF")
    return w


def _run(audio_toml, *args):
    return CliRunner().invoke(cli_mod.cli, ["--config", str(audio_toml), *args], obj={})


class TestAdd:
    def test_projector_is_found_beside_the_weights(self, tmp_path, core_config):
        toml = tmp_path / "audio.toml"
        result = _run(toml, "add", str(_weights(tmp_path)), "--name", "asr")
        assert result.exit_code == 0, result.output
        assert "Found projector" in result.output
        recipe = load_config(toml).audio_models[0].audio_recipe()
        assert recipe.mmproj.endswith("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf")

    def test_asr_without_a_projector_is_refused(self, tmp_path, core_config):
        """Without it llama-server loads a text LLM and invents words."""
        lonely = tmp_path / "solo.gguf"
        lonely.write_bytes(b"GGUF")
        result = _run(tmp_path / "audio.toml", "add", str(lonely), "--name", "asr")
        assert result.exit_code != 0
        assert "mmproj" in result.output

    def test_ports_avoid_the_core_registry(self, tmp_path, core_config):
        from arc_llama.config import ModelConfig

        core_config.models.append(ModelConfig(
            name="llm", path="/m.gguf", port=18090, gpu_pci_slot="0000:03:00.0"))
        toml = tmp_path / "audio.toml"
        _run(toml, "add", str(_weights(tmp_path)), "--name", "asr")
        assert load_config(toml).audio_models[0].port != 18090

    def test_duplicate_names_are_refused(self, tmp_path, core_config):
        toml = tmp_path / "audio.toml"
        _run(toml, "add", str(_weights(tmp_path)), "--name", "asr")
        second = _run(toml, "add", str(_weights(tmp_path)), "--name", "asr")
        assert second.exit_code != 0
        assert "already registered" in second.output

    def test_options_are_typed(self, tmp_path, core_config):
        toml = tmp_path / "audio.toml"
        _run(toml, "add", "k2-fsa/OmniVoice", "--name", "tts", "--task", "tts",
             "--option", "num_step=8", "--option", "compile=false")
        options = load_config(toml).audio_models[0].audio_recipe().options
        assert options == {"num_step": 8, "compile": False}


class TestVoices:
    def test_add_and_list(self, tmp_path, core_config):
        toml = tmp_path / "audio.toml"
        _run(toml, "voice", "add", "glados", "--instruct", "female", "--alias", "alloy")
        assert load_config(toml).find_voice("alloy").name == "glados"
        assert "design" in _run(toml, "voice", "list").output

    def test_missing_reference_audio_is_refused(self, tmp_path, core_config):
        result = _run(tmp_path / "audio.toml", "voice", "add", "x",
                      "--ref-audio", str(tmp_path / "nope.wav"))
        assert result.exit_code != 0


class TestBench:
    """The bench must be able to contradict the model's compile settings.

    A compiled run that never returns is the case this exists for: without an
    override, measuring eager means editing config and restarting.
    """

    def _registered(self, tmp_path, core_config, options=""):
        toml = tmp_path / "audio.toml"
        args = ["add", "k2-fsa/OmniVoice", "--name", "tts", "--task", "tts"]
        if options:
            args += ["--option", options]
        _run(toml, *args)
        return toml

    def _argv(self, monkeypatch, toml, *extra):
        captured: dict = {}

        class _Done:
            returncode = 0

        def _fake_run(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["timeout"] = kwargs.get("timeout")
            return _Done()

        monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
        # The plan needs a real engine but not a real model.
        result = _run(toml, "bench", "tts", *extra)
        assert "argv" in captured, result.output
        return captured

    def test_no_compile_wins_over_the_configured_default(self, tmp_path, core_config, monkeypatch):
        toml = self._registered(tmp_path, core_config, "compile=true")
        argv = self._argv(monkeypatch, toml, "--no-compile")["argv"]
        # Both appear; argparse takes the last, which is why order matters.
        assert argv.index("--no-compile") > argv.index("--compile")

    def test_compile_is_inherited_when_not_overridden(self, tmp_path, core_config, monkeypatch):
        toml = self._registered(tmp_path, core_config, "compile=true")
        argv = self._argv(monkeypatch, toml)["argv"]
        assert "--compile" in argv and "--no-compile" not in argv

    def test_a_hang_is_bounded_by_default(self, tmp_path, core_config, monkeypatch):
        toml = self._registered(tmp_path, core_config)
        assert self._argv(monkeypatch, toml)["timeout"] == 900

    def test_timeout_zero_waits_forever(self, tmp_path, core_config, monkeypatch):
        toml = self._registered(tmp_path, core_config)
        assert self._argv(monkeypatch, toml, "--timeout", "0")["timeout"] is None

    def test_timing_out_explains_what_to_try(self, tmp_path, core_config, monkeypatch):
        toml = self._registered(tmp_path, core_config)

        def _boom(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or 0)

        monkeypatch.setattr(cli_mod.subprocess, "run", _boom)
        result = _run(toml, "bench", "tts")
        assert result.exit_code != 0
        assert "--no-compile" in result.output
        assert "audio_heads" in result.output
