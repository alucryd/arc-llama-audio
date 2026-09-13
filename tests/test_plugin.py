"""The plugin's contract with the core, and the two endpoints."""
from __future__ import annotations

import sys

import pytest
from arc_llama.config import Config, GPUConfig, PathsConfig
from arc_llama.server import create_app
from fastapi.testclient import TestClient

from arc_llama_audio import proxy
from arc_llama_audio.config import AudioConfig, AudioModelConfig, VoiceConfig, load_config


def _gpu(**kw):
    d = dict(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=16384,
             enabled=True, backend="sycl")
    d.update(kw)
    return GPUConfig(**d)


def _asr_model(tmp_path, **kw):
    weights = tmp_path / "Qwen3-ASR-0.6B-Q8_0.gguf"
    weights.write_bytes(b"GGUF")
    mmproj = tmp_path / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
    mmproj.write_bytes(b"GGUF")
    d = dict(name="asr", path=str(weights), port=18090, gpu_pci_slot="0000:03:00.0",
             task="asr", engine="llamacpp", recipe={"mmproj": str(mmproj)})
    d.update(kw)
    return AudioModelConfig(**d)


class FakeBackendPlan:
    backend_url = "http://fake-backend"


class FakeServer:
    plan = FakeBackendPlan()
    is_running = True
    ready = True


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    def __init__(self, content=b'{"text": "hello"}'):
        self.content = content


class RecordingClient:
    """Captures the request the proxy builds for the backend."""

    last: dict = {}
    response = FakeResponse

    def __init__(self, timeout=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def post(self, url, **kwargs):
        RecordingClient.last = {"url": url, **kwargs}
        return RecordingClient.response()

    async def aclose(self):
        return None


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A real core app with the plugin's state pre-seeded."""
    audio = AudioConfig(audio_models=[_asr_model(tmp_path, aliases=["whisper-1"])])
    monkeypatch.setattr(proxy.httpx, "AsyncClient", RecordingClient)
    monkeypatch.setattr(
        proxy.Backends, "ensure",
        lambda self, model, cfg, audio, log_dir: _completed(FakeServer()),
    )
    application = create_app(Config(paths=PathsConfig(state_dir=str(tmp_path)), gpus=[_gpu()]))
    application.state.audio_config = audio
    return application


def _completed(value):
    async def _coro():
        return value
    return _coro()


class TestDiscovery:
    def test_core_finds_both_plugins(self):
        """The entry points are what make this a plugin at all."""
        from arc_llama.plugins import load_plugins

        names = {p.name for p in load_plugins()}
        assert {"audio-asr", "audio-tts"} <= names

    def test_routes_are_mounted(self, app):
        paths = {r.path for r in app.routes}
        assert "/v1/audio/transcriptions" in paths
        assert "/v1/audio/speech" in paths

    def test_importing_the_plugin_does_not_import_torch(self):
        """The core imports this at app creation; torch must stay out of it."""
        for mod in ("arc_llama_audio.asr", "arc_llama_audio.tts_plugin"):
            __import__(mod)
        assert "torch" not in sys.modules


class TestTranscriptions:
    def test_multipart_upload_is_forwarded(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "asr", "language": "en"},
                files={"file": ("speech.wav", b"RIFF", "audio/wav")},
            )
        assert r.status_code == 200
        call = RecordingClient.last
        assert call["url"] == "http://fake-backend/v1/audio/transcriptions"
        assert call["data"]["model"] == "asr"
        assert call["files"]["file"][1] == b"RIFF"

    def test_alias_is_rewritten_to_the_backend_id(self, app):
        """The backend only answers to its own id, whatever alias got us here."""
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-1"},
                files={"file": ("s.wav", b"RIFF", "audio/wav")},
            )
        assert r.status_code == 200
        assert RecordingClient.last["data"]["model"] == "asr"

    def test_json_body_form_is_supported(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                json={"model": "asr", "audio": "/srv/clip.wav"},
            )
        assert r.status_code == 200
        assert RecordingClient.last["json"]["audio"] == "/srv/clip.wav"

    def test_missing_file_is_rejected(self, app):
        with TestClient(app) as client:
            assert client.post("/v1/audio/transcriptions", data={"model": "asr"}).status_code == 400

    def test_stream_needs_a_streaming_model(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "asr", "stream": "true"},
                files={"file": ("s.wav", b"RIFF", "audio/wav")},
            )
        assert r.status_code == 400
        assert "streaming" in r.json()["detail"]

    def test_no_models_registered_returns_501(self, tmp_path, monkeypatch):
        monkeypatch.setattr(proxy.httpx, "AsyncClient", RecordingClient)
        app = create_app(Config(paths=PathsConfig(state_dir=str(tmp_path)), gpus=[_gpu()]))
        app.state.audio_config = AudioConfig()
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "x"},
                files={"file": ("s.wav", b"RIFF", "audio/wav")},
            )
        assert r.status_code == 501


class TestAsrMarkers:
    def test_qwen3_framing_is_stripped(self, app, monkeypatch):
        """HA must not have to match 'language English<asr_text>' as a command."""
        monkeypatch.setattr(
            RecordingClient, "response",
            lambda: FakeResponse(b'{"text": "language English<asr_text>lights off"}'),
        )
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "asr"},
                files={"file": ("s.wav", b"RIFF", "audio/wav")},
            )
        assert r.json() == {"text": "lights off"}

    def test_plain_transcript_untouched(self):
        assert proxy.strip_asr_markers("lights off") == "lights off"

    def test_non_json_body_is_passed_through(self):
        assert proxy.sanitize_transcription(b"not json") == b"not json"


class TestConfigFile:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "audio.toml"
        cfg = AudioConfig(
            tts_python="/venv/bin/python",
            audio_models=[_asr_model(tmp_path, aliases=["whisper-1"])],
            voices=[VoiceConfig(name="glados", instruct="female", aliases=["alloy"])],
        )
        cfg.save(path)
        loaded = load_config(path)
        assert loaded.tts_python == "/venv/bin/python"
        assert loaded.audio_models[0].aliases == ["whisper-1"]
        assert loaded.find_voice("alloy").name == "glados"
        assert loaded.audio_models[0].audio_recipe().mmproj.endswith("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf")

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_config(tmp_path / "nope.toml").audio_models == []

    def test_flat_launch_keys_migrate_into_the_recipe(self, tmp_path):
        path = tmp_path / "audio.toml"
        path.write_text(
            'version = 1\n\n[[audio_models]]\nname = "asr"\npath = "/m.gguf"\nport = 1\n'
            'gpu_pci_slot = "0000:03:00.0"\nmmproj = "/mm.gguf"\nctx = 2048\n',
            encoding="utf-8",
        )
        recipe = load_config(path).audio_models[0].audio_recipe()
        assert recipe.mmproj == "/mm.gguf"
        assert recipe.ctx == 2048

    def test_ctx_defaults_are_sane(self, tmp_path):
        """Not the GGUF's 65536, which is ~7 GB of KV for a 1.7B model."""
        assert _asr_model(tmp_path).audio_recipe().ctx == 4096

    def test_voices_narrow_to_their_models(self):
        cfg = AudioConfig(voices=[
            VoiceConfig(name="everywhere"),
            VoiceConfig(name="only-a", models=["a"]),
        ])
        assert {v.name for v in cfg.voices_for("a")} == {"everywhere", "only-a"}
        assert {v.name for v in cfg.voices_for("b")} == {"everywhere"}
