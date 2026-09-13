"""The OmniVoice sidecar's own behaviour.

Loaded from its file rather than imported as a module: it runs under a
foreign interpreter that has no arc_llama or arc_llama_audio at all, and
loading it this way exercises that same standalone contract — if it ever
grows an import from either package, every test here fails.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _sidecar():
    from arc_llama_audio.tts import omnivoice

    spec = importlib.util.spec_from_file_location("omnivoice_server", omnivoice.SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SIDECAR = _sidecar()
Engine = _SIDECAR.Engine
VoiceBook = _SIDECAR.VoiceBook
BadRequestError = _SIDECAR.BadRequestError
build_parser = _SIDECAR.build_parser
encode_audio = _SIDECAR.encode_audio


class TestVoiceBook:
    def _book(self, tmp_path, payload):

        path = tmp_path / "voices.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return VoiceBook(str(path)), path

    def test_exact_and_case_insensitive_lookup(self, tmp_path):
        book, _ = self._book(tmp_path, {"voices": {"GLaDOS": {"instruct": "female"}}})
        assert book.lookup("GLaDOS")[0] == "GLaDOS"
        assert book.lookup("glados")[0] == "GLaDOS"

    def test_alias_lookup(self, tmp_path):
        book, _ = self._book(
            tmp_path, {"voices": {"glados": {"instruct": "female", "aliases": ["alloy"]}}}
        )
        assert book.lookup("alloy")[0] == "glados"

    def test_unknown_voice_falls_back_to_the_default(self, tmp_path):
        """A substituted voice beats a failed request for a speech client."""
        book, _ = self._book(
            tmp_path,
            {"default_voice": "glados", "voices": {"glados": {"instruct": "female"}}},
        )
        assert book.lookup("nova")[0] == "glados"

    def test_no_match_and_no_default_is_none(self, tmp_path):
        book, _ = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("nova") is None

    def test_an_edited_file_is_picked_up_without_a_restart(self, tmp_path):
        book, path = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("narrator") is None

        import os

        payload = {"voices": {"glados": {"instruct": "female"}, "narrator": {"instruct": "male"}}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        os.utime(path, (0, 0))  # force a different mtime

        assert book.lookup("narrator")[0] == "narrator"

    def test_a_corrupt_file_keeps_the_last_good_table(self, tmp_path):
        """A half-written voices file must not take TTS down."""
        book, path = self._book(tmp_path, {"voices": {"glados": {"instruct": "female"}}})
        assert book.lookup("glados") is not None

        import os

        path.write_text("{not json", encoding="utf-8")
        os.utime(path, (0, 0))
        assert book.lookup("glados") is not None



class TestAutoVoiceSynthesis:
    """A fine-tuned model must reach generate() with no prompt attached."""

    def _generate_kwargs(self, voices_json, voice_field):
        import json as _json
        import tempfile
        from pathlib import Path as _Path

        d = _Path(tempfile.mkdtemp())
        (d / "v.json").write_text(_json.dumps(voices_json), encoding="utf-8")
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--voices", str(d / "v.json")]
        )
        engine = Engine(args, VoiceBook(str(d / "v.json")))
        captured: dict = {}

        class FakeModel:
            def generate(self, **kw):
                captured.update(kw)
                return [[0.0]]

        engine.model = FakeModel()
        original = _SIDECAR.encode_audio
        _SIDECAR.encode_audio = lambda s, r, f: (b"", "audio/wav")
        try:
            engine.synthesize({"input": "hi", "voice": voice_field})
        finally:
            _SIDECAR.encode_audio = original
        return captured

    def test_no_registered_voices_means_the_models_own_voice(self):
        """Clients must send a `voice`; with none registered it is ignored."""
        kw = self._generate_kwargs({"voices": {}}, "alloy")
        assert kw["text"] == "hi"
        assert "voice_clone_prompt" not in kw
        assert "instruct" not in kw

    def test_a_registered_auto_voice_adds_nothing(self):
        kw = self._generate_kwargs({"voices": {"glados": {}}}, "glados")
        assert "voice_clone_prompt" not in kw
        assert "instruct" not in kw

    def test_a_design_voice_would_override_the_finetune(self):
        """The failure mode to avoid: a prompt layered on baked-in weights."""
        kw = self._generate_kwargs(
            {"default_voice": "narrator", "voices": {"narrator": {"instruct": "male"}}},
            "alloy",
        )
        assert kw["instruct"] == "male"



class TestSidecarImportIsolation:
    """A sidecar's own directory is on `sys.path`, so its neighbours matter.

    Running a script puts its directory at the front of `sys.path` — ahead of
    both site-packages and the stdlib. These scripts therefore live alone in
    `sidecars/`; when `omnivoice_server.py` sat beside the engine module,
    `from omnivoice import OmniVoice` resolved to the engine module
    and failed with "cannot import name 'OmniVoice'" on machines where
    OmniVoice was installed and importable.
    """

    def _sidecar_dir(self):

        from arc_llama_audio.tts import omnivoice

        return Path(omnivoice.SERVER_SCRIPT).parent

    def test_no_neighbour_shadows_anything_a_sidecar_imports(self):
        import ast

        directory = self._sidecar_dir()
        scripts = sorted(directory.glob("*.py"))
        assert scripts, f"no sidecar scripts found in {directory}"

        # Every top-level module name any sidecar imports, however it imports it.
        imported: set[str] = set()
        for script in scripts:
            tree = ast.parse(script.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])

        neighbours = {p.stem for p in directory.glob("*.py")}
        clashes = sorted(neighbours & imported)
        assert not clashes, (
            f"{directory} contains {clashes}, which would shadow the same-named "
            "module for every sidecar run from that directory"
        )

    def test_the_engine_module_is_not_a_neighbour(self):
        """The specific collision that broke `from omnivoice import OmniVoice`."""

        from arc_llama_audio.tts import omnivoice

        engine_module = Path(omnivoice.__file__)
        assert engine_module.stem == "omnivoice"
        assert engine_module.parent != self._sidecar_dir()

    def test_running_as_a_script_resolves_the_installed_package(self, tmp_path):
        """End to end: launch it the way arc-llama does and see what wins."""
        import os
        import shutil
        import subprocess
        import sys

        from arc_llama_audio.tts import omnivoice

        script_dir = tmp_path / "sidecars"
        script_dir.mkdir()
        shutil.copy(omnivoice.SERVER_SCRIPT, script_dir / "omnivoice_server.py")

        # The real package, as a virtualenv would provide it.
        site = tmp_path / "site"
        site.mkdir()
        (site / "omnivoice.py").write_text(
            "WHICH = 'installed'\nOmniVoice = object\n", encoding="utf-8"
        )

        driver = script_dir / "driver.py"
        driver.write_text(
            "import runpy, sys\n"
            "runpy.run_path(str(sys.argv[1]), run_name='not_main')\n"
            "import omnivoice\n"
            "print(omnivoice.WHICH)\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(driver), str(script_dir / "omnivoice_server.py")],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(site)}, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "installed"



class TestSidecarEncoding:
    def _samples(self):
        np = pytest.importorskip("numpy")
        return np.linspace(-1.0, 1.0, 480, dtype="float32")

    def test_wav_is_a_real_riff_file(self):
        import io
        import wave


        data, media_type = encode_audio(self._samples(), 24000, "wav")
        assert media_type == "audio/wav"
        with wave.open(io.BytesIO(data)) as w:
            assert w.getframerate() == 24000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() == 480

    def test_pcm_is_raw_s16le_at_the_model_rate(self):

        data, media_type = encode_audio(self._samples(), 24000, "pcm")
        assert media_type == "application/octet-stream"
        assert len(data) == 480 * 2  # no container, 16-bit mono

    def test_out_of_range_samples_are_clipped_not_wrapped(self):
        """A sample above 1.0 wrapping through int16 is loud noise, not audio."""
        np = pytest.importorskip("numpy")


        loud = np.array([2.0, -2.0], dtype="float32")
        data, _ = encode_audio(loud, 24000, "pcm")
        assert np.frombuffer(data, dtype="<i2").tolist() == [32767, -32767]



class TestSidecarRequestValidation:
    def _engine(self, tmp_path, **overrides):

        argv = ["--model", "k2-fsa/OmniVoice", "--port", "18091"]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        args = build_parser().parse_args(argv)
        engine = Engine(args, VoiceBook(None))
        engine.model = object()  # never reached by the validation paths below
        return engine

    def test_empty_input_is_a_client_error(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="input"):
            engine.synthesize({"input": "   "})

    def test_unknown_response_format_lists_the_valid_ones(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="response_format"):
            engine.synthesize({"input": "hi", "response_format": "ogg-vorbis"})

    def test_speed_outside_openais_range_is_refused(self, tmp_path):

        engine = self._engine(tmp_path)
        with pytest.raises(BadRequestError, match="speed"):
            engine.synthesize({"input": "hi", "speed": 12.0})

    def test_generation_arguments_are_assembled(self, tmp_path, monkeypatch):
        """Voice cloning wins over a design instruction; both never apply at once."""

        path = tmp_path / "voices.json"
        path.write_text(
            json.dumps({"voices": {"glados": {"instruct": "female", "language": "English"}}}),
            encoding="utf-8",
        )
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--voices", str(path)]
        )
        engine = Engine(args, VoiceBook(str(path)))

        captured = {}

        class FakeModel:
            def generate(self, **kwargs):
                captured.update(kwargs)
                return [[0.0, 0.1]]

        engine.model = FakeModel()
        monkeypatch.setattr(
            _SIDECAR, "encode_audio", lambda samples, rate, fmt: (b"audio", "audio/wav")
        )

        engine.synthesize({"input": "hello", "voice": "glados", "speed": 1.2})

        assert captured["text"] == "hello"
        assert captured["instruct"] == "female"
        assert captured["language"] == "English"
        assert captured["speed"] == 1.2
        assert captured["num_step"] == 32


# ---------------------------------------------------------------------------
# Sidecar startup: warmup and compile
# ---------------------------------------------------------------------------


def _fake_torch(compiled: list[str] | None = None):
    """A stand-in for torch, enough to run the sidecar's load path.

    The real thing is not installed here by design — arc-llama does not depend
    on it — but the ordering and the compile target selection are exactly the
    parts worth pinning down, and both are pure control flow.
    """
    import types

    torch = types.ModuleType("torch")

    class _Dtype:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"torch.{self.name}"

    torch.dtype = _Dtype
    torch.float16 = _Dtype("float16")
    torch.bfloat16 = _Dtype("bfloat16")

    class _Module:
        pass

    torch.nn = types.SimpleNamespace(Module=_Module)
    torch.inference_mode = contextlib.nullcontext

    def _compile(target, **kwargs):
        if compiled is not None:
            compiled.append(type(target).__name__)
        return target

    torch.compile = _compile
    torch.xpu = types.SimpleNamespace(synchronize=lambda: None)
    return torch


class _StubOmniVoice:
    """A model whose submodules are the ones the sidecar compiles."""

    sampling_rate = 24000

    def __init__(self, torch_mod):
        class _LLM(torch_mod.nn.Module):
            pass

        class _Heads(torch_mod.nn.Module):
            pass

        self.llm = _LLM()
        self.audio_heads = _Heads()
        self.generated: list[dict] = []

    def generate(self, **kwargs):
        self.generated.append(kwargs)
        return [[0.0] * 2400]


def _load_engine(monkeypatch, argv_extra=(), model=None, torch_mod=None):
    """Run Engine.load() against stubbed torch/omnivoice modules."""
    import types

    torch_mod = torch_mod or _fake_torch()
    stub = model if model is not None else _StubOmniVoice(torch_mod)
    omnivoice = types.ModuleType("omnivoice")
    omnivoice.OmniVoice = types.SimpleNamespace(from_pretrained=lambda *a, **k: stub)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "omnivoice", omnivoice)

    args = build_parser().parse_args(
        ["--model", "k2-fsa/OmniVoice", "--port", "18091", *argv_extra]
    )
    engine = Engine(args, VoiceBook(None))
    engine.load()
    return engine, stub



class TestSidecarWarmup:
    def test_warmup_runs_before_the_model_is_published(self, monkeypatch):
        """/health must not go green until the first-call cost has been paid.

        `ready` is `self.model is not None`, and the router forwards as soon as
        health passes — so publishing the model first would put lazy kernel
        init back inside a real request, which is what the warmup exists to
        prevent.
        """
        seen_ready: list[bool] = []
        torch_mod = _fake_torch()
        stub = _StubOmniVoice(torch_mod)
        engine_box: list = []

        def _record(**kwargs):
            seen_ready.append(engine_box[0].ready)
            return [[0.0] * 2400]

        stub.generate = _record
        engine, _ = _load_engine(monkeypatch, model=stub, torch_mod=torch_mod)
        engine_box.append(engine)

        # Re-run the load now that the box is populated: the first pass proved
        # it loads at all, this one observes `ready` from inside generate().
        engine.model = None
        engine.load()
        assert seen_ready and not any(seen_ready), "warmup ran after health went green"
        assert engine.ready

    def test_warmup_can_be_turned_off(self, monkeypatch):
        engine, stub = _load_engine(monkeypatch, ["--no-warmup"])
        assert stub.generated == []
        assert engine.ready

    def test_a_failing_warmup_still_serves(self, monkeypatch):
        """A generate() signature we guessed wrong must not cost the backend."""
        torch_mod = _fake_torch()
        stub = _StubOmniVoice(torch_mod)

        def _boom(**kwargs):
            raise TypeError("generate() got an unexpected keyword argument")

        stub.generate = _boom
        engine, _ = _load_engine(monkeypatch, model=stub, torch_mod=torch_mod)
        assert engine.ready



class TestSidecarCompile:
    def test_compiles_submodules_not_the_wrapper(self, monkeypatch):
        """torch.compile(model) is a no-op here: generate() is not forward().

        Compiling the top-level wrapper leaves `self` inside generate() bound
        to the original module, so nothing in the hot loop is ever traced.
        """
        compiled: list[str] = []
        torch_mod = _fake_torch(compiled)
        stub = _StubOmniVoice(torch_mod)
        _load_engine(monkeypatch, ["--compile"], model=stub, torch_mod=torch_mod)
        assert compiled == ["_LLM", "_Heads"]
        assert "_StubOmniVoice" not in compiled

    def test_unknown_targets_are_skipped_not_fatal(self, monkeypatch):
        compiled: list[str] = []
        torch_mod = _fake_torch(compiled)
        stub = _StubOmniVoice(torch_mod)
        engine, _ = _load_engine(
            monkeypatch,
            ["--compile", "--compile-targets", "llm,nope"],
            model=stub, torch_mod=torch_mod,
        )
        assert compiled == ["_LLM"]
        assert engine.ready

    def test_compile_is_off_by_default(self, monkeypatch):
        compiled: list[str] = []
        torch_mod = _fake_torch(compiled)
        _load_engine(monkeypatch, model=_StubOmniVoice(torch_mod), torch_mod=torch_mod)
        assert compiled == []

    def test_dynamic_defaults_to_torchs_automatic_mode(self, monkeypatch):
        """Forcing dynamic=True makes Inductor unable to benchmark extern kernels.

        Symbolic sizes cannot be resolved into the concrete ones a library
        kernel's benchmark request needs, so Inductor logs "Constructing
        input/output tensor meta failed for Extern Choice" per op and falls
        back to empty metadata.
        """
        seen: list = []
        torch_mod = _fake_torch()
        torch_mod.compile = lambda target, **kw: seen.append(kw.get("dynamic")) or target
        _load_engine(
            monkeypatch, ["--compile"], model=_StubOmniVoice(torch_mod), torch_mod=torch_mod
        )
        assert seen == [None, None]

    def test_dynamic_can_be_forced(self, monkeypatch):
        seen: list = []
        torch_mod = _fake_torch()
        torch_mod.compile = lambda target, **kw: seen.append(kw.get("dynamic")) or target
        _load_engine(
            monkeypatch,
            ["--compile", "--compile-dynamic", "true"],
            model=_StubOmniVoice(torch_mod), torch_mod=torch_mod,
        )
        assert seen == [True, True]



class TestSidecarBench:
    def test_sweep_reports_a_row_per_step(self, capsys, tmp_path):
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "The lights are on.",
             "--bench-steps", "8,16", "--bench-runs", "1"]
        )
        engine = Engine(args, VoiceBook(None))
        seen: list[int] = []

        class FakeModel:
            def generate(self, **kwargs):
                seen.append(kwargs["num_step"])
                return [[0.0] * 24000]

        engine.model = FakeModel()
        engine.sampling_rate = 24000

        assert _SIDECAR.run_bench(engine, args) == 0
        out = capsys.readouterr().out
        # One discarded warmup plus one timed run, per step.
        assert seen == [8, 8, 16, 16]
        assert "num_step" in out and "RTF" in out
        for step in ("8", "16"):
            assert any(line.strip().startswith(step) for line in out.splitlines())

    def test_a_failing_step_does_not_abort_the_sweep(self, capsys):
        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "hi",
             "--bench-steps", "8", "--bench-runs", "1"]
        )
        engine = Engine(args, VoiceBook(None))

        class FakeModel:
            def generate(self, **kwargs):
                raise RuntimeError("out of memory")

        engine.model = FakeModel()
        assert _SIDECAR.run_bench(engine, args) == 0
        assert "failed: RuntimeError: out of memory" in capsys.readouterr().out


def _toml_inline(options):
    """Render a dict as a TOML inline table.

    `json.dumps` is right for the *values* (its `true`/`false` and quoted
    strings are valid TOML) but wrong for the table: TOML wants `key = value`,
    not `"key": value`.
    """
    import json as _json

    return ", ".join(f"{k} = {_json.dumps(v)}" for k, v in options.items())


class TestSidecarCompileSweepGuards:
    """A num_step sweep under --compile is a trap, and must say so.

    Dynamo guards on `num_step` because it is an ordinary Python int driving
    the solver loop, so each swept value is a different graph paying a full
    compile — measured at ~9.5 minutes each on Battlemage.
    """

    def _bench(self, extra, steps="8,16"):
        import contextlib
        import io

        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "hi",
             "--bench-steps", steps, "--bench-runs", "1", *extra]
        )
        engine = Engine(args, VoiceBook(None))

        class FakeModel:
            def generate(self, **kwargs):
                return [[0.0] * 24000]

        engine.model = FakeModel()
        engine.sampling_rate = 24000
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _SIDECAR.run_bench(engine, args)
        return buf.getvalue()

    def test_a_multi_step_compiled_sweep_is_flagged(self):
        out = self._bench(["--compile"])
        assert "separate compiles" in out
        assert "--no-compile" in out

    def test_eager_sweeps_say_nothing(self):
        assert "separate compiles" not in self._bench([])

    def test_a_single_step_compiled_run_is_fine(self):
        assert "separate compiles" not in self._bench(["--compile"], steps="32")



class TestInductorNoise:
    def test_the_extern_choice_flood_is_silenced(self):
        """Hundreds of identical lines per compile bury the numbers."""
        import logging as _logging

        logger = _logging.getLogger("torch._inductor.select_algorithm")
        original = logger.level
        try:
            logger.setLevel(_logging.NOTSET)
            _SIDECAR.quiet_inductor_noise()
            assert not logger.isEnabledFor(_logging.WARNING)
            # Real failures must still get through.
            assert logger.isEnabledFor(_logging.ERROR)
        finally:
            logger.setLevel(original)



class TestBenchEncodeReport:
    """Encoding is reported, but only nudged about when it is worth avoiding."""

    def _bench_output(self, fmt, encoder):
        import contextlib
        import io

        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "hi", "--bench-steps", "8",
             "--bench-runs", "1", "--default-response-format", fmt]
        )
        engine = Engine(args, VoiceBook(None))

        class FakeModel:
            def generate(self, **kwargs):
                return [[0.0] * 48000]

        engine.model = FakeModel()
        engine.sampling_rate = 24000
        original = _SIDECAR.encode_audio
        _SIDECAR.encode_audio = encoder
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                _SIDECAR.run_bench(engine, args)
        finally:
            _SIDECAR.encode_audio = original
        return next(
            line for line in buf.getvalue().splitlines()
            if line.startswith("encode")
        )

    def test_a_fast_mp3_encode_is_reported_without_advice(self):
        """~17 ms of libsndfile mp3 is noise, and mp3 is smaller over wifi."""
        line = self._bench_output("mp3", lambda s, r, f: (b"x" * 14336, "audio/mpeg"))
        assert "encode to mp3" in line
        assert "response_format=wav" not in line

    def test_a_slow_encode_suggests_wav(self):
        import time as _time

        def _slow(samples, rate, fmt):
            _time.sleep(_SIDECAR._SLOW_ENCODE_S * 1.5)
            return b"x" * 14336, "audio/mpeg"

        assert "response_format=wav" in self._bench_output("mp3", _slow)

    def test_wav_is_never_told_to_use_wav(self):
        line = self._bench_output("wav", lambda s, r, f: (b"x" * 94208, "audio/wav"))
        assert "response_format=wav" not in line



class TestDynamoRecompileDiagnostic:
    """A compiled module slower than eager is usually re-tracing per call.

    The decoder runs once per solver step, so anything the tracer specialises
    on that varies per step makes every call a fresh graph. Compile counts
    that grow with the step count are that, and the bench should show them
    rather than leave it to be inferred from timings.
    """

    def _fake_torch_with_counters(self, monkeypatch, counters):
        import types

        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        torch.xpu = types.SimpleNamespace(synchronize=lambda: None)
        dynamo = types.ModuleType("torch._dynamo")
        utils = types.ModuleType("torch._dynamo.utils")
        utils.counters = counters
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
        monkeypatch.setitem(sys.modules, "torch._dynamo.utils", utils)

    def test_counters_are_read_defensively(self, monkeypatch):
        """These live under a private module; a rename must not break a run."""
        import types

        broken = types.ModuleType("torch._dynamo.utils")  # no `counters` at all
        monkeypatch.setitem(sys.modules, "torch._dynamo.utils", broken)
        assert _SIDECAR.dynamo_stats() == {}

    def test_growth_per_step_is_reported(self, monkeypatch):
        import contextlib as _contextlib
        import io

        counters = {"frames": {"total": 0}, "stats": {"unique_graphs": 0}}
        self._fake_torch_with_counters(monkeypatch, counters)

        class FakeModel:
            def generate(self, **kwargs):
                # One trace per solver step: the failure mode itself.
                counters["frames"]["total"] += kwargs["num_step"]
                return [[0.0] * 48000]

        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "hi", "--bench-steps", "8",
             "--bench-runs", "1", "--compile", "--default-response-format", "wav"]
        )
        engine = Engine(args, VoiceBook(None))
        engine.model = FakeModel()
        engine.sampling_rate = 24000
        buf = io.StringIO()
        with _contextlib.redirect_stdout(buf):
            _SIDECAR.run_bench(engine, args)
        out = buf.getvalue()
        assert "dynamo during this step" in out
        assert "frames.total=16" in out  # 8 steps x (1 discarded + 1 timed)
        assert "TORCH_LOGS=recompiles" in out

    def test_eager_runs_say_nothing_about_dynamo(self):
        import contextlib as _contextlib
        import io

        args = build_parser().parse_args(
            ["--model", "m", "--port", "1", "--bench", "hi", "--bench-steps", "8",
             "--bench-runs", "1", "--default-response-format", "wav"]
        )
        engine = Engine(args, VoiceBook(None))

        class FakeModel:
            def generate(self, **kwargs):
                return [[0.0] * 48000]

        engine.model = FakeModel()
        engine.sampling_rate = 24000
        buf = io.StringIO()
        with _contextlib.redirect_stdout(buf):
            _SIDECAR.run_bench(engine, args)
        assert "dynamo" not in buf.getvalue()


