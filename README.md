# arc-llama-audio

Speech for [arc-llama](https://github.com/offbyonebit/arc-llama): OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/speech`, so one endpoint covers chat,
transcription and synthesis for a client like Home Assistant.

Installed as a plugin — arc-llama discovers it through entry points, and its
core dependency set is unchanged.

```bash
pip install arc-llama-audio          # transcription only
pip install "arc-llama-audio[tts]"   # + speech synthesis (pulls a torch stack)
```

Two plugins ship in the one package, `audio-asr` and `audio-tts`, so a missing
torch takes out synthesis and leaves transcription serving. Disable either with
arc-llama's own switch:

```bash
ARC_LLAMA_PLUGINS=audio-asr arc-llama serve
```

## Speech to text

Runs on the `llama-server` arc-llama already manages — the binary an Arc box
has, the only transcription runtime with a SYCL build, and it inherits the arch
env profiles and device selector.

```bash
arc-llama-audio add ~/models/Qwen3-ASR-0.6B-Q8_0.gguf --name qwen3-asr --alias whisper-1
curl http://127.0.0.1:11437/v1/audio/transcriptions -F model=qwen3-asr -F file=@speech.wav
```

Both shapes clients send are accepted: a multipart upload (Home Assistant, Open
WebUI, the OpenAI SDKs) and a JSON body naming a server-local path.

llama.cpp keeps the audio encoder in a separate `mmproj-*.gguf`. It is
required, not optional — without it `llama-server` loads the model as a plain
text LLM and transcription returns fluent nonsense rather than failing, so
registration refuses to proceed. It is auto-detected when it sits beside the
weights.

> **Qwen3-ASR emits `language English<asr_text>the actual words`** and llama.cpp
> forwards it verbatim ([#26749](https://github.com/ggml-org/llama.cpp/issues/26749)).
> A voice pipeline then tries to match that prefix as part of your command.
> Stripped by default; `--no-strip-markers` keeps it.

## Text to speech

```bash
arc-llama-audio set-python ~/venvs/omnivoice/bin/python
arc-llama-audio add k2-fsa/OmniVoice --task tts --name omnivoice
arc-llama-audio voice add glados --ref-audio ~/voices/glados.wav \
    --ref-text "All right, look. We've both said a lot of things." --alias alloy
```

[OmniVoice](https://github.com/k2-fsa/OmniVoice) is a Python library rather
than a binary, so it runs in a sidecar under its own interpreter. That keeps
torch out of arc-llama's environment and lets stopping the model actually
return its VRAM.

A fine-tuned model needs no voice at all — the speaker is in the weights, and a
clone or design prompt would fight it.

### Latency

Measured on an Arc B50 Pro, ~2 s of audio, eager:

| `num_step` | generate | RTF |
|---|---|---|
| 8 | 0.26 s | 0.13 |
| 32 (default) | 0.96 s | 0.48 |

```bash
arc-llama-audio bench omnivoice --no-compile
```

**Leave `compile` off.** Compiling `llm` cost ~9.5 minutes on that card, and
compiling `audio_heads` made generation 3–4× *slower* — the decoder runs once
per solver step, so anything the tracer specialises on that varies per step
makes every call a fresh graph. An int8 checkpoint measured the same speed as
fp16, so int8 is a VRAM decision rather than a latency one.

## Configuration

Its own `audio.toml`, beside arc-llama's `config.toml`. The core drops unknown
top-level tables (with a warning), so the plugin's schema lives in its own file
and can move without waiting on a core release.

```toml
version = 1
tts_python = "/home/me/venvs/omnivoice/bin/python"

[[audio_models]]
name = "qwen3-asr"
path = "/home/me/models/Qwen3-ASR-0.6B-Q8_0.gguf"
port = 18090
gpu_pci_slot = "0000:03:00.0"
task = "asr"
engine = "llamacpp"
aliases = ["whisper-1"]

[audio_models.recipe]
mmproj = "/home/me/models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
ctx = 4096          # never left to llama.cpp's default of 0 ("use the GGUF's
                    # 65536"), which is ~7 GB of KV for a 1.7B model
```

## Differences from the in-core version

- **Speech backends are always resident.** The core router only evicts models
  from its own registry, so this plugin owns its subprocesses: started on first
  use, stopped when arc-llama stops. That is what you want anyway — evicting a
  sub-gigabyte ASR model would make each utterance cost two cold starts. The
  cost is that the core's VRAM fit guard cannot see this footprint when
  admitting an LLM, so leave headroom.
- **Its own CLI.** A plugin cannot add subcommands to `arc-llama`, hence
  `arc-llama-audio`.
- **`arc-llama scan` may register an ASR GGUF as a chat model.** The skip logic
  was core-side. Qwen3-ASR reports `architecture: qwen3vl`, so metadata alone
  cannot distinguish it; keep ASR weights out of your scan paths, or remove the
  entry afterwards.
