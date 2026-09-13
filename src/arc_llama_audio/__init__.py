"""Speech-to-text and text-to-speech for arc-llama, as a plugin.

Nothing heavy is imported here: the entry points name ``arc_llama_audio.asr``
and ``arc_llama_audio.tts_plugin``, and the torch stack a TTS engine needs is
imported inside its sidecar process, never in arc-llama's.
"""

__version__ = "0.1.0"
