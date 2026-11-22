"""Pick the AI backends from .env.

    LLM_PROVIDER     groq | ollama
    STT_PROVIDER     groq | whisper | faster_whisper
    VISION_PROVIDER  auto | groq | ollama | none     (auto follows LLM_PROVIDER)

The three are independent, so mixing is fine (for example local Whisper + Groq analysis).
"""
from __future__ import annotations

import threading

from .. import config
from .base import LLMProvider, ProviderError, STTProvider, VisionProvider

_cache: dict[str, object] = {}
_lock = threading.Lock()

_LLM_ALIASES = {'groq': 'groq', 'ollama': 'ollama', 'local': 'ollama'}
_STT_ALIASES = {'groq': 'groq', 'whisper': 'whisper', 'openai-whisper': 'whisper', 'openai_whisper': 'whisper',
                'local': 'whisper', 'faster_whisper': 'faster_whisper', 'faster-whisper': 'faster_whisper',
                'fasterwhisper': 'faster_whisper'}
_VISION_ALIASES = {'auto': 'auto', 'groq': 'groq', 'ollama': 'ollama', 'none': 'none', 'off': 'none'}


def _choice(var: str, default: str, aliases: dict[str, str]) -> str:
    raw = config.env(var, default).lower()
    if raw not in aliases:
        raise ProviderError(f'{var}="{raw}" is not valid. Choose one of: {", ".join(sorted(set(aliases.values())))}.')
    return aliases[raw]


def llm_choice() -> str:
    return _choice('LLM_PROVIDER', 'groq', _LLM_ALIASES)


def stt_choice() -> str:
    return _choice('STT_PROVIDER', 'groq', _STT_ALIASES)


def vision_choice() -> str:
    c = _choice('VISION_PROVIDER', 'auto', _VISION_ALIASES)
    return llm_choice() if c == 'auto' else c


def get_llm() -> LLMProvider:
    with _lock:
        if 'llm' not in _cache:
            if llm_choice() == 'ollama':
                from .ollama import OllamaLLM
                _cache['llm'] = OllamaLLM()
            else:
                from .groq import GroqLLM
                _cache['llm'] = GroqLLM()
        return _cache['llm']  # type: ignore[return-value]


def get_stt() -> STTProvider:
    with _lock:
        if 'stt' not in _cache:
            c = stt_choice()
            if c == 'whisper':
                from .stt_local import WhisperSTT
                _cache['stt'] = WhisperSTT()
            elif c == 'faster_whisper':
                from .stt_local import FasterWhisperSTT
                _cache['stt'] = FasterWhisperSTT()
            else:
                from .groq import GroqSTT
                _cache['stt'] = GroqSTT()
        return _cache['stt']  # type: ignore[return-value]


def get_vision() -> VisionProvider:
    with _lock:
        if 'vision' not in _cache:
            c = vision_choice()
            if c == 'ollama':
                from .ollama import OllamaVision
                _cache['vision'] = OllamaVision()
            elif c == 'groq':
                from .groq import GroqVision
                _cache['vision'] = GroqVision()
            else:
                _cache['vision'] = VisionProvider()
        return _cache['vision']  # type: ignore[return-value]


def reset() -> None:
    """Forget cached providers (used by tests)."""
    with _lock:
        _cache.clear()


def status() -> dict:
    """What the UI shows: one entry per provider with ok / detail."""
    out = {}
    for key, getter in (('llm', get_llm), ('stt', get_stt), ('vision', get_vision)):
        try:
            p = getter()
            h = p.health()
            out[key] = {'provider': p.name, 'model': p.model, 'ok': bool(h.get('ok')), 'detail': h.get('detail', '')}
        except ProviderError as e:
            out[key] = {'provider': 'unknown', 'model': '', 'ok': False, 'detail': str(e)}
    return out
