"""Interfaces every provider implements, plus helpers shared by all of them.

SessionMind needs exactly three things from an AI backend:

* ``LLMProvider.chat_json``   - structured analysis (decisions, tasks, questions, report text)
* ``STTProvider.transcribe``  - speech to text
* ``VisionProvider.read``     - read captions / slides out of a screenshot (optional)

Cloud (Groq) and local (Ollama, Whisper) backends implement the same interface,
so the rest of the app never knows which one is in use.
"""
from __future__ import annotations

import json
import re
import threading
import wave
from pathlib import Path


class ProviderError(RuntimeError):
    """A provider could not do its job. The message is shown to the user, so keep it actionable."""


# ---------------------------------------------------------------- JSON handling

_THINK = re.compile(r'<think>.*?(</think>|$)', re.S | re.I)


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply.

    Local models often wrap JSON in prose or ```json fences, and reasoning models
    prepend a <think> block. All of that is skipped.
    """
    t = _THINK.sub('', text or '').strip()
    decoder = json.JSONDecoder()
    tried = 0
    for m in re.finditer(r'[\{\[]', t):
        tried += 1
        if tried > 25:
            break
        try:
            obj, _ = decoder.raw_decode(t[m.start():])
            return obj
        except ValueError:
            continue
    raise ProviderError('The model did not return valid JSON.')


# ---------------------------------------------------------------- LLM

class LLMProvider:
    name = 'llm'
    model = ''
    concurrency = 1            # simultaneous requests this backend should receive
    max_input_chars = 60000    # how much transcript one report call may carry
    scan_window_chars = 6000   # size of one extraction window
    report_max_tokens = 4000

    def __init__(self):
        self._gate = threading.BoundedSemaphore(max(1, self.concurrency))

    def chat_json(self, system: str, user: str, *, schema: dict | None = None, max_tokens: int = 3000):
        """Ask for a JSON reply. Retries once if the reply is not valid JSON."""
        last: Exception | None = None
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                user + '\n\nIMPORTANT: your previous reply was not valid JSON. Reply with ONE JSON object and nothing else.')
            with self._gate:
                text = self._complete(system, prompt, schema, max_tokens)
            try:
                return parse_json(text)
            except ProviderError as e:
                last = e
        raise last  # type: ignore[misc]

    def _complete(self, system: str, user: str, schema: dict | None, max_tokens: int) -> str:
        raise NotImplementedError

    def health(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f'{self.name} ({self.model})' if self.model else self.name


# ---------------------------------------------------------------- speech to text

class STTProvider:
    name = 'stt'
    model = ''
    chunk_seconds = 10  # how long the browser records before uploading a chunk

    def transcribe(self, wav: Path) -> dict:
        """Return ``{'text': str, 'duration': float, 'segments': [{'start','end','text'}]}``."""
        raise NotImplementedError

    def health(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f'{self.name} ({self.model})' if self.model else self.name


# Phrases Whisper-family models invent on silence or background noise.
_HALLUCINATIONS = (
    'thanks for watching', 'thank you for watching', 'please subscribe', 'like and subscribe',
    'subtitles by', 'subtitles made by', 'amara.org', 'transcribed by', 'www.mooji.org',
)
_SHORT_JUNK = {'you', 'thank you', 'thank you.', 'thanks.', 'bye.', 'bye', '.', '...', 'okay.', 'uh', 'um', 'hmm'}


def clean_segments(segments: list[dict]) -> list[dict]:
    """Drop empty segments, silence hallucinations and looping repeats."""
    out: list[dict] = []
    for s in segments or []:
        text = ' '.join(str(s.get('text') or '').split())
        if not text or not re.search(r'\w', text):
            continue
        low = text.lower()
        no_speech = s.get('no_speech_prob')
        logprob = s.get('avg_logprob')
        shaky = (isinstance(no_speech, (int, float)) and no_speech > 0.6) or \
                (isinstance(logprob, (int, float)) and logprob < -1.0)
        if any(h in low for h in _HALLUCINATIONS):
            continue
        if low in _SHORT_JUNK and shaky:
            continue
        if isinstance(no_speech, (int, float)) and no_speech > 0.85 and isinstance(logprob, (int, float)) and logprob < -0.7:
            continue
        # The same line three times in a row is a decoding loop, not speech.
        if len(out) >= 2 and low == out[-1]['text'].lower() == out[-2]['text'].lower():
            continue
        out.append({'start': float(s.get('start') or 0.0), 'end': float(s.get('end') or 0.0), 'text': text})
    return out


def make_result(text: str, segments: list[dict], duration: float | None) -> dict:
    segs = clean_segments(segments)
    if segments and not segs:
        text = ''
    elif segs:
        text = ' '.join(s['text'] for s in segs)
    return {'text': (text or '').strip(), 'duration': float(duration or 0.0), 'segments': segs}


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), 'rb') as w:
            return w.getnframes() / float(w.getframerate() or 16000)
    except Exception:
        return 0.0


# ---------------------------------------------------------------- vision

class VisionProvider:
    name = 'none'
    model = ''
    enabled = False

    def read(self, image_bytes: bytes, prompt: str) -> str | None:
        return None

    def health(self) -> dict:
        return {'ok': True, 'detail': 'Screenshot text reading is off (the audio transcript is still captured).'}

    def describe(self) -> str:
        return f'{self.name} ({self.model})' if self.model else self.name
