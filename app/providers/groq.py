"""Groq cloud backend (OpenAI-compatible API). This was SessionMind's original and only backend."""
from __future__ import annotations

import base64
import logging
import threading
import time

import requests

from .. import config
from .base import LLMProvider, ProviderError, STTProvider, VisionProvider, make_result

log = logging.getLogger('sessionmind.groq')

API = 'https://api.groq.com/openai/v1'
MODEL_CACHE_SECONDS = 600
_models: dict[str, tuple[float, list[str]]] = {}
_models_lock = threading.Lock()

CHAT_PREFERENCE = ('llama-3.3-70b-versatile', 'openai/gpt-oss-120b', 'qwen/qwen3-32b',
                   'llama-3.1-70b-versatile', 'llama-3.1-8b-instant')
VISION_PREFERENCE = ('meta-llama/llama-4-scout-17b-16e-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct',
                     'llama-3.2-90b-vision-preview', 'llama-3.2-11b-vision-preview')


def api_keys() -> list[str]:
    return config.env_list('GROQ_API_KEY') or config.env_list('GROQ_API_KEYS')


def _headers(key: str) -> dict:
    return {'Authorization': f'Bearer {key}'}


def available_models(key: str) -> list[str]:
    """Model ids for this key. Cached, so we do not hit /models on every request."""
    now = time.time()
    with _models_lock:
        hit = _models.get(key)
        if hit and now - hit[0] < MODEL_CACHE_SECONDS:
            return hit[1]
    try:
        r = requests.get(f'{API}/models', headers={**_headers(key), 'accept': 'application/json'}, timeout=20)
        ids = [x.get('id', '') for x in r.json().get('data', []) if x.get('id')] if r.ok else []
    except Exception:
        ids = []
    with _models_lock:
        _models[key] = (now, ids)
    return ids


def _retry_after(r: requests.Response) -> float:
    try:
        return float(r.headers.get('retry-after', 0))
    except (TypeError, ValueError):
        return 0.0


def _call_with_keys(fn, what: str):
    """Run ``fn(key) -> Response`` across the configured keys.

    Rate limits (429) and server errors (5xx) are retried with backoff, rotating to the
    next key first. Auth errors move on to the next key. Anything else raises.
    """
    keys = api_keys()
    if not keys:
        raise ProviderError('GROQ_API_KEY is not set. Add it to .env, or switch to local models (see .env.example).')
    last = ''
    for rnd in range(3):
        retry = False
        delay = 0.0
        for key in keys:
            try:
                r = fn(key)
            except requests.RequestException as e:
                last, retry = f'Groq {what}: network error: {e}', True
                continue
            if r.status_code in (401, 403):
                last = f'Groq authentication failed ({r.status_code}). Check GROQ_API_KEY. {r.text[:200]}'
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last, retry = f'Groq {what}: rate limit or server error ({r.status_code}). {r.text[:200]}', True
                delay = max(delay, _retry_after(r))
                continue
            if not r.ok:
                try:
                    detail = r.json().get('error', {}).get('message') or r.text
                except Exception:
                    detail = r.text
                last = f'Groq {what} error {r.status_code}: {detail[:400]}'
                continue
            return r
        if not retry or rnd == 2:
            break
        wait = min(25.0, delay or 2.0 * (2 ** rnd))
        log.warning('Groq %s throttled; waiting %.0fs before retrying', what, wait)
        time.sleep(wait)
    raise ProviderError(last or f'Groq {what} failed')


class GroqLLM(LLMProvider):
    name = 'groq'
    concurrency = 2
    max_input_chars = 80000
    scan_window_chars = 6000
    report_max_tokens = 5000

    def __init__(self):
        self.model = config.env('GROQ_CHAT_MODEL', 'llama-3.3-70b-versatile')
        self.max_input_chars = config.env_int('LLM_MAX_INPUT_CHARS', self.max_input_chars)
        self.scan_window_chars = config.env_int('SCAN_WINDOW_CHARS', self.scan_window_chars)
        super().__init__()

    def _model_for(self, key: str) -> str:
        ids = available_models(key)
        if not ids or self.model in ids:
            return self.model
        for m in CHAT_PREFERENCE:
            if m in ids:
                return m
        blocked = ('whisper', 'embed', 'guard', 'safeguard', 'compound')
        text = [x for x in ids if not any(b in x.lower() for b in blocked)]
        return text[0] if text else self.model

    def _complete(self, system, user, schema, max_tokens):
        def call(key):
            payload = {
                'model': self._model_for(key),
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'temperature': 0.1,
                'max_tokens': max_tokens,
                'response_format': {'type': 'json_object'},
            }
            return requests.post(f'{API}/chat/completions', headers={**_headers(key), 'Content-Type': 'application/json'},
                                 json=payload, timeout=300)

        r = _call_with_keys(call, 'analysis')
        return (r.json().get('choices') or [{}])[0].get('message', {}).get('content', '') or ''

    def health(self):
        if not api_keys():
            return {'ok': False, 'detail': 'GROQ_API_KEY is not set in .env.'}
        return {'ok': True, 'detail': f'Groq cloud, model {self.model}'}


class GroqSTT(STTProvider):
    name = 'groq'
    chunk_seconds = 10

    def __init__(self):
        self.model = config.env('GROQ_TRANSCRIPTION_MODEL', 'whisper-large-v3-turbo')
        self.chunk_seconds = config.env_int('CHUNK_SECONDS', 10)

    def transcribe(self, wav):
        data = {'model': self.model, 'response_format': 'verbose_json', 'temperature': '0'}
        if config.env('WHISPER_LANGUAGE'):
            data['language'] = config.env('WHISPER_LANGUAGE')
        if config.env('WHISPER_PROMPT'):
            data['prompt'] = config.env('WHISPER_PROMPT')[:800]

        def call(key):
            with wav.open('rb') as f:
                return requests.post(f'{API}/audio/transcriptions', headers=_headers(key),
                                     files={'file': (wav.name, f, 'audio/wav')}, data=data, timeout=180)

        j = _call_with_keys(call, 'transcription').json()
        return make_result(j.get('text', ''), j.get('segments') or [], j.get('duration'))

    def health(self):
        if not api_keys():
            return {'ok': False, 'detail': 'GROQ_API_KEY is not set in .env.'}
        return {'ok': True, 'detail': f'Groq cloud, model {self.model}'}


class GroqVision(VisionProvider):
    name = 'groq'
    enabled = True

    def __init__(self):
        self.model = config.env('GROQ_VISION_MODEL')

    def _model_for(self, key):
        if self.model:
            return self.model
        ids = available_models(key)
        for m in VISION_PREFERENCE:
            if m in ids:
                return m
        for i in ids:
            if any(w in i.lower() for w in ('vision', 'scout', 'maverick')):
                return i
        return None

    def read(self, image_bytes, prompt):
        b64 = base64.b64encode(image_bytes).decode()
        for key in api_keys():
            model = self._model_for(key)
            if not model:
                continue
            try:
                r = requests.post(
                    f'{API}/chat/completions', headers={**_headers(key), 'Content-Type': 'application/json'}, timeout=60,
                    json={'model': model, 'temperature': 0, 'max_tokens': 300,
                          'messages': [{'role': 'user', 'content': [
                              {'type': 'text', 'text': prompt},
                              {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}]}]})
                if not r.ok:
                    log.info('Groq vision returned %s; skipping this screenshot', r.status_code)
                    continue
                text = ((r.json().get('choices') or [{}])[0].get('message', {}).get('content') or '').strip()
                return None if not text or text.upper() == 'NONE' else text
            except Exception as e:
                log.info('Groq vision failed: %s', e)
        return None

    def health(self):
        return {'ok': bool(api_keys()), 'detail': 'Groq cloud vision (best effort)' if api_keys() else 'GROQ_API_KEY is not set.'}
