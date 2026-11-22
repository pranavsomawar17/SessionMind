"""Ollama backend: run the analysis (and optionally screenshot reading) on your own machine.

Things that matter for local models, and are handled here:

* ``num_ctx``      Ollama's default context is small and silently truncates long prompts.
                   We always send an explicit value (OLLAMA_NUM_CTX).
* ``format``       Structured output. We pass a JSON schema when the caller has one, which makes
                   small models far more reliable than "please return JSON".
* ``concurrency``  One GPU/CPU cannot serve parallel requests well; calls are serialised by default.
* clear errors     "Ollama is not running" / "model not installed: run ollama pull X".
"""
from __future__ import annotations

import base64

import requests

from .. import config
from .base import LLMProvider, ProviderError, VisionProvider


def base_url() -> str:
    return config.env('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')


def _request(method: str, path: str, **kw) -> requests.Response:
    try:
        return requests.request(method, base_url() + path, **kw)
    except requests.ConnectionError:
        raise ProviderError(f'Cannot reach Ollama at {base_url()}. Start it with `ollama serve` (or open the Ollama app).')
    except requests.Timeout:
        raise ProviderError('Ollama took too long to answer. A larger model on CPU can be slow: raise OLLAMA_TIMEOUT '
                            'or use a smaller model.')


def installed_models() -> list[str]:
    r = _request('GET', '/api/tags', timeout=5)
    if not r.ok:
        raise ProviderError(f'Ollama answered {r.status_code} on /api/tags.')
    return [m.get('name', '') for m in r.json().get('models', [])]


def has_model(name: str, installed: list[str]) -> bool:
    return name in installed or f'{name}:latest' in installed


def _error_text(r: requests.Response) -> str:
    try:
        return str(r.json().get('error') or r.text)[:400]
    except Exception:
        return r.text[:400]


class OllamaLLM(LLMProvider):
    name = 'ollama'

    def __init__(self):
        self.model = config.env('OLLAMA_MODEL', 'qwen2.5:7b-instruct')
        self.num_ctx = config.env_int('OLLAMA_NUM_CTX', 16384)
        self.keep_alive = config.env('OLLAMA_KEEP_ALIVE', '30m')
        self.timeout = config.env_int('OLLAMA_TIMEOUT', 600)
        self.concurrency = config.env_int('LLM_CONCURRENCY', 1)
        # Roughly 3.2 characters per token; leave room for the instructions and the reply.
        self.max_input_chars = config.env_int('LLM_MAX_INPUT_CHARS', max(6000, int((self.num_ctx - 6000) * 3.2)))
        self.scan_window_chars = config.env_int('SCAN_WINDOW_CHARS', 4000)
        self.report_max_tokens = min(4000, max(1500, self.num_ctx // 4))
        super().__init__()

    def _complete(self, system, user, schema, max_tokens):
        payload = {
            'model': self.model,
            'stream': False,
            'format': schema or 'json',
            'keep_alive': self.keep_alive,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'options': {'temperature': 0.1, 'num_ctx': self.num_ctx, 'num_predict': max_tokens},
        }
        r = _request('POST', '/api/chat', json=payload, timeout=self.timeout)
        if r.status_code == 400 and schema is not None:
            # Older Ollama versions only understand format="json".
            payload['format'] = 'json'
            r = _request('POST', '/api/chat', json=payload, timeout=self.timeout)
        if r.status_code == 404:
            raise ProviderError(f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if not r.ok:
            raise ProviderError(f'Ollama error {r.status_code}: {_error_text(r)}')
        return (r.json().get('message') or {}).get('content', '') or ''

    def health(self):
        try:
            if not has_model(self.model, installed_models()):
                return {'ok': False, 'detail': f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"}
        except ProviderError as e:
            return {'ok': False, 'detail': str(e)}
        return {'ok': True, 'detail': f'Ollama at {base_url()}, model {self.model}, context {self.num_ctx}'}


class OllamaVision(VisionProvider):
    name = 'ollama'

    def __init__(self):
        self.model = config.env('OLLAMA_VISION_MODEL')
        self.enabled = bool(self.model)
        self.timeout = config.env_int('OLLAMA_TIMEOUT', 600)

    def read(self, image_bytes, prompt):
        if not self.enabled:
            return None
        try:
            r = _request('POST', '/api/chat', timeout=self.timeout, json={
                'model': self.model, 'stream': False, 'keep_alive': config.env('OLLAMA_KEEP_ALIVE', '30m'),
                'options': {'temperature': 0, 'num_predict': 300},
                'messages': [{'role': 'user', 'content': prompt, 'images': [base64.b64encode(image_bytes).decode()]}]})
        except ProviderError:
            return None
        if not r.ok:
            return None
        text = ((r.json().get('message') or {}).get('content') or '').strip()
        return None if not text or text.upper().startswith('NONE') else text

    def health(self):
        if not self.enabled:
            return {'ok': True, 'detail': 'Off. Set OLLAMA_VISION_MODEL (for example llama3.2-vision) to read captions from screenshots.'}
        try:
            if not has_model(self.model, installed_models()):
                return {'ok': False, 'detail': f"Vision model '{self.model}' is not installed. Run: ollama pull {self.model}"}
        except ProviderError as e:
            return {'ok': False, 'detail': str(e)}
        return {'ok': True, 'detail': f'Ollama vision model {self.model}'}
