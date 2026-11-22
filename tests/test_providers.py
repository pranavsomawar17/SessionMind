import importlib.machinery
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

from app import providers
from app.providers import groq as groq_mod
from app.providers.base import LLMProvider, ProviderError, clean_segments, make_result, parse_json


# ---------------------------------------------------------------- JSON parsing
@pytest.mark.parametrize('text', [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure! Here is the result:\n{"a": 1}\nHope that helps.',
    '<think>let me think {not json}</think>{"a": 1}',
    '<think>cut off before the answer',   # handled below: no JSON at all
])
def test_parse_json_variants(text):
    if 'cut off' in text:
        with pytest.raises(ProviderError):
            parse_json(text)
    else:
        assert parse_json(text) == {'a': 1}


def test_parse_json_accepts_a_list_and_rejects_prose():
    assert parse_json('[1, 2]') == [1, 2]
    with pytest.raises(ProviderError):
        parse_json('no json here')


# ---------------------------------------------------------------- transcript cleaning
def test_whisper_silence_hallucinations_are_dropped():
    segs = [{'start': 0, 'end': 2, 'text': 'Thanks for watching!'},
            {'start': 2, 'end': 3, 'text': 'you', 'no_speech_prob': 0.9, 'avg_logprob': -1.4},
            {'start': 3, 'end': 5, 'text': 'We ship on Friday.', 'no_speech_prob': 0.02, 'avg_logprob': -0.2},
            {'start': 5, 'end': 6, 'text': '...'}]
    assert [s['text'] for s in clean_segments(segs)] == ['We ship on Friday.']


def test_real_short_answers_survive_and_decoding_loops_do_not():
    kept = clean_segments([{'start': 0, 'end': 1, 'text': 'Okay.', 'no_speech_prob': 0.01, 'avg_logprob': -0.1}])
    assert [s['text'] for s in kept] == ['Okay.']
    loop = [{'start': i, 'end': i + 1, 'text': 'See you next week.'} for i in range(6)]
    assert len(clean_segments(loop)) == 2
    assert make_result('x', [{'start': 0, 'end': 1, 'text': 'Thank you for watching'}], 1)['text'] == ''


def test_chat_json_retries_once_then_raises():
    class Bad(LLMProvider):
        calls = 0

        def _complete(self, *a):
            Bad.calls += 1
            return 'never json'
    with pytest.raises(ProviderError):
        Bad().chat_json('s', 'u')
    assert Bad.calls == 2


# ---------------------------------------------------------------- Ollama, against a real local HTTP server
class _Ollama:
    def __init__(self):
        self.requests, self.responses = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append(('GET', self.path, None))
                self._send(200, {'models': [{'name': 'qwen2.5:7b-instruct'}, {'name': 'llama3.2-vision:latest'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.requests.append(('POST', self.path, body))
                code, payload = outer.responses.pop(0) if outer.responses else (200, {'message': {'content': '{"ok": true}'}})
                self._send(code, payload)
        self.server = HTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def ollama(monkeypatch):
    srv = _Ollama()
    monkeypatch.setenv('OLLAMA_URL', srv.url)
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    yield srv
    srv.close()


def test_ollama_request_has_context_schema_and_options(ollama, monkeypatch):
    monkeypatch.setenv('OLLAMA_NUM_CTX', '8192')
    llm = providers.get_llm()
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
    assert llm.chat_json('sys', 'usr', schema=schema, max_tokens=500) == {'ok': True}
    _, path, body = ollama.requests[-1]
    assert path == '/api/chat' and body['model'] == 'qwen2.5:7b-instruct' and body['stream'] is False
    assert body['format'] == schema
    assert body['options']['num_ctx'] == 8192 and body['options']['num_predict'] == 500
    assert [m['role'] for m in body['messages']] == ['system', 'user']


def test_ollama_falls_back_to_plain_json_mode_on_old_servers(ollama):
    ollama.responses = [(400, {'error': 'invalid format'}), (200, {'message': {'content': '{"ok": 1}'}})]
    assert providers.get_llm().chat_json('s', 'u', schema={'type': 'object'}) == {'ok': 1}
    assert [r[2]['format'] for r in ollama.requests if r[0] == 'POST'] == [{'type': 'object'}, 'json']


def test_ollama_missing_model_tells_you_what_to_run(ollama):
    ollama.responses = [(404, {'error': "model 'qwen2.5:7b-instruct' not found"})]
    with pytest.raises(ProviderError, match='ollama pull qwen2.5:7b-instruct'):
        providers.get_llm().chat_json('s', 'u')


def test_ollama_not_running_is_a_clear_error(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('OLLAMA_URL', 'http://127.0.0.1:9')       # nothing listens here
    with pytest.raises(ProviderError, match='ollama serve'):
        providers.get_llm().chat_json('s', 'u')
    assert providers.get_llm().health()['ok'] is False


def test_ollama_health_checks_the_model_is_installed(ollama, monkeypatch):
    assert providers.get_llm().health()['ok'] is True
    providers.reset()
    monkeypatch.setenv('OLLAMA_MODEL', 'mistral')
    h = providers.get_llm().health()
    assert h['ok'] is False and 'ollama pull mistral' in h['detail']


def test_ollama_context_size_drives_the_input_budget(ollama, monkeypatch):
    monkeypatch.setenv('OLLAMA_NUM_CTX', '32768')
    big = providers.get_llm().max_input_chars
    providers.reset()
    monkeypatch.setenv('OLLAMA_NUM_CTX', '8192')
    assert big > providers.get_llm().max_input_chars >= 6000
    assert providers.get_llm().concurrency == 1


def test_ollama_vision_is_off_unless_a_model_is_set(ollama, monkeypatch):
    assert providers.get_vision().enabled is False and providers.get_vision().read(b'x', 'p') is None
    providers.reset()
    monkeypatch.setenv('OLLAMA_VISION_MODEL', 'llama3.2-vision')
    ollama.responses = [(200, {'message': {'content': 'Priya: shipping Friday'}})]
    v = providers.get_vision()
    assert v.enabled and v.read(b'imagebytes', 'prompt') == 'Priya: shipping Friday'
    assert ollama.requests[-1][2]['messages'][0]['images']


# ---------------------------------------------------------------- Groq (requests mocked)
class _Resp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._body, self.headers = code, body or {}, headers or {}
        self.ok = code < 400
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'key1,key2')
    groq_mod._models.clear()
    sleeps, posts = [], []
    monkeypatch.setattr(groq_mod.time, 'sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr(groq_mod.requests, 'get', lambda *a, **k: _Resp(200, {'data': [{'id': 'llama-3.3-70b-versatile'}]}))
    return types.SimpleNamespace(sleeps=sleeps, posts=posts, mp=monkeypatch)


def _chat(content='{"a": 1}'):
    return {'choices': [{'message': {'content': content}}]}


def test_groq_rotates_to_the_second_key_on_rate_limit(groq):
    def post(url, headers=None, **kw):
        groq.posts.append(headers['Authorization'])
        return _Resp(429, {'error': 'slow down'}) if headers['Authorization'].endswith('key1') else _Resp(200, _chat())
    groq.mp.setattr(groq_mod.requests, 'post', post)
    assert providers.get_llm().chat_json('s', 'u') == {'a': 1}
    assert groq.posts == ['Bearer key1', 'Bearer key2'] and groq.sleeps == []


def test_groq_backs_off_and_retries_when_every_key_is_limited(groq):
    attempts = []

    def post(url, headers=None, **kw):
        attempts.append(1)
        return _Resp(429, {}, {'retry-after': '3'}) if len(attempts) <= 2 else _Resp(200, _chat())
    groq.mp.setattr(groq_mod.requests, 'post', post)
    assert providers.get_llm().chat_json('s', 'u') == {'a': 1}
    assert groq.sleeps == [3.0] and len(attempts) == 3


def test_groq_gives_a_clear_error_without_a_key_or_with_a_bad_one(groq, monkeypatch):
    groq.mp.setattr(groq_mod.requests, 'post', lambda *a, **k: _Resp(401, {'error': 'bad key'}))
    with pytest.raises(ProviderError, match='authentication failed'):
        providers.get_llm().chat_json('s', 'u')
    monkeypatch.delenv('GROQ_API_KEY')
    providers.reset()
    with pytest.raises(ProviderError, match='GROQ_API_KEY'):
        providers.get_llm().chat_json('s', 'u')
    assert providers.get_llm().health()['ok'] is False


def test_groq_model_list_is_cached_not_fetched_per_call(groq):
    gets = []
    groq.mp.setattr(groq_mod.requests, 'get', lambda *a, **k: gets.append(1) or _Resp(200, {'data': [{'id': 'llama-3.3-70b-versatile'}]}))
    groq.mp.setattr(groq_mod.requests, 'post', lambda *a, **k: _Resp(200, _chat()))
    llm = providers.get_llm()
    for _ in range(5):
        llm.chat_json('s', 'u')
    assert len(gets) == 1


def test_groq_stt_returns_normalised_segments_and_sends_language(groq, tmp_path, monkeypatch):
    monkeypatch.setenv('WHISPER_LANGUAGE', 'hi')
    sent = {}

    def post(url, headers=None, files=None, data=None, **kw):
        sent.update(data)
        return _Resp(200, {'text': 'hello', 'duration': 3.0, 'segments': [{'start': 0, 'end': 1, 'text': ' hello ', 'no_speech_prob': 0.01}]})
    groq.mp.setattr(groq_mod.requests, 'post', post)
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    out = providers.get_stt().transcribe(wav)
    assert out['segments'] == [{'start': 0.0, 'end': 1.0, 'text': 'hello'}] and sent['language'] == 'hi'


# ---------------------------------------------------------------- local Whisper (libraries faked)
@pytest.fixture(autouse=True)
def _restore_modules():
    saved = dict(sys.modules)
    yield
    for name in ('whisper', 'torch', 'faster_whisper', 'ctranslate2'):
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)


def _fake_module(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


@pytest.fixture
def wav(tmp_path):
    import wave
    p = tmp_path / 'c.wav'
    with wave.open(str(p), 'wb') as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 16000 * 3)
    return p


def test_openai_whisper_provider_transcribes_and_filters(monkeypatch, wav):
    seen = {}

    class Model:
        def transcribe(self, path, **kw):
            seen.update(kw, path=path)
            return {'segments': [{'start': 0.0, 'end': 2.5, 'text': ' Rahul will test checkout. ', 'no_speech_prob': 0.01, 'avg_logprob': -0.2},
                                 {'start': 2.5, 'end': 3.0, 'text': 'Thanks for watching', 'no_speech_prob': 0.7, 'avg_logprob': -1.5}]}
    loaded = {}
    _fake_module('whisper', load_model=lambda name, device=None: loaded.update(name=name, device=device) or Model())
    _fake_module('torch', cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setenv('STT_PROVIDER', 'whisper')
    monkeypatch.setenv('WHISPER_MODEL', 'base')
    monkeypatch.setenv('WHISPER_LANGUAGE', 'en')
    stt = providers.get_stt()
    assert stt.health()['ok'] and stt.chunk_seconds == 20
    out = stt.transcribe(wav)
    assert out['segments'] == [{'start': 0.0, 'end': 2.5, 'text': 'Rahul will test checkout.'}]
    assert out['duration'] == pytest.approx(3.0) and loaded == {'name': 'base', 'device': 'cpu'}
    assert seen['language'] == 'en' and seen['condition_on_previous_text'] is False and seen['fp16'] is False
    stt.transcribe(wav)
    assert loaded  # model is loaded once and reused


def test_faster_whisper_provider_uses_vad_and_int8_on_cpu(monkeypatch, wav):
    seen = {}

    class Seg:
        def __init__(self, s, e, t):
            self.start, self.end, self.text, self.no_speech_prob, self.avg_logprob = s, e, t, 0.01, -0.2

    class WM:
        def __init__(self, name, device, compute_type):
            seen.update(name=name, device=device, compute=compute_type)

        def transcribe(self, path, **kw):
            seen.update(kw)
            return iter([Seg(0, 1, ' Meera asks finance. ')]), object()
    _fake_module('faster_whisper', WhisperModel=WM)
    _fake_module('ctranslate2', get_cuda_device_count=lambda: 0)
    monkeypatch.setenv('STT_PROVIDER', 'faster-whisper')
    out = providers.get_stt().transcribe(wav)
    assert out['segments'][0]['text'] == 'Meera asks finance.'
    assert seen['device'] == 'cpu' and seen['compute'] == 'int8' and seen['vad_filter'] is True


def test_local_whisper_not_installed_gives_install_instructions(monkeypatch, wav):
    for name in ('whisper', 'faster_whisper'):
        sys.modules.pop(name, None)
    monkeypatch.setattr('importlib.util.find_spec', lambda name: None)
    monkeypatch.setenv('STT_PROVIDER', 'whisper')
    stt = providers.get_stt()
    assert stt.health()['ok'] is False and 'requirements-local.txt' in stt.health()['detail']
    with pytest.raises(ProviderError, match='requirements-local.txt'):
        stt.transcribe(wav)


# ---------------------------------------------------------------- factory
def test_defaults_keep_the_original_cloud_setup():
    assert providers.get_llm().name == 'groq' and providers.get_stt().name == 'groq'
    assert providers.get_vision().name == 'groq'


def test_vision_follows_the_llm_unless_told_otherwise(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    assert providers.get_vision().name == 'ollama'         # local stays local: nothing goes to a cloud service by surprise
    providers.reset()
    monkeypatch.setenv('VISION_PROVIDER', 'none')
    assert providers.get_vision().enabled is False


def test_invalid_provider_name_lists_the_options(monkeypatch):
    monkeypatch.setenv('STT_PROVIDER', 'wispr')
    with pytest.raises(ProviderError, match='faster_whisper'):
        providers.get_stt()
    assert providers.status()['stt']['ok'] is False
