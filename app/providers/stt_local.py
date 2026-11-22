"""Speech to text on your own machine.

``STT_PROVIDER=whisper``         the official ``openai-whisper`` package (needs PyTorch).
``STT_PROVIDER=faster_whisper``  the ``faster-whisper`` package (CTranslate2). Same models and
                                 accuracy, usually several times faster, much lighter install.

Both load the model once (lazily, on first use) and transcribe one chunk at a time:
a Whisper model instance is not safe to share between threads.
"""
from __future__ import annotations

import importlib.util
import logging
import threading

from .. import config
from .base import ProviderError, STTProvider, make_result, wav_seconds

log = logging.getLogger('sessionmind.whisper')


class _LocalWhisper(STTProvider):
    package = ''
    default_model = 'small'

    def __init__(self):
        self.model = config.env('WHISPER_MODEL', self.default_model)
        self.language = config.env('WHISPER_LANGUAGE') or None
        self.prompt = config.env('WHISPER_PROMPT') or None
        self.device_setting = config.env('WHISPER_DEVICE', 'auto').lower()
        # Longer chunks give Whisper more context and cost less overhead; local models also
        # need time per chunk, so the default is longer than the cloud one.
        self.chunk_seconds = config.env_int('CHUNK_SECONDS', 20)
        self._engine = None
        self._device = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def installed(self) -> bool:
        return importlib.util.find_spec(self.package) is not None

    def _require(self):
        if not self.installed():
            raise ProviderError(f'The "{self.package}" package is not installed. Run: pip install -r requirements-local.txt')

    def _engine_ready(self):
        with self._load_lock:
            if self._engine is None:
                self._require()
                log.info('Loading Whisper model "%s" (first use can take a minute)...', self.model)
                self._engine = self._load()
                log.info('Whisper model loaded on %s', self._device)
            return self._engine

    def _load(self):
        raise NotImplementedError

    def _run(self, engine, wav):
        raise NotImplementedError

    def transcribe(self, wav):
        engine = self._engine_ready()
        with self._run_lock:
            try:
                segments = self._run(engine, wav)
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(f'Local Whisper failed: {e}')
        return make_result('', segments, wav_seconds(wav))

    def health(self):
        if not self.installed():
            return {'ok': False, 'detail': f'"{self.package}" is not installed. Run: pip install -r requirements-local.txt'}
        return {'ok': True, 'detail': f'{self.package}, model {self.model} (loaded on first use)'}


class WhisperSTT(_LocalWhisper):
    """The official OpenAI Whisper library."""
    name = 'whisper'
    package = 'whisper'

    def _load(self):
        import whisper  # type: ignore

        device = self.device_setting
        if device == 'auto':
            try:
                import torch  # type: ignore
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                device = 'cpu'
        self._device = device
        return whisper.load_model(self.model, device=device)

    def _run(self, engine, wav):
        result = engine.transcribe(
            str(wav),
            language=self.language,
            task='transcribe',
            fp16=(self._device == 'cuda'),
            temperature=0.0,
            condition_on_previous_text=False,   # stops one bad chunk from poisoning the next
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            initial_prompt=self.prompt,
            verbose=None,
        )
        return result.get('segments') or []


class FasterWhisperSTT(_LocalWhisper):
    """faster-whisper (CTranslate2): same models, faster, built-in silence filter."""
    name = 'faster_whisper'
    package = 'faster_whisper'

    def _load(self):
        from faster_whisper import WhisperModel  # type: ignore

        device = self.device_setting
        if device == 'auto':
            try:
                import ctranslate2  # type: ignore
                device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
            except Exception:
                device = 'cpu'
        self._device = device
        compute = config.env('WHISPER_COMPUTE_TYPE') or ('float16' if device == 'cuda' else 'int8')
        return WhisperModel(self.model, device=device, compute_type=compute)

    def _run(self, engine, wav):
        segments, _info = engine.transcribe(
            str(wav),
            language=self.language,
            beam_size=5,
            vad_filter=True,                    # skip silence: the main defence against hallucinated text
            condition_on_previous_text=False,
            initial_prompt=self.prompt,
        )
        return [{'start': s.start, 'end': s.end, 'text': s.text,
                 'no_speech_prob': getattr(s, 'no_speech_prob', None),
                 'avg_logprob': getattr(s, 'avg_logprob', None)} for s in segments]
