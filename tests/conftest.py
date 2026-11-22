import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, providers  # noqa: E402,F401  (imported first so .env is loaded before the fixture clears it)

_PREFIXES = ('GROQ', 'OLLAMA', 'LLM_', 'STT_', 'VISION_', 'WHISPER', 'CHUNK_', 'SCAN_', 'LIVE_')


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets an empty data dir and a clean provider configuration."""
    for k in list(os.environ):
        if k.startswith(_PREFIXES):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    providers.reset()
    yield
    providers.reset()
