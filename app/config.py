"""Settings. Everything comes from environment variables (loaded from .env).

Values are read when they are needed, not at import time, so tests and
`.env` edits behave predictably.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(BASE / '.env')
except Exception:  # python-dotenv is optional
    pass


def env(name: str, default: str = '') -> str:
    v = os.getenv(name)
    return default if v is None or not v.strip() else v.strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default


def env_list(name: str) -> list[str]:
    return [x.strip() for x in env(name).split(',') if x.strip()]


def data_dir() -> Path:
    return Path(env('DATA_DIR', str(BASE / 'data'))).expanduser()


def media_dir() -> Path:
    return data_dir() / 'media'


def db_path() -> Path:
    return data_dir() / 'sessionmind.db'


def rel_to_base(p: Path) -> str:
    """Path stored in the DB: relative to the project when possible (keeps old databases valid)."""
    try:
        return str(Path(p).resolve().relative_to(BASE))
    except ValueError:
        return str(Path(p).resolve())
