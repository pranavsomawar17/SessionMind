"""SQLite access. The schema is created once per database file, not on every call."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = '''
CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meetings(
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL,
    started_at TEXT NOT NULL, ended_at TEXT, report TEXT,
    last_analyzed_ts REAL DEFAULT 0, last_analyzed_id INTEGER DEFAULT 0, current_topic TEXT);
CREATE TABLE IF NOT EXISTS transcript(
    id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, speaker TEXT NOT NULL, text TEXT NOT NULL,
    timestamp REAL NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS screenshots(id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, path TEXT NOT NULL, timestamp REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, type TEXT NOT NULL, content TEXT NOT NULL,
    assignee TEXT, deadline TEXT, timestamp REAL NOT NULL, confidence REAL NOT NULL, status TEXT NOT NULL,
    evidence TEXT DEFAULT '', detail TEXT);
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER UNIQUE NOT NULL, html_path TEXT, pdf_path TEXT,
    docx_path TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS slides(id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, slide_no INTEGER NOT NULL, title TEXT, text TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS materials(id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id INTEGER NOT NULL, kind TEXT NOT NULL, filename TEXT, detail TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_transcript_meeting ON transcript(meeting_id, id);
CREATE INDEX IF NOT EXISTS idx_events_meeting ON events(meeting_id, id);
'''

# Columns added after the first release. Applied once to databases created by older versions.
MIGRATIONS = (
    ('meetings', 'last_analyzed_ts', 'REAL DEFAULT 0'),
    ('meetings', 'last_analyzed_id', 'INTEGER DEFAULT 0'),
    ('meetings', 'current_topic', 'TEXT'),
    ('reports', 'docx_path', 'TEXT'),
    ('events', 'detail', 'TEXT'),
)

_lock = threading.Lock()
_ready: set[str] = set()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create tables / apply migrations. Runs once per database file per process."""
    path = str(config.db_path())
    with _lock:
        if path in _ready:
            return
        config.data_dir().mkdir(parents=True, exist_ok=True)
        config.media_dir().mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(path, timeout=30)
        try:
            c.execute('PRAGMA journal_mode=WAL')  # readers don't block the audio writers
            c.executescript(SCHEMA)
            for table, col, decl in MIGRATIONS:
                have = {r[1] for r in c.execute(f'PRAGMA table_info({table})')}
                if col not in have:
                    c.execute(f'ALTER TABLE {table} ADD COLUMN {col} {decl}')
            c.commit()
        finally:
            c.close()
        _ready.add(path)


def connect() -> sqlite3.Connection:
    init_db()
    c = sqlite3.connect(str(config.db_path()), timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA busy_timeout=30000')
    return c


@contextmanager
def session():
    """`with db.session() as c:` commits on success, rolls back on error, always closes."""
    c = connect()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
