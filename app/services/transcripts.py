"""Turning uploaded audio chunks into transcript rows."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .. import config, db
from ..providers import get_stt
from .audio import extract_wav

LIVE_SOURCE = 'live'
GENERIC_SPEAKER = 'Meeting Speaker'


def parse_ts(v, fallback=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


def save_transcript(mid: int, text: str, ts: float, speaker: str = GENERIC_SPEAKER, source: str = LIVE_SOURCE):
    text = (text or '').strip()
    if not text:
        return None
    with db.session() as c:
        cur = c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',
                        (mid, speaker, text, ts, source))
        return dict(c.execute('SELECT * FROM transcript WHERE id=?', (cur.lastrowid,)).fetchone())


def ingest_audio_chunk(mid: int, data: bytes, filename: str | None, started_at: float | None, speaker: str | None) -> dict:
    """Decode one browser audio chunk, transcribe it and store the lines."""
    ext = Path(filename or 'capture.webm').suffix.lower() or '.webm'
    media = config.media_dir()
    media.mkdir(parents=True, exist_ok=True)
    src = media / f'{mid}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}{ext}'
    src.write_bytes(data)
    wav = None
    try:
        wav = extract_wav(src)
        result = get_stt().transcribe(wav)
        base = started_at or time.time() - float(result.get('duration') or 0)
        label = (speaker or '').strip() or GENERIC_SPEAKER
        created = []
        for s in result.get('segments') or []:
            row = save_transcript(mid, s['text'], base + parse_ts(s.get('start')), label)
            if row:
                created.append(row)
        if not created and result.get('text') and not result.get('segments'):
            row = save_transcript(mid, result['text'], base, label)
            if row:
                created.append(row)
        return {'ok': True, 'text': result.get('text', ''), 'segments': created}
    finally:
        for p in (src, wav):
            if p:
                p.unlink(missing_ok=True)
