"""Screenshots: store the frame, and (best effort) read captions or slide text out of it."""
from __future__ import annotations

import logging
import time

from .. import config, db
from ..providers import get_vision
from .transcripts import save_transcript

log = logging.getLogger('sessionmind.screenshots')

VISION_PROMPT = (
    'This image is a screenshot of a live video-call screen (e.g. Zoom or Google Meet). '
    'If live captions/subtitles are visible, transcribe them exactly, including the speaker '
    'name if one is labeled next to the caption. Otherwise, if a presentation slide or shared '
    'document with meaningful text is visible, extract its key heading/text briefly. '
    'Otherwise reply with exactly: NONE. Reply with only the transcription or extracted text, '
    'nothing else - no preamble, no quotes.')


def store_screenshot(mid: int, data: bytes, ts: float | None):
    ts = ts or time.time()
    media = config.media_dir()
    media.mkdir(parents=True, exist_ok=True)
    p = media / f'{mid}_{int(ts * 1000)}.jpg'
    p.write_bytes(data)
    with db.session() as c:
        cur = c.execute('INSERT INTO screenshots(meeting_id,path,timestamp) VALUES(?,?,?)', (mid, config.rel_to_base(p), ts))
        row = dict(c.execute('SELECT * FROM screenshots WHERE id=?', (cur.lastrowid,)).fetchone())
    # A vision failure must never break capture.
    try:
        vision = get_vision()
        if vision.enabled:
            text = vision.read(data, VISION_PROMPT)
            if text:
                save_transcript(mid, text, ts, 'Screen', source='vision')
    except Exception as e:
        log.info('Screenshot reading skipped: %s', e)
    return row
