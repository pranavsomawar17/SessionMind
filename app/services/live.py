"""Live insights: analyse new transcript lines while the meeting is running.

Uses the same extraction engine as the final report, so what you see live matches what the
report finds. Differences from the old version:

* The cursor is the transcript row id, not a timestamp. Audio chunks upload concurrently, so a
  slow chunk can arrive with an older timestamp than lines already analysed; a timestamp cursor
  skipped those lines forever.
* The few lines before the new ones are included as context.
* Only one analysis runs per meeting at a time (important for a local model that needs several
  seconds per call); a poll that arrives while one is running just returns the current events.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime

from .. import config, db
from ..report_render import is_unknown, same_text
from .extraction import extract_window, make_windows

log = logging.getLogger('sessionmind.live')

_locks: dict[int, threading.Lock] = {}
_guard = threading.Lock()
CONTEXT_LINES = 4


def _lock_for(mid: int) -> threading.Lock:
    with _guard:
        return _locks.setdefault(mid, threading.Lock())


def _events(c, mid: int) -> list[dict]:
    return [dict(x) for x in c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY id', (mid,)).fetchall()]


def _to_row(x, t0: float) -> dict:
    return {'id': x['id'], 'abs': x['timestamp'], 'timestamp': max(0.0, x['timestamp'] - t0),
            'speaker': x['speaker'], 'text': x['text']}


def _is_duplicate(kind: str, content: str, owner: str, existing: list[dict]) -> bool:
    for e in existing:
        if e['type'] != kind or not same_text(content, e['content']):
            continue
        a, b = (owner or '').strip().lower(), (e.get('assignee') or '').strip().lower()
        if kind == 'task' and a and b and not is_unknown(a) and not is_unknown(b) and a != b:
            continue
        return True
    return False


def analyze(mid: int, llm) -> dict:
    lock = _lock_for(mid)
    if not lock.acquire(blocking=False):
        with db.session() as c:
            m = c.execute('SELECT current_topic FROM meetings WHERE id=?', (mid,)).fetchone()
            return {'events': _events(c, mid), 'current_topic': m['current_topic'] if m else None, 'new': 0, 'busy': True}
    try:
        return _analyze(mid, llm)
    finally:
        lock.release()


def _analyze(mid: int, llm) -> dict:
    with db.session() as c:
        m = c.execute('SELECT * FROM meetings WHERE id=?', (mid,)).fetchone()
        if not m:
            raise LookupError('Meeting not found')
        cursor = m['last_analyzed_id'] or 0
        new = c.execute('SELECT * FROM transcript WHERE meeting_id=? AND id>? ORDER BY timestamp,id', (mid, cursor)).fetchall()
        prev = c.execute('SELECT * FROM transcript WHERE meeting_id=? AND id<=? ORDER BY timestamp DESC,id DESC LIMIT ?',
                         (mid, cursor, CONTEXT_LINES)).fetchall()[::-1]
        existing = _events(c, mid)
        topic = m['current_topic']
    if not new:
        return {'events': existing, 'current_topic': topic, 'new': 0, 'busy': False}

    try:
        t0 = datetime.fromisoformat(m['started_at']).timestamp()
    except Exception:
        t0 = new[0]['timestamp']
    rows = [_to_row(x, t0) for x in list(prev) + list(new)]
    windows = make_windows(rows, llm.scan_window_chars, start_at=len(prev))[:config.env_int('LIVE_MAX_WINDOWS', 2)]
    known = [f"{e['type']}: {e['content']}" for e in existing if e['status'] != 'dismissed']

    found, done_upto = [], len(prev)
    for i, win in enumerate(windows):
        try:
            r = extract_window(llm, rows, win, known=known)
        except Exception:
            if i == 0:
                raise          # nothing processed: let the caller report it; the cursor stays put
            log.warning('Live analysis stopped after %d window(s)', i, exc_info=True)
            break
        found.append(r)
        done_upto = win.end
        topic = r.topic or topic
        known += [f"decision: {x['decision']}" for x in r.decisions] + [f"task: {x['task']}" for x in r.actions] \
            + [f"question: {x['question']}" for x in r.questions]

    new_cursor = max(rows[i]['id'] for i in range(len(prev), done_upto)) if done_upto > len(prev) else cursor
    with db.session() as c:
        c.execute('UPDATE meetings SET last_analyzed_id=?, current_topic=? WHERE id=?', (new_cursor, topic, mid))
        for r in found:
            for kind, items, key in (('decision', r.decisions, 'decision'), ('task', r.actions, 'task'), ('question', r.questions, 'question')):
                for it in items:
                    content = it[key]
                    if _is_duplicate(kind, content, it.get('owner', ''), existing):
                        continue
                    ts = it.get('abs_ts') or rows[done_upto - 1]['abs']
                    detail = {'status': it.get('status', ''), 'raised_by': it.get('raised_by', ''), 'clock': it.get('timestamp', '')}
                    c.execute('INSERT INTO events(meeting_id,type,content,assignee,deadline,timestamp,confidence,status,evidence,detail) '
                              'VALUES(?,?,?,?,?,?,?,?,?,?)',
                              (mid, kind, content, it.get('owner') if kind == 'task' else None,
                               it.get('deadline') if kind == 'task' else None, ts,
                               0.8 if it.get('abs_ts') else 0.6, 'detected', it.get('evidence', ''), json.dumps(detail, ensure_ascii=False)))
                    existing.append({'type': kind, 'content': content, 'assignee': it.get('owner')})
        events = _events(c, mid)
    return {'events': events, 'current_topic': topic, 'new': done_upto - len(prev), 'busy': False}
