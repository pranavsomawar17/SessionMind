"""Projects, meetings, live audio / screenshots, live insights and ending a meeting."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import db
from ..providers import get_llm
from ..providers.base import ProviderError
from ..services import live
from ..services.report import build_report, hydrate_report
from ..services.screenshots import store_screenshot
from ..services.transcripts import ingest_audio_chunk

log = logging.getLogger('sessionmind.routes')
router = APIRouter()


class ProjectIn(BaseModel):
    name: str


class MeetingIn(BaseModel):
    project_id: int
    title: str = 'Meeting'


class EventStatus(BaseModel):
    status: str


@router.get('/api/projects')
def projects():
    with db.session() as c:
        return [dict(x) for x in c.execute('SELECT * FROM projects ORDER BY id DESC').fetchall()]


@router.post('/api/projects')
def create_project(x: ProjectIn):
    name = x.name.strip() or 'My Project'
    with db.session() as c:
        cur = c.execute('INSERT INTO projects(name,created_at) VALUES(?,?)', (name, db.now()))
        return dict(c.execute('SELECT * FROM projects WHERE id=?', (cur.lastrowid,)).fetchone())


@router.get('/api/projects/{pid}/meetings')
def meetings(pid: int):
    with db.session() as c:
        rows = c.execute('SELECT id,title,status,started_at,ended_at FROM meetings WHERE project_id=? ORDER BY id DESC', (pid,)).fetchall()
        return [dict(x) for x in rows]


@router.post('/api/meetings')
def create_meeting(x: MeetingIn):
    with db.session() as c:
        if not c.execute('SELECT id FROM projects WHERE id=?', (x.project_id,)).fetchone():
            raise HTTPException(404, 'Project not found')
        cur = c.execute('INSERT INTO meetings(project_id,title,status,started_at) VALUES(?,?,?,?)',
                        (x.project_id, x.title.strip() or 'Meeting', 'live', db.now()))
        return dict(c.execute('SELECT * FROM meetings WHERE id=?', (cur.lastrowid,)).fetchone())


@router.get('/api/meetings/{mid}')
def get_meeting(mid: int):
    with db.session() as c:
        m = c.execute('SELECT * FROM meetings WHERE id=?', (mid,)).fetchone()
        if not m:
            raise HTTPException(404, 'Meeting not found')
        t = c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp', (mid,)).fetchall()
        s = c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp', (mid,)).fetchall()
    report = None
    if m['report']:
        try:
            report = hydrate_report(m['report'])
        except Exception:
            report = None
    return {**dict(m), 'transcript': [dict(x) for x in t], 'screenshots': [dict(x) for x in s], 'report_data': report}


# Plain `def` (not async): FastAPI runs these in worker threads, so decoding and transcribing
# one chunk does not block every other request.
@router.post('/api/meetings/{mid}/audio')
def audio(mid: int, audio: UploadFile = File(...), started_at: float | None = Form(None), speaker: str | None = Form(None)):
    with db.session() as c:
        m = c.execute('SELECT status FROM meetings WHERE id=?', (mid,)).fetchone()
    if not m:
        raise HTTPException(404, 'Meeting not found')
    if m['status'] != 'live':
        raise HTTPException(409, 'Meeting is not live')
    data = audio.file.read()
    if len(data) < 4000:
        return {'ok': True, 'ignored': True, 'reason': 'small audio'}
    try:
        return ingest_audio_chunk(mid, data, audio.filename, started_at, speaker)
    except Exception as e:
        log.warning('Audio chunk failed: %s', e)
        raise HTTPException(422, f'Audio processing error: {e}')


@router.post('/api/meetings/{mid}/screenshot')
def screenshot(mid: int, image: UploadFile = File(...), timestamp: float | None = Form(None)):
    with db.session() as c:
        if not c.execute('SELECT id FROM meetings WHERE id=?', (mid,)).fetchone():
            raise HTTPException(404, 'Meeting not found')
    return store_screenshot(mid, image.file.read(), timestamp)


@router.post('/api/meetings/{mid}/analyze')
def analyze(mid: int):
    """Live insights: analyse transcript lines added since the last call."""
    try:
        return live.analyze(mid, get_llm())
    except LookupError:
        raise HTTPException(404, 'Meeting not found')
    except ProviderError as e:
        raise HTTPException(502, f'Live analysis failed: {e}')
    except Exception as e:
        log.exception('Live analysis crashed')
        raise HTTPException(502, f'Live analysis failed: {e}')


@router.get('/api/meetings/{mid}/events')
def list_events(mid: int):
    with db.session() as c:
        if not c.execute('SELECT id FROM meetings WHERE id=?', (mid,)).fetchone():
            raise HTTPException(404, 'Meeting not found')
        return [dict(x) for x in c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY id', (mid,)).fetchall()]


@router.patch('/api/meetings/{mid}/events/{eid}')
def update_event(mid: int, eid: int, x: EventStatus):
    if x.status not in ('detected', 'confirmed', 'dismissed'):
        raise HTTPException(400, 'Invalid status')
    with db.session() as c:
        cur = c.execute('UPDATE events SET status=? WHERE id=? AND meeting_id=?', (x.status, eid, mid))
        if cur.rowcount == 0:
            raise HTTPException(404, 'Event not found')
        return dict(c.execute('SELECT * FROM events WHERE id=?', (eid,)).fetchone())


@router.post('/api/meetings/{mid}/end')
def end_meeting(mid: int):
    with db.session() as c:
        if not c.execute('SELECT id FROM meetings WHERE id=?', (mid,)).fetchone():
            raise HTTPException(404, 'Meeting not found')
        c.execute('UPDATE meetings SET status="ended",ended_at=? WHERE id=?', (db.now(), mid))
    try:
        return build_report(mid)
    except Exception as e:
        # The meeting stays ended; show a useful message instead of an opaque 500.
        log.exception('Report generation failed')
        raise HTTPException(502, f'Report generation failed: {e}')
