"""Optional uploads at the end of a meeting: a recording and PowerPoint slides."""
from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import config, db
from ..services.materials import process_recording, process_slides

router = APIRouter()


@router.post('/api/meetings/{mid}/materials')
def upload_material(mid: int, kind: str = Form(...), file: UploadFile = File(...), replace_live: str = Form('true')):
    if kind not in ('recording', 'slides'):
        raise HTTPException(400, 'kind must be "recording" or "slides"')
    with db.session() as c:
        m = c.execute('SELECT id,started_at FROM meetings WHERE id=?', (mid,)).fetchone()
    if not m:
        raise HTTPException(404, 'Meeting not found')
    name = Path(file.filename or 'upload').name
    suffix = Path(name).suffix.lower()
    if kind == 'slides' and suffix != '.pptx':
        raise HTTPException(400, 'Slides must be a .pptx file. In Google Slides or Keynote, export/download as PowerPoint first.')
    media = config.media_dir()
    media.mkdir(parents=True, exist_ok=True)
    dest = media / f'{mid}_{kind}_{int(time.time() * 1000)}{suffix or ".bin"}'
    with dest.open('wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        if kind == 'recording':
            t0 = datetime.fromisoformat(m['started_at']).timestamp()
            info = process_recording(mid, dest, str(replace_live).lower() != 'false', t0)
            detail = f"{info['minutes']} min · {info['lines']} transcript lines · {info['stills']} stills"
        else:
            detail = f'{process_slides(mid, dest)} slides'
    except Exception as e:
        raise HTTPException(422, f'{kind.title()} processing failed: {e}')
    finally:
        dest.unlink(missing_ok=True)
    with db.session() as c:
        c.execute('INSERT INTO materials(meeting_id,kind,filename,detail,created_at) VALUES(?,?,?,?,?)', (mid, kind, name, detail, db.now()))
    return {'ok': True, 'kind': kind, 'filename': name, 'detail': detail}
