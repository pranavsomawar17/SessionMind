"""Optional materials: a recorded meeting (audio/video) and PowerPoint slides."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from .. import config, db
from ..providers import get_stt
from .audio import ffprobe_duration, run_ff
from .transcripts import parse_ts

log = logging.getLogger('sessionmind.materials')

SEGMENT_SECONDS = 480  # 8-minute pieces keep every upload far below cloud file-size limits


def process_recording(mid: int, src: Path, replace_live: bool, t0: float) -> dict:
    """Recording -> transcript lines (8-minute pieces) + about 40 evenly spaced stills."""
    media = config.media_dir()
    work = media / f'rec_{mid}_{int(time.time() * 1000)}'
    work.mkdir(parents=True)
    try:
        duration = ffprobe_duration(src)
        audio = work / 'audio.wav'
        run_ff(['-i', str(src), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(audio)])
        if not audio.exists() or audio.stat().st_size < 1000:
            raise RuntimeError('The recording has no readable audio track.')
        run_ff(['-i', str(audio), '-f', 'segment', '-segment_time', str(SEGMENT_SECONDS), '-c', 'copy', str(work / 'part_%03d.wav')])
        stt = get_stt()
        new_rows = []
        for idx, part in enumerate(sorted(work.glob('part_*.wav'))):
            res = stt.transcribe(part)
            offset = idx * SEGMENT_SECONDS
            for sg in res.get('segments') or []:
                if sg['text'].strip():
                    new_rows.append((sg['text'].strip(), t0 + offset + parse_ts(sg.get('start'))))
            if not res.get('segments') and (res.get('text') or '').strip():
                new_rows.append((res['text'].strip(), t0 + offset))
        if not new_rows:
            raise RuntimeError('No speech was recognised in the recording.')

        frames = []
        interval = max(60.0, duration / 40) if duration > 0 else 60.0
        try:
            run_ff(['-i', str(src), '-vf', f'fps=1/{interval}', '-frames:v', '40', '-q:v', '4', str(work / 'frame_%03d.jpg')], timeout=1800)
            frames = sorted(work.glob('frame_*.jpg'))
        except Exception as e:
            log.info('No stills from recording: %s', e)
        stamp = int(time.time() * 1000)
        with db.session() as c:
            for i, f in enumerate(frames):
                dest = media / f'{mid}_rec_{stamp}_{i:03d}.jpg'
                shutil.copy(f, dest)
                c.execute('INSERT INTO screenshots(meeting_id,path,timestamp) VALUES(?,?,?)', (mid, config.rel_to_base(dest), t0 + i * interval))
            if replace_live:
                c.execute("DELETE FROM transcript WHERE meeting_id=? AND source IN ('live','groq_whisper')", (mid,))
                # The live events were found in lines that no longer exist; the report rescans the recording.
                c.execute('UPDATE meetings SET last_analyzed_id=0 WHERE id=?', (mid,))
            for txt, ts in new_rows:
                c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',
                          (mid, 'Recording', txt, ts, 'recording'))
        return {'lines': len(new_rows), 'minutes': round(duration / 60, 1), 'stills': len(frames), 'replaced_live': replace_live}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def process_slides(mid: int, src: Path) -> int:
    """PowerPoint -> per-slide title, text, tables and speaker notes."""
    from pptx import Presentation

    try:
        prs = Presentation(str(src))
    except Exception as e:
        raise RuntimeError(f'Could not read this PowerPoint file: {e}')
    rows = []
    for i, slide in enumerate(prs.slides, 1):
        title = ''
        try:
            if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
                title = slide.shapes.title.text_frame.text.strip()
        except Exception:
            pass
        parts = []
        for shp in slide.shapes:
            if getattr(shp, 'has_text_frame', False) and shp.has_text_frame:
                t = '\n'.join(p.text for p in shp.text_frame.paragraphs if p.text.strip()).strip()
                if t and t != title:
                    parts.append(t)
            if getattr(shp, 'has_table', False) and shp.has_table:
                for row in shp.table.rows:
                    parts.append(' | '.join(cell.text.strip() for cell in row.cells))
        notes = ''
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        rows.append((i, title, '\n'.join(parts), notes))
    with db.session() as c:
        c.execute('DELETE FROM slides WHERE meeting_id=?', (mid,))
        for r in rows:
            c.execute('INSERT INTO slides(meeting_id,slide_no,title,text,notes) VALUES(?,?,?,?,?)', (mid, *r))
    return len(rows)
