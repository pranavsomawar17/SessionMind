"""Report downloads and transcript search."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import config, db

router = APIRouter()


@router.get('/api/reports/{mid}/docx')
def docx_report(mid: int):
    p = config.data_dir() / f'report_{mid}.docx'
    if not p.exists():
        raise HTTPException(404, 'Report not found')
    return FileResponse(p, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        filename=f'sessionmind_{mid}.docx')


@router.get('/api/reports/{mid}/html')
def report_html(mid: int):
    p = config.data_dir() / f'report_{mid}.html'
    if not p.exists():
        raise HTTPException(404, 'Report not found')
    return FileResponse(p, media_type='text/html')


@router.get('/api/search')
def search(q: str, project_id: int | None = None):
    like = f'%{q}%'
    sql = ('SELECT t.*,m.title FROM transcript t JOIN meetings m ON m.id=t.meeting_id '
           'WHERE (t.text LIKE ? OR t.speaker LIKE ?)')
    args: list = [like, like]
    if project_id:
        sql += ' AND m.project_id=?'
        args.append(project_id)
    with db.session() as c:
        return [dict(x) for x in c.execute(sql + ' ORDER BY t.timestamp DESC LIMIT 100', args).fetchall()]
