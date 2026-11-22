"""SessionMind: AI meeting intelligence. App wiring only; the work lives in routes/ and services/."""
from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, providers
from .routes import materials, meetings, reports

logging.basicConfig(level=config.env('LOG_LEVEL', 'INFO').upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')

config.media_dir().mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title='SessionMind', version='4.0.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=str(config.BASE / 'static')), name='static')
app.mount('/media', StaticFiles(directory=str(config.media_dir())), name='media')
app.include_router(meetings.router)
app.include_router(materials.router)
app.include_router(reports.router)


@app.get('/')
def root():
    return FileResponse(config.BASE / 'static/index.html')


@app.get('/api/status')
def status():
    problems = []
    ffmpeg = bool(shutil.which('ffmpeg'))
    if not ffmpeg:
        problems.append('FFmpeg is not installed (needed to decode the browser audio).')
    st = providers.status()
    labels = {'stt': 'Speech-to-text', 'llm': 'Analysis model'}
    for key in ('stt', 'llm'):
        if not st[key]['ok']:
            problems.append(f"{labels[key]}: {st[key]['detail']}")
    if not problems and not st['vision']['ok']:   # optional feature: only worth mentioning when the essentials work
        problems.append(f"Screenshot reading: {st['vision']['detail']}")
    try:
        chunk = providers.get_stt().chunk_seconds
    except Exception:
        chunk = 10
    return {
        'ok': not problems, 'missing': problems, 'problems': problems, 'providers': st, 'chunk_seconds': chunk,
        'ffmpeg': ffmpeg, 'capture': 'browser screen + system audio',
        'transcription': f"{st['stt']['provider']} {st['stt']['model']}".strip(),
        'analysis': f"{st['llm']['provider']} {st['llm']['model']}".strip(),
        'local': st['llm']['provider'] == 'ollama' and st['stt']['provider'] in ('whisper', 'faster_whisper'),
    }
