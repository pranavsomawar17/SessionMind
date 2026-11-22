import json
import math
import sqlite3
import struct
import wave
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import config, db, providers
from tests.fakes import LINES, SAMPLE, STEP, FakeSTT, ScriptedLLM


def make_client(llm=None, stt=None):
    providers._cache['llm'] = llm or ScriptedLLM()
    providers._cache['stt'] = stt or FakeSTT()
    providers._cache['vision'] = providers.VisionProvider()
    from app.main import app
    return TestClient(app)


def start_meeting(client, title='Website relaunch weekly sync'):
    pid = client.post('/api/projects', json={'name': 'Relaunch'}).json()['id']
    return client.post('/api/meetings', json={'project_id': pid, 'title': title}).json()


def load_sample(mid, upto=None):
    """Put the sample meeting into the transcript table, timed relative to the meeting start."""
    with db.session() as c:
        t0 = datetime.fromisoformat(c.execute('SELECT started_at FROM meetings WHERE id=?', (mid,)).fetchone()['started_at']).timestamp()
        for i, text in enumerate(LINES[:upto]):
            c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',
                      (mid, SAMPLE['speaker'], text, t0 + i * STEP, 'live'))
    return t0


# ---------------------------------------------------------------- setup / status
def test_status_reports_each_provider_and_problems(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('OLLAMA_URL', 'http://127.0.0.1:9')
    with TestClient(__import__('app.main', fromlist=['app']).app) as client:
        s = client.get('/api/status').json()
    assert s['ok'] is False and s['providers']['llm']['provider'] == 'ollama'
    assert any('ollama serve' in p for p in s['problems']) and any('GROQ_API_KEY' in p for p in s['problems'])
    assert s['chunk_seconds'] == 10


def test_status_ok_with_working_providers():
    with make_client() as client:
        s = client.get('/api/status').json()
    assert s['providers']['llm']['ok'] and s['providers']['stt']['ok']
    assert s['transcription'].startswith('fake') and s['analysis'].startswith('scripted')


def test_schema_is_created_once_not_on_every_call():
    """The old db() ran CREATE TABLE / ALTER on every call. Prove DDL no longer re-runs: drop a table, connect repeatedly."""
    db.init_db()
    raw = sqlite3.connect(config.db_path())
    raw.execute('DROP TABLE slides')
    raw.commit()
    raw.close()
    for _ in range(5):
        db.connect().close()
    with db.session() as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'slides' not in tables and 'meetings' in tables


def test_old_database_is_migrated_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'old'))
    (tmp_path / 'old').mkdir()
    old = sqlite3.connect(tmp_path / 'old' / 'sessionmind.db')
    old.executescript('''CREATE TABLE meetings(id INTEGER PRIMARY KEY AUTOINCREMENT,project_id INTEGER NOT NULL,title TEXT NOT NULL,status TEXT NOT NULL,started_at TEXT NOT NULL,ended_at TEXT,report TEXT);
        CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER NOT NULL,type TEXT NOT NULL,content TEXT NOT NULL,assignee TEXT,deadline TEXT,timestamp REAL NOT NULL,confidence REAL NOT NULL,status TEXT NOT NULL,evidence TEXT DEFAULT '');
        CREATE TABLE reports(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER UNIQUE NOT NULL,html_path TEXT,pdf_path TEXT,created_at TEXT NOT NULL);
        INSERT INTO meetings(project_id,title,status,started_at) VALUES(1,'Old meeting','ended','2026-01-01T10:00:00+00:00');''')
    old.commit(), old.close()
    db._ready.discard(str(config.db_path()))
    with db.session() as c:
        cols = {r[1] for r in c.execute('PRAGMA table_info(meetings)')}
        assert {'last_analyzed_id', 'current_topic', 'last_analyzed_ts'} <= cols
        assert 'detail' in {r[1] for r in c.execute('PRAGMA table_info(events)')}
        assert c.execute('SELECT title FROM meetings').fetchone()['title'] == 'Old meeting'


# ---------------------------------------------------------------- audio -> transcript
def _wav_bytes(path, seconds=1.5):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(i / 20))) for i in range(int(16000 * seconds))))
    return path.read_bytes()


def test_audio_chunk_is_decoded_transcribed_and_stored(tmp_path):
    data = _wav_bytes(tmp_path / 'chunk.wav')          # a .wav upload used to collide with ffmpeg's output path
    with make_client() as client:
        m = start_meeting(client)
        r = client.post(f"/api/meetings/{m['id']}/audio", files={'audio': ('chunk.wav', data, 'audio/wav')},
                        data={'started_at': '1000', 'speaker': 'Alex'})
        assert r.status_code == 200, r.text
        segs = r.json()['segments']
        assert [s['text'] for s in segs] == ['Hello from the fake transcriber.', 'Can you send the notes?']
        assert segs[1]['timestamp'] == 1002.0 and segs[0]['speaker'] == 'Alex'
        assert not list(config.media_dir().glob('*.wav')) and not list(config.media_dir().glob('*.webm'))   # temp files cleaned up


def test_tiny_audio_is_ignored_and_stt_errors_become_422():
    class Boom(FakeSTT):
        def transcribe(self, wav):
            raise providers.base.ProviderError('whisper exploded')
    with make_client(stt=Boom()) as client:
        m = start_meeting(client)
        assert client.post(f"/api/meetings/{m['id']}/audio", files={'audio': ('a.webm', b'x' * 100, 'audio/webm')}).json()['ignored']
        r = client.post(f"/api/meetings/{m['id']}/audio", files={'audio': ('a.wav', b'x' * 5000, 'audio/wav')})
        assert r.status_code == 422


# ---------------------------------------------------------------- live insights
def test_live_analysis_extracts_items_with_real_timestamps_and_is_incremental():
    llm = ScriptedLLM()
    with make_client(llm) as client:
        m = start_meeting(client)
        mid = m['id']
        load_sample(mid)
        r = client.post(f'/api/meetings/{mid}/analyze').json()
        assert r['busy'] is False and r['current_topic'] == 'Test topic'
        kinds = [e['type'] for e in r['events']]
        # LIVE_MAX_WINDOWS (2) windows per poll: the rest of the transcript is picked up by the next polls
        calls_after_first = llm.scan_calls
        assert calls_after_first == 2
        client.post(f'/api/meetings/{mid}/analyze')
        r = client.post(f'/api/meetings/{mid}/analyze').json()
        events = r['events']
        assert sum(e['type'] == 'task' for e in events) == 7 and sum(e['type'] == 'question' for e in events) == 4
        assert sum(e['type'] == 'decision' for e in events) == 3
        task = next(e for e in events if e['assignee'] == 'Meera')
        assert task['deadline'] == 'Monday' and 'finance' in task['evidence'].lower()   # evidence is the real line, not the last 500 chars
        # nothing new -> no model call
        before = llm.scan_calls
        client.post(f'/api/meetings/{mid}/analyze')
        assert llm.scan_calls == before


def test_late_arriving_chunk_with_an_older_timestamp_is_still_analysed():
    """Regression: the old cursor was a timestamp, so a chunk that finished transcribing late was skipped forever."""
    with make_client() as client:
        mid = start_meeting(client)['id']
        t0 = load_sample(mid, upto=3)                       # lines 1-3
        client.post(f'/api/meetings/{mid}/analyze')
        with db.session() as c:                             # a LATER stretch of the meeting arrives first (00:72) ...
            c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',
                      (mid, 'Meeting Speaker', LINES[6], t0 + 6 * STEP, 'live'))
        events = client.post(f'/api/meetings/{mid}/analyze').json()['events']
        assert [e['type'] for e in events] == ['decision']
        with db.session() as c:                             # ... then the slower, EARLIER chunk (00:36) lands
            c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',
                      (mid, 'Meeting Speaker', LINES[3], t0 + 3 * STEP, 'live'))
        events = client.post(f'/api/meetings/{mid}/analyze').json()['events']
        tasks = [e for e in events if e['type'] == 'task']
        assert len(tasks) == 1 and tasks[0]['assignee'] == 'Rahul'


def test_a_failing_model_gives_502_and_does_not_lose_the_lines():
    state = {'down': True}
    llm = ScriptedLLM(fail_scan=lambda user: state['down'])
    with make_client(llm) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        r = client.post(f'/api/meetings/{mid}/analyze')
        assert r.status_code == 502 and 'scan model offline' in r.json()['detail']
        state['down'] = False
        assert client.post(f'/api/meetings/{mid}/analyze').json()['events']        # same lines retried successfully


def test_overlapping_polls_do_not_run_two_analyses_at_once():
    from app.services import live
    with make_client() as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        lock = live._lock_for(mid)
        assert lock.acquire(blocking=False)
        try:
            r = client.post(f'/api/meetings/{mid}/analyze').json()
            assert r['busy'] is True and r['events'] == []
        finally:
            lock.release()


def test_dismissed_and_confirmed_events():
    with make_client() as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        for _ in range(3):
            events = client.post(f'/api/meetings/{mid}/analyze').json()['events']
        dismissed = next(e for e in events if e['assignee'] == 'Dan')
        assert client.patch(f"/api/meetings/{mid}/events/{dismissed['id']}", json={'status': 'dismissed'}).status_code == 200
        assert client.patch(f"/api/meetings/{mid}/events/{dismissed['id']}", json={'status': 'nope'}).status_code == 400
        rep = client.post(f'/api/meetings/{mid}/end').json()
        # Dan's task is dismissed in the live panel but the full scan still finds it: the scan is the source of truth for the report.
        assert any(a['owner'] == 'Dan' for a in rep['action_items'])


# ---------------------------------------------------------------- final report
def test_report_contains_every_task_decision_and_open_question():
    llm = ScriptedLLM()
    with make_client(llm) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        r = client.post(f'/api/meetings/{mid}/end')
        assert r.status_code == 200, r.text
        rep = r.json()
        assert len(rep['action_items']) == 7 and len(rep['decisions']) == 3 and len(rep['open_questions']) == 4
        assert rep['warnings'] == [] and rep['executive_summary'].startswith('The team agreed')
        assert llm.narrative_calls == 1
        # the narrative call is told about the extracted items but not asked to retype them
        narrative_prompt = [u for s, u in llm.prompts if s.startswith('You are SessionMind')][0]
        assert 'EXTRACTED OPEN QUESTIONS' in narrative_prompt and 'cookie consent' in narrative_prompt
        assert 'scripted (test)' in rep['sources'][-1] and 'fake' in rep['sources'][-1]
        html = client.get(f'/api/reports/{mid}/html').text
        assert 'Open Questions (4)' in html and 'Meera' in html
        docx = client.get(f'/api/reports/{mid}/docx')
        assert docx.status_code == 200 and docx.content[:2] == b'PK'
        got = client.get(f'/api/meetings/{mid}').json()['report_data']
        assert got['mind_view'] in ('tree', 'points') and len(got['open_questions']) == 4


def test_report_is_still_produced_when_the_summary_step_fails():
    with make_client(ScriptedLLM(fail_narrative=True)) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        r = client.post(f'/api/meetings/{mid}/end')
        assert r.status_code == 200
        rep = r.json()
        assert len(rep['action_items']) == 7 and len(rep['open_questions']) == 4
        assert any('summary could not be generated' in w for w in rep['warnings'])
        assert 'summary could not be generated' in client.get(f'/api/reports/{mid}/html').text


def test_report_flags_sections_where_analysis_failed():
    llm = ScriptedLLM(fail_scan=lambda user: '#20 ' in user.split('END OF SLICE')[0])
    with make_client(llm) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        rep = client.post(f'/api/meetings/{mid}/end').json()
        assert rep['warnings'] and 'Unverified' in rep['warnings'][0]
        assert any(q['status'] == 'Unverified' and 'cookie' in q['question'] for q in rep['open_questions'])
        assert any(a['owner'] == 'Rahul' for a in rep['action_items'])


def test_a_long_meeting_sends_section_summaries_instead_of_cutting_off_the_start():
    llm = ScriptedLLM()
    llm.max_input_chars = 1200               # pretend this is a small-context local model
    with make_client(llm) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        rep = client.post(f'/api/meetings/{mid}/end').json()
        prompt = [u for s, u in llm.prompts if s.startswith('You are SessionMind')][0]
        assert 'SECTION SUMMARIES' in prompt and '[00:00]' in prompt          # the beginning is represented
        assert any('section summaries' in w for w in rep['warnings'])
        assert len(rep['open_questions']) == 4


def test_empty_meeting_gives_a_helpful_error():
    with make_client() as client:
        mid = start_meeting(client)['id']
        r = client.post(f'/api/meetings/{mid}/end')
        assert r.status_code == 502 and 'Share audio' in r.json()['detail']


def test_confirmed_live_wording_is_kept_over_the_scan_wording():
    with make_client() as client:
        mid = start_meeting(client)['id']
        t0 = load_sample(mid)
        with db.session() as c:
            c.execute("INSERT INTO events(meeting_id,type,content,assignee,deadline,timestamp,confidence,status,evidence) VALUES(?,?,?,?,?,?,?,?,?)",
                      (mid, 'task', 'Ask finance to approve CloudNine quote', 'Meera', 'Monday', t0 + 13 * STEP, 0.9, 'confirmed', 'I confirmed this'))
        rep = client.post(f'/api/meetings/{mid}/end').json()
        meera = [a for a in rep['action_items'] if a['owner'] == 'Meera']
        assert len(meera) == 1 and meera[0]['task'] == 'Ask finance to approve CloudNine quote'


def test_material_upload_validation_and_slides(tmp_path):
    from pptx import Presentation
    deck = tmp_path / 'deck.pptx'
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = 'Launch plan'
    s.placeholders[1].text = 'Go live March 15'
    prs.save(deck)
    with make_client() as client:
        mid = start_meeting(client)['id']
        assert client.post(f'/api/meetings/{mid}/materials', data={'kind': 'slides'}, files={'file': ('x.pdf', b'abc')}).status_code == 400
        r = client.post(f'/api/meetings/{mid}/materials', data={'kind': 'slides'}, files={'file': ('deck.pptx', deck.read_bytes())})
        assert r.status_code == 200 and r.json()['detail'] == '1 slides'


def test_report_can_be_rebuilt_by_ending_the_meeting_again_after_fixing_the_model():
    """README: if the report has a warning, fix the cause and call /end again."""
    llm = ScriptedLLM(fail_scan=lambda user: True)
    with make_client(llm) as client:
        mid = start_meeting(client)['id']
        load_sample(mid)
        first = client.post(f'/api/meetings/{mid}/end').json()
        assert 'whole transcript' in first['warnings'][0]
        providers._cache['llm'] = ScriptedLLM()                 # the model is back
        second = client.post(f'/api/meetings/{mid}/end').json()
        assert second['warnings'] == [] and len(second['action_items']) == 7 and len(second['open_questions']) == 4
        assert all(q['status'] != 'Unverified' for q in second['open_questions'])
        with db.session() as c:
            assert c.execute('SELECT COUNT(*) FROM reports WHERE meeting_id=?', (mid,)).fetchone()[0] == 1   # updated, not duplicated
