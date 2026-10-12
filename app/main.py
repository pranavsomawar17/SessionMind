from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import sqlite3, os, json, time, subprocess, shutil, re, html
import requests

BASE=Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(BASE/'.env')
except Exception:
    pass
DATA=BASE/'data'; MEDIA=DATA/'media'; DATA.mkdir(exist_ok=True); MEDIA.mkdir(exist_ok=True)
DB=DATA/'sessionmind.db'
app=FastAPI(title='SessionMind',version='3.0.0')
app.mount('/static',StaticFiles(directory=str(BASE/'static')),name='static')
app.mount('/media',StaticFiles(directory=str(MEDIA)),name='media')

class ProjectIn(BaseModel): name:str
class MeetingIn(BaseModel): project_id:int; title:str='Meeting'


def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('PRAGMA busy_timeout=5000')
    c.executescript('''
    CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS meetings(id INTEGER PRIMARY KEY AUTOINCREMENT,project_id INTEGER NOT NULL,title TEXT NOT NULL,status TEXT NOT NULL,started_at TEXT NOT NULL,ended_at TEXT,report TEXT);
    CREATE TABLE IF NOT EXISTS transcript(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER NOT NULL,speaker TEXT NOT NULL,text TEXT NOT NULL,timestamp REAL NOT NULL,source TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS screenshots(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER NOT NULL,path TEXT NOT NULL,timestamp REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER NOT NULL,type TEXT NOT NULL,content TEXT NOT NULL,assignee TEXT,deadline TEXT,timestamp REAL NOT NULL,confidence REAL NOT NULL,status TEXT NOT NULL,evidence TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY AUTOINCREMENT,meeting_id INTEGER UNIQUE NOT NULL,html_path TEXT,pdf_path TEXT,created_at TEXT NOT NULL);
    ''')
    # Migration-safe additions for meetings created before live-insights support existed.
    for col,decl in (('last_analyzed_ts','REAL DEFAULT 0'),('current_topic','TEXT')):
        try: c.execute(f'ALTER TABLE meetings ADD COLUMN {col} {decl}')
        except sqlite3.OperationalError: pass
    return c

def now(): return datetime.now(timezone.utc).isoformat()
def env_keys(name): return [x.strip() for x in os.getenv(name,'').split(',') if x.strip()]
def groq_keys(): return env_keys('GROQ_API_KEY') or env_keys('GROQ_API_KEYS')

def require_config():
    missing=[]
    if not groq_keys(): missing.append('GROQ_API_KEY')
    if not shutil.which('ffmpeg'): missing.append('ffmpeg')
    return missing

def groq_transcribe(wav:Path):
    last=''
    for key in groq_keys():
        try:
            with wav.open('rb') as f:
                r=requests.post('https://api.groq.com/openai/v1/audio/transcriptions',headers={'Authorization':f'Bearer {key}'},files={'file':(wav.name,f,'audio/wav')},data={'model':os.getenv('GROQ_TRANSCRIPTION_MODEL','whisper-large-v3-turbo'),'response_format':'verbose_json','temperature':'0'},timeout=180)
            if r.status_code in (401,403): last=f'Groq authentication failed: {r.text[:500]}'; continue
            if r.status_code==429: last=f'Groq quota/rate limit: {r.text[:500]}'; continue
            r.raise_for_status(); return r.json()
        except Exception as e: last=str(e)
    raise RuntimeError(last or 'Groq transcription failed')

def extract_wav(src:Path)->Path:
    ff=shutil.which('ffmpeg')
    if not ff: raise RuntimeError('FFmpeg is not installed. Fedora: sudo dnf install ffmpeg')
    out=src.with_suffix('.wav')
    # Robustly decode browser WebM/Opus. Explicitly ignore video because the upload is audio-only.
    cmd=[ff,'-hide_banner','-loglevel','error','-y','-i',str(src),'-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(out)]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if p.returncode or not out.exists() or out.stat().st_size<1000:
        raise RuntimeError('FFmpeg could not decode this audio recording. Browser capture may have produced an empty stream. '+p.stderr[-800:])
    return out

def groq_chat_model(key):
    """Pick a strong currently available Groq chat model.
    A configured GROQ_CHAT_MODEL wins when it is available; otherwise use a
    sensible preference order and finally any available text-generation model.
    """
    configured=os.getenv('GROQ_CHAT_MODEL','llama-3.3-70b-versatile').strip()
    try:
        r=requests.get('https://api.groq.com/openai/v1/models',
                       headers={'Authorization':f'Bearer {key}','accept':'application/json'},
                       timeout=30)
        if r.ok:
            ids=[x.get('id','') for x in r.json().get('data',[]) if x.get('id')]
            if configured in ids:
                return configured
            preferred=[
                'llama-3.3-70b-versatile',
                'openai/gpt-oss-120b',
                'qwen/qwen3-32b',
                'llama-3.1-70b-versatile',
                'llama-3.1-8b-instant'
            ]
            for model in preferred:
                if model in ids:
                    return model
            # Avoid embedding/whisper/guard models when possible.
            blocked=('whisper','embed','guard','safeguard','compound')
            text_models=[x for x in ids if not any(b in x.lower() for b in blocked)]
            if text_models:
                return text_models[0]
    except Exception:
        pass
    return configured

def groq_json(system,user):
    """Run structured meeting analysis through Groq's OpenAI-compatible API."""
    last=''
    for key in groq_keys():
        model=groq_chat_model(key)
        payload={
            'model':model,
            'messages':[{'role':'system','content':system},{'role':'user','content':user}],
            'temperature':0.15,
            'max_tokens':8000,
            'response_format':{'type':'json_object'}
        }
        try:
            r=requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},
                json=payload,timeout=300
            )
            if r.status_code in (401,403):
                last=f'Groq authentication failed: {r.text[:1200]}'; continue
            if r.status_code==429:
                last=f'Groq rate/quota limit: {r.text[:1200]}'; continue
            if not r.ok:
                try: detail=r.json().get('error',{}).get('message') or r.text
                except Exception: detail=r.text
                last=f'Groq API error {r.status_code} using model {model}: {detail[:1500]}'
                continue
            data=r.json()
            text=data.get('choices',[{}])[0].get('message',{}).get('content','').strip()
            try:
                return json.loads(text)
            except Exception:
                m=re.search(r'\{.*\}',text,re.S)
                if m:
                    return json.loads(m.group(0))
                raise RuntimeError('Groq returned non-JSON output')
        except Exception as e:
            last=str(e)
    raise RuntimeError(last or 'Groq meeting analysis failed')

def groq_vision_model(key):
    """Best-effort pick of a currently available Groq vision-capable model.
    Returns None if the account has no vision model available — callers must
    treat that as 'skip visual analysis', never as a fatal error."""
    try:
        r=requests.get('https://api.groq.com/openai/v1/models',
                       headers={'Authorization':f'Bearer {key}','accept':'application/json'},
                       timeout=30)
        if r.ok:
            ids=[x.get('id','') for x in r.json().get('data',[]) if x.get('id')]
            preferred=[
                'meta-llama/llama-4-scout-17b-16e-instruct',
                'meta-llama/llama-4-maverick-17b-128e-instruct',
                'llama-3.2-90b-vision-preview',
                'llama-3.2-11b-vision-preview',
            ]
            for model in preferred:
                if model in ids: return model
            for i in ids:
                low=i.lower()
                if 'vision' in low or 'scout' in low or 'maverick' in low:
                    return i
    except Exception:
        pass
    return None

def groq_vision_read(image_bytes):
    """Read live captions or slide/document text out of a screenshot.
    Best-effort: returns None on any failure or when no vision model is
    available, so a screenshot never blocks or breaks the meeting."""
    import base64
    b64=base64.b64encode(image_bytes).decode()
    prompt=('This image is a screenshot of a live video-call screen (e.g. Zoom or Google Meet). '
            'If live captions/subtitles are visible, transcribe them exactly, including the speaker '
            'name if one is labeled next to the caption. Otherwise, if a presentation slide or shared '
            'document with meaningful text is visible, extract its key heading/text briefly. '
            'Otherwise reply with exactly: NONE. Reply with only the transcription or extracted text, '
            'nothing else — no preamble, no quotes.')
    for key in groq_keys():
        model=groq_vision_model(key)
        if not model: continue
        try:
            payload={
                'model':model,
                'messages':[{'role':'user','content':[
                    {'type':'text','text':prompt},
                    {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}}
                ]}],
                'temperature':0,'max_tokens':300
            }
            r=requests.post('https://api.groq.com/openai/v1/chat/completions',
                            headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},
                            json=payload,timeout=60)
            if not r.ok: continue
            text=r.json().get('choices',[{}])[0].get('message',{}).get('content','').strip()
            if text and text.upper()!='NONE':
                return text
            return None
        except Exception:
            continue
    return None

def save_transcript(mid,text,ts,speaker='Meeting Speaker',source='groq_whisper'):
    text=(text or '').strip()
    if not text:return []
    c=db(); cur=c.execute('INSERT INTO transcript(meeting_id,speaker,text,timestamp,source) VALUES(?,?,?,?,?)',(mid,speaker,text,ts,source)); c.commit(); row=c.execute('SELECT * FROM transcript WHERE id=?',(cur.lastrowid,)).fetchone(); c.close(); return dict(row)

def parse_ts(v, fallback=0):
    try:return float(v)
    except:return fallback

def previous_context(mid):
    c=db(); m=c.execute('SELECT project_id FROM meetings WHERE id=?',(mid,)).fetchone()
    if not m:c.close();return ''
    rows=c.execute('SELECT title,report FROM meetings WHERE project_id=? AND id<? AND status="ended" ORDER BY id DESC LIMIT 3',(m['project_id'],mid)).fetchall(); c.close()
    out=[]
    for r in rows:
        if not r['report']:continue
        try:
            x=json.loads(r['report']); out.append({'title':r['title'],'summary':x.get('executive_summary',''),'decisions':x.get('decisions',[]),'action_items':x.get('action_items',[]),'open_questions':x.get('open_questions',[])})
        except:pass
    return json.dumps(out,ensure_ascii=False)[:16000]

@app.on_event('startup')
def startup():
    db().close()

@app.get('/')
def root(): return FileResponse(BASE/'static/index.html')

@app.get('/api/status')
def status():
    missing=require_config()
    return {'ok':not missing,'missing':missing,'groq_configured':bool(groq_keys()),'llm_configured':bool(groq_keys()),'ffmpeg':bool(shutil.which('ffmpeg')),'capture':'browser screen + system audio','transcription':'Groq Whisper','analysis':'Groq LLM'}

@app.get('/api/projects')
def projects():
    c=db(); rows=c.execute('SELECT * FROM projects ORDER BY id DESC').fetchall(); c.close(); return [dict(x) for x in rows]
@app.post('/api/projects')
def create_project(x:ProjectIn):
    name=x.name.strip() or 'My Project'; c=db(); cur=c.execute('INSERT INTO projects(name,created_at) VALUES(?,?)',(name,now())); c.commit(); row=c.execute('SELECT * FROM projects WHERE id=?',(cur.lastrowid,)).fetchone(); c.close(); return dict(row)
@app.get('/api/projects/{pid}/meetings')
def meetings(pid:int):
    c=db(); rows=c.execute('SELECT id,title,status,started_at,ended_at FROM meetings WHERE project_id=? ORDER BY id DESC',(pid,)).fetchall(); c.close(); return [dict(x) for x in rows]
@app.post('/api/meetings')
def create_meeting(x:MeetingIn):
    c=db(); p=c.execute('SELECT id FROM projects WHERE id=?',(x.project_id,)).fetchone()
    if not p:c.close(); raise HTTPException(404,'Project not found')
    cur=c.execute('INSERT INTO meetings(project_id,title,status,started_at) VALUES(?,?,?,?)',(x.project_id,x.title.strip() or 'Meeting','live',now())); c.commit(); row=c.execute('SELECT * FROM meetings WHERE id=?',(cur.lastrowid,)).fetchone(); c.close(); return dict(row)

@app.post('/api/meetings/{mid}/audio')
def audio(mid:int,audio:UploadFile=File(...),started_at:float|None=Form(None),speaker:str|None=Form(None)):
    # Plain `def`, not `async def`: FastAPI runs sync endpoints in a worker
    # thread automatically, so the ffmpeg decode + Groq call below (which can
    # take several seconds) no longer blocks the event loop for every other
    # in-flight request (screenshots, status checks, other audio chunks).
    c=db(); m=c.execute('SELECT status,started_at FROM meetings WHERE id=?',(mid,)).fetchone(); c.close()
    if not m: raise HTTPException(404,'Meeting not found')
    if m['status']!='live': raise HTTPException(409,'Meeting is not live')
    data=audio.file.read()
    if len(data)<4000: return {'ok':True,'ignored':True,'reason':'small audio'}
    ext=Path(audio.filename or 'capture.webm').suffix.lower() or '.webm'
    src=MEDIA/f'{mid}_{int(time.time()*1000)}{ext}'; src.write_bytes(data)
    wav=None
    try:
        wav=extract_wav(src); result=groq_transcribe(wav)
        base=started_at or time.time()-float(result.get('duration') or 0)
        created=[]
        segs=result.get('segments') or []
        label=(speaker or '').strip() or 'Meeting Speaker'
        if segs:
            for s in segs:
                txt=(s.get('text') or '').strip()
                if txt: created.append(save_transcript(mid,txt,base+parse_ts(s.get('start'),0),label))
        elif result.get('text'): created.append(save_transcript(mid,result['text'],base,label))
        return {'ok':True,'text':result.get('text',''),'segments':created}
    except Exception as e:
        raise HTTPException(422,f'Audio processing error: {e}')
    finally:
        for p in (src,wav):
            if p:
                try:p.unlink()
                except:pass

@app.post('/api/meetings/{mid}/screenshot')
def screenshot(mid:int,image:UploadFile=File(...),timestamp:float|None=Form(None)):
    c=db(); ok=c.execute('SELECT id FROM meetings WHERE id=?',(mid,)).fetchone(); c.close()
    if not ok:raise HTTPException(404,'Meeting not found')
    ts=timestamp or time.time()
    data=image.file.read()
    p=MEDIA/f'{mid}_{int(ts*1000)}.jpg'; p.write_bytes(data)
    c=db(); cur=c.execute('INSERT INTO screenshots(meeting_id,path,timestamp) VALUES(?,?,?)',(mid,str(p.relative_to(BASE)),ts)); c.commit(); row=c.execute('SELECT * FROM screenshots WHERE id=?',(cur.lastrowid,)).fetchone(); c.close()
    # Best-effort: read captions/slide text out of the frame and fold it into
    # the transcript as evidence. Never lets a vision failure break capture.
    try:
        text=groq_vision_read(data)
        if text: save_transcript(mid,text,ts,'Screen',source='groq_vision')
    except Exception:
        pass
    return dict(row)

@app.post('/api/meetings/{mid}/analyze')
def analyze(mid:int):
    """Incremental live extraction: pulls transcript lines added since the
    last call, asks Groq for any new tasks/decisions/questions/topic change,
    and stores them in the existing (previously unused) events table."""
    c=db()
    m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone()
    if not m: c.close(); raise HTTPException(404,'Meeting not found')
    rows=c.execute('SELECT * FROM transcript WHERE meeting_id=? AND timestamp>? ORDER BY timestamp',(mid,m['last_analyzed_ts'] or 0)).fetchall()
    if not rows:
        events=[dict(x) for x in c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY id',(mid,)).fetchall()]
        c.close()
        return {'events':events,'current_topic':m['current_topic'],'new':0}
    window='\n'.join(f'[{x["timestamp"]:.1f}] {x["speaker"]}: {x["text"]}' for x in rows)
    existing=c.execute('SELECT type,content FROM events WHERE meeting_id=?',(mid,)).fetchall()
    existing_desc='\n'.join(f'- ({x["type"]}) {x["content"]}' for x in existing) or 'None yet'
    system=('You are the real-time extraction engine for SessionMind. Read the new transcript segment '
            'and extract structured events. Extract a "decision" only when there is clear agreement or '
            'confirmation, not a mere suggestion. Extract a "task" only when someone is clearly assigned '
            'to or commits to an action. Extract a "question" only for genuine unanswered questions. '
            'Do not invent information not present in the transcript. Do not repeat already-detected '
            'events. Return ONLY valid JSON with this exact shape: '
            '{"topic":string|null,"events":[{"type":"task|decision|question","description":"","assignee":string|null,"deadline":string|null,"confidence":0.0}]}')
    user=f'CURRENT TOPIC (may be unchanged): {m["current_topic"] or "unknown"}\n\nNEW TRANSCRIPT SEGMENT:\n{window}\n\nALREADY-DETECTED EVENTS (do not repeat):\n{existing_desc}'
    try:
        result=groq_json(system,user)
    except Exception as e:
        c.close(); raise HTTPException(502,f'Live analysis failed: {e}')
    last_ts=rows[-1]['timestamp']
    topic=result.get('topic') or m['current_topic']
    c.execute('UPDATE meetings SET last_analyzed_ts=?,current_topic=? WHERE id=?',(last_ts,topic,mid))
    for ev in (result.get('events') or []):
        content=(ev.get('description') or '').strip()
        etype=ev.get('type') if ev.get('type') in ('task','decision','question') else None
        if not content or not etype: continue
        dup=c.execute('SELECT id FROM events WHERE meeting_id=? AND type=? AND lower(content)=lower(?)',(mid,etype,content)).fetchone()
        if dup: continue
        c.execute('INSERT INTO events(meeting_id,type,content,assignee,deadline,timestamp,confidence,status,evidence) VALUES(?,?,?,?,?,?,?,?,?)',
                  (mid,etype,content,ev.get('assignee'),ev.get('deadline'),last_ts,float(ev.get('confidence') or 0.7),'detected',window[-500:]))
    c.commit()
    events=[dict(x) for x in c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY id',(mid,)).fetchall()]
    c.close()
    return {'events':events,'current_topic':topic,'new':len(rows)}

@app.get('/api/meetings/{mid}/events')
def list_events(mid:int):
    c=db(); ok=c.execute('SELECT id FROM meetings WHERE id=?',(mid,)).fetchone()
    if not ok: c.close(); raise HTTPException(404,'Meeting not found')
    rows=c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY id',(mid,)).fetchall(); c.close(); return [dict(x) for x in rows]

class EventStatus(BaseModel): status:str
@app.patch('/api/meetings/{mid}/events/{eid}')
def update_event(mid:int,eid:int,x:EventStatus):
    if x.status not in ('detected','confirmed','dismissed'): raise HTTPException(400,'Invalid status')
    c=db(); cur=c.execute('UPDATE events SET status=? WHERE id=? AND meeting_id=?',(x.status,eid,mid))
    if cur.rowcount==0: c.close(); raise HTTPException(404,'Event not found')
    c.commit(); row=c.execute('SELECT * FROM events WHERE id=?',(eid,)).fetchone(); c.close(); return dict(row)

@app.post('/api/meetings/{mid}/end')
def end_meeting(mid:int):
    c=db(); m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone(); c.close()
    if not m:raise HTTPException(404,'Meeting not found')
    c=db(); c.execute('UPDATE meetings SET status="ended",ended_at=? WHERE id=?',(now(),mid)); c.commit(); c.close()
    try:return build_report(mid)
    except Exception as e:
        # Keep the meeting ended and expose a useful error instead of an opaque 500.
        raise HTTPException(502,f'Report generation failed: {e}')

def build_report(mid):
    c=db(); m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone(); p=c.execute('SELECT * FROM projects WHERE id=?',(m['project_id'],)).fetchone(); t=c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); s=c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); e=c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); c.close()
    if not t: raise RuntimeError('No transcript was captured. Make sure the screen-share dialog has Share audio enabled and allow the meeting to run for at least 10 seconds.')
    transcript='\n'.join(f'[{x["timestamp"]:.2f}] {x["speaker"]}: {x["text"]}' for x in t)
    # Keep enough context for a detailed report while avoiding context-limit 400s on long meetings.
    transcript=transcript[-80000:]
    system='''You are SessionMind, an evidence-first meeting analyst. Analyze the complete meeting transcript. Do not invent facts. Suggestions are not decisions; only treat a decision as confirmed when agreement/confirmation is explicit. Every extracted item must be traceable to transcript timestamps. Return ONLY valid JSON with this exact shape: {"executive_summary":"","meeting_overview":"","key_discussion_points":[""],"decisions":[{"decision":"","evidence":""}],"action_items":[{"owner":"","task":"","deadline":"","status":"Assigned","evidence":""}],"open_questions":[{"question":"","status":"Unanswered","evidence":""}],"risks":[{"risk":"","evidence":""}],"topics":[{"name":"","summary":"","evidence":""}],"important_moments":[{"timestamp":"","description":"","evidence":""}],"changes_from_previous_meeting":[""],"participants":[""]}. Use "Unknown" when ownership/deadline is not stated. Keep it concise but comprehensive.'''
    user=f'Meeting: {m["title"]}\nTranscript:\n{transcript}\n\nPrevious project context:\n{previous_context(mid)}'
    rep=groq_json(system,user)
    rep.update({'meeting_id':mid,'title':m['title'],'project':p['name'],'started_at':m['started_at'],'ended_at':m['ended_at'],'screenshot_count':len(s)})
    for ev in rep.get('action_items',[]):
        ev.setdefault('status','Assigned')
    html_path=DATA/f'report_{mid}.html'; pdf_path=DATA/f'report_{mid}.pdf'
    render_html(rep,t,s,html_path); render_pdf(rep,t,s,pdf_path)
    c=db(); c.execute('UPDATE meetings SET report=? WHERE id=?',(json.dumps(rep,ensure_ascii=False),mid)); c.execute('INSERT INTO reports(meeting_id,html_path,pdf_path,created_at) VALUES(?,?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET html_path=excluded.html_path,pdf_path=excluded.pdf_path,created_at=excluded.created_at',(mid,str(html_path.relative_to(BASE)),str(pdf_path.relative_to(BASE)),now())); c.commit(); c.close()
    return {**rep,'files':{'html':f'/api/reports/{mid}/html','pdf':f'/api/reports/{mid}/pdf'}}

def H(x):return html.escape(str(x or ''))
def render_html(rep,t,s,path):
    topics=rep.get('topics',[]); tasks=rep.get('action_items',[])
    topic_nodes=''.join(f'<div class="node">{H(x.get("name"))}</div>' for x in topics)
    shots=''.join(f'<figure><img src="/media/{H(Path(x["path"]).name)}"><figcaption>{x["timestamp"]:.0f}s</figcaption></figure>' for x in s)
    task_rows=''.join(f'<tr><td>{H(x.get("owner"))}</td><td>{H(x.get("task"))}</td><td>{H(x.get("deadline"))}</td><td>{H(x.get("status"))}</td></tr>' for x in tasks) or '<tr><td colspan="4">No action items recorded.</td></tr>'
    def bullets(a,key=None):return ''.join('<li>'+H(x if key is None else x.get(key,''))+'</li>' for x in a) or '<li>None recorded.</li>'
    transcript=''.join(f'<p><b>{x["timestamp"]:.0f}s · {H(x["speaker"])}</b> — {H(x["text"])}</p>' for x in t)
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>{H(rep["title"])} - SessionMind</title><style>body{{font:15px system-ui;max-width:1000px;margin:40px auto;color:#18202a;line-height:1.55}}h1{{margin-bottom:4px}}h2{{margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:7px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}figure{{display:inline-block;width:46%;vertical-align:top}}img{{max-width:100%;border:1px solid #ddd}}.mind{{padding:20px;border:1px solid #ddd;border-radius:14px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}.center{{width:100%;font-weight:700}}.node{{border:1px solid #aaa;border-radius:20px;padding:8px 14px}}</style></head><body><h1>{H(rep["title"])}</h1><div>SessionMind AI Meeting Report · {H(rep["project"])}</div><h2>Executive Summary</h2><p>{H(rep.get("executive_summary"))}</p><h2>Meeting Overview</h2><p>{H(rep.get("meeting_overview"))}</p><h2>Participants</h2><ul>{bullets(rep.get("participants",[]))}</ul><h2>Key Discussion Points</h2><ul>{bullets(rep.get("key_discussion_points",[]))}</ul><h2>Decisions</h2><ul>{''.join(f'<li><b>{H(x.get("decision"))}</b> — {H(x.get("evidence"))}</li>' for x in rep.get("decisions",[])) or '<li>None recorded.</li>'}</ul><h2>Action Items</h2><table><tr><th>Owner</th><th>Task</th><th>Deadline</th><th>Status</th></tr>{task_rows}</table><h2>Open Questions</h2><ul>{''.join(f'<li>{H(x.get("question"))} — {H(x.get("status"))} [{H(x.get("evidence"))}]</li>' for x in rep.get("open_questions",[])) or '<li>None recorded.</li>'}</ul><h2>Risks / Blockers</h2><ul>{''.join(f'<li>{H(x.get("risk"))} — {H(x.get("evidence"))}</li>' for x in rep.get("risks",[])) or '<li>None recorded.</li>'}</ul><h2>Topics / Mind Map</h2><div class="mind"><div class="center">{H(rep["title"])}</div>{topic_nodes or '<div>No topics recorded.</div>'}</div><h2>Important Moments</h2><ul>{''.join(f'<li><b>{H(x.get("timestamp"))}</b> — {H(x.get("description"))} [{H(x.get("evidence"))}]</li>' for x in rep.get("important_moments",[])) or '<li>None recorded.</li>'}</ul><h2>Changes From Previous Meeting</h2><ul>{bullets(rep.get("changes_from_previous_meeting",[]))}</ul><h2>Meeting Screenshots</h2>{shots or '<p>No screenshots captured.</p>'}<h2>Evidence Transcript</h2>{transcript}</body></html>'''
    path.write_text(body,encoding='utf-8')

def render_pdf(rep,t,s,path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    st=getSampleStyleSheet(); st.add(ParagraphStyle(name='Tiny',parent=st['BodyText'],fontSize=8,textColor=colors.grey))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=14*mm,leftMargin=14*mm,topMargin=14*mm,bottomMargin=14*mm)
    story=[Paragraph('SessionMind — AI Meeting Report',st['Title']),Paragraph(H(rep['title']),st['Heading2']),Paragraph(f'Project: {H(rep["project"])}',st['Tiny']),Spacer(1,8),Paragraph('Executive Summary',st['Heading2']),Paragraph(H(rep.get('executive_summary','')),st['BodyText'])]
    for title,items in [('Meeting Overview',[rep.get('meeting_overview','')]),('Key Discussion Points',rep.get('key_discussion_points',[]))]:
        story += [Paragraph(title,st['Heading2'])]
        for x in items: story += [Paragraph(H(x),st['BodyText']),Spacer(1,3)]
    story += [Paragraph('Decisions',st['Heading2'])]
    for x in rep.get('decisions',[]): story.append(Paragraph(f'<b>{H(x.get("decision"))}</b> — {H(x.get("evidence"))}',st['BodyText']))
    story += [Paragraph('Action Items',st['Heading2'])]
    data=[['Owner','Task','Deadline','Status']]+[[x.get('owner','Unknown'),x.get('task',''),x.get('deadline','Unknown'),x.get('status','Assigned')] for x in rep.get('action_items',[])]
    if len(data)==1:data.append(['—','None recorded.','—','—'])
    tab=Table(data,colWidths=[30*mm,78*mm,32*mm,30*mm],repeatRows=1); tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),('VALIGN',(0,0),(-1,-1),'TOP'),('FONTSIZE',(0,0),(-1,-1),8)])); story += [tab]
    for title,key in [('Open Questions','open_questions'),('Risks / Blockers','risks'),('Important Moments','important_moments'),('Changes From Previous Meeting','changes_from_previous_meeting')]:
        story += [Paragraph(title,st['Heading2'])]
        arr=rep.get(key,[])
        for x in arr:
            if isinstance(x,dict): text=' — '.join(str(x.get(k,'')) for k in ('question','risk','description','timestamp','evidence') if x.get(k))
            else:text=str(x)
            story.append(Paragraph(H(text),st['BodyText']))
    story += [Paragraph('Topics / Mind Map',st['Heading2']),Paragraph(H(rep['title']),st['BodyText'])]
    for x in rep.get('topics',[]): story.append(Paragraph('→ '+H(x.get('name','')),st['BodyText']))
    if s:
        story += [Paragraph('Meeting Screenshots',st['Heading2'])]
        for x in s[:10]:
            p=BASE/x['path']
            if p.exists():
                try:story += [Image(str(p),width=165*mm,height=92*mm),Paragraph(f'Evidence timestamp: {x["timestamp"]:.0f}s',st['Tiny']),Spacer(1,5)]
                except:pass
    story += [PageBreak(),Paragraph('Evidence Transcript',st['Heading2'])]
    for x in t: story.append(Paragraph(f'<font color="grey">{x["timestamp"]:.0f}s · {H(x["speaker"])}</font> — {H(x["text"])}',st['BodyText']))
    doc.build(story)

@app.get('/api/meetings/{mid}')
def get_meeting(mid:int):
    c=db(); m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone()
    if not m:c.close();raise HTTPException(404,'Meeting not found')
    t=c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); s=c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); c.close(); return {**dict(m),'transcript':[dict(x) for x in t],'screenshots':[dict(x) for x in s]}
@app.get('/api/reports/{mid}/pdf')
def pdf(mid:int):
    p=DATA/f'report_{mid}.pdf'
    if not p.exists():raise HTTPException(404,'Report not found')
    return FileResponse(p,media_type='application/pdf',filename=f'sessionmind_{mid}.pdf')
@app.get('/api/reports/{mid}/html')
def report_html(mid:int):
    p=DATA/f'report_{mid}.html'
    if not p.exists():raise HTTPException(404,'Report not found')
    return FileResponse(p,media_type='text/html')
@app.get('/api/search')
def search(q:str,project_id:int|None=None):
    c=db(); like=f'%{q}%'; sql='SELECT t.*,m.title FROM transcript t JOIN meetings m ON m.id=t.meeting_id WHERE (t.text LIKE ? OR t.speaker LIKE ?)'; args=[like,like]
    if project_id:sql+=' AND m.project_id=?';args.append(project_id)
    rows=c.execute(sql+' ORDER BY t.timestamp DESC LIMIT 100',args).fetchall();c.close();return [dict(x) for x in rows]
