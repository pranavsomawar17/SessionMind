from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import sqlite3, os, json, time, subprocess, shutil, re, html, difflib
import requests
from .report_render import (build_mind_map, clock, render_docx, render_html, question_rows)

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
    for table,col,decl in (('meetings','last_analyzed_ts','REAL DEFAULT 0'),('meetings','current_topic','TEXT'),('reports','docx_path','TEXT')):
        try: c.execute(f'ALTER TABLE {table} ADD COLUMN {col} {decl}')
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
            'to or commits to an action. Extract a "question" for every question or open issue raised in the '
            'segment that is not answered within that same segment, including questions the group defers, '
            'items marked "we need to check/confirm", and unknown owners or deadlines that were flagged. '
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

QUESTION_SYSTEM=('You extract OPEN QUESTIONS from one segment of a meeting transcript. Include every unresolved item: '
 'explicit questions that the segment does not answer; questions only partly answered; items explicitly deferred, tabled, '
 'or "to be checked/confirmed later"; options or disagreements left undecided; unknown owners, deadlines, numbers or facts '
 'the group said are still needed; blockers that need information. Do not include questions the segment fully answers. '
 'Do not invent anything that is not in the segment. Return ONLY valid JSON: '
 '{"open_questions":[{"question":"","raised_by":"","timestamp":"","status":"Unanswered|Partially answered|Deferred","evidence":""}]}. '
 'Evidence must quote or closely paraphrase the transcript line.')

def transcript_chunks(rows,limit=24000):
    """Yield the transcript in ~24k-char pieces, split on line boundaries."""
    chunk=[]; size=0
    for x in rows:
        line=f'[{clock(x["timestamp"])}] {x["speaker"]}: {x["text"]}'
        if chunk and size+len(line)>limit:
            yield '\n'.join(chunk); chunk=[]; size=0
        chunk.append(line); size+=len(line)+1
    if chunk: yield '\n'.join(chunk)

def norm_q(s): return re.sub(r'\W+',' ',str(s or '').lower()).strip()

def same_question(a,b):
    """Duplicate check. Near-identical wording counts as the same question only when the
    numbers in it match too, so 'item 5?' and 'item 10?' stay two separate questions."""
    na,nb=norm_q(a),norm_q(b)
    if not na or not nb: return False
    if na==nb: return True
    if re.findall(r'\d+',na)!=re.findall(r'\d+',nb): return False
    return difflib.SequenceMatcher(None,na,nb).ratio()>=0.9

def clean_question(q,default_status='Unanswered'):
    if isinstance(q,str): q={'question':q}
    if not isinstance(q,dict): return None
    text=str(q.get('question') or '').strip()
    if not text: return None
    return {'question':text,'status':str(q.get('status') or default_status).strip() or default_status,
            'raised_by':str(q.get('raised_by') or '').strip(),'timestamp':str(q.get('timestamp') or '').strip(),
            'evidence':str(q.get('evidence') or '').strip()}

def merge_questions(*groups):
    out=[]
    for g in groups:
        for q in g:
            if q and not any(same_question(q['question'],o['question']) for o in out): out.append(q)
    return out

def scan_open_questions(rows):
    """Pass 1: question-focused extraction over the WHOLE transcript, chunk by chunk,
    so a long meeting's early questions are not lost to the final report's window."""
    found=[]
    for chunk in transcript_chunks(rows):
        try: res=groq_json(QUESTION_SYSTEM,'TRANSCRIPT SEGMENT:\n'+chunk)
        except Exception as e:
            print(f'[SessionMind] question scan chunk failed: {e}'); continue
        found+=[q for q in (clean_question(x) for x in (res.get('open_questions') or [])) if q]
    return merge_questions(found)

def hydrate_report(raw):
    rep=json.loads(raw); rep['mind_map_tree']=build_mind_map(rep); return rep

@app.get('/api/meetings/{mid}')
def get_meeting(mid:int):
    c=db(); m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone()
    if not m:c.close();raise HTTPException(404,'Meeting not found')
    t=c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); s=c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); c.close()
    report=None
    if m['report']:
        try: report=hydrate_report(m['report'])
        except Exception: report=None
    return {**dict(m),'transcript':[dict(x) for x in t],'screenshots':[dict(x) for x in s],'report_data':report}

def build_report(mid):
    c=db(); m=c.execute('SELECT * FROM meetings WHERE id=?',(mid,)).fetchone(); p=c.execute('SELECT * FROM projects WHERE id=?',(m['project_id'],)).fetchone(); t=c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); s=c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); e=c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY timestamp',(mid,)).fetchall(); c.close()
    if not t: raise RuntimeError('No transcript was captured. Make sure the screen-share dialog has Share audio enabled and allow the meeting to run for at least 10 seconds.')
    # Timestamps shown to the model and in the report are meeting-relative (mm:ss), not epoch seconds.
    try: t0=datetime.fromisoformat(m['started_at']).timestamp()
    except Exception: t0=t[0]['timestamp']
    rel=lambda ts: max(0.0,float(ts)-t0)
    rows=[{**dict(x),'timestamp':rel(x['timestamp'])} for x in t]
    shots=[{**dict(x),'timestamp':rel(x['timestamp'])} for x in s]
    # Pass 1: question-focused scan over the complete transcript, plus questions detected live.
    scanned=scan_open_questions(rows)
    live=[clean_question({'question':x['content'],'evidence':x['evidence'] or '','timestamp':clock(rel(x['timestamp']))}) for x in e if x['type']=='question' and x['status']!='dismissed']
    candidates=merge_questions(scanned,[q for q in live if q])
    # Whole candidate entries only, within a size budget (anything left out is restored by the safety net below).
    cand_items=[]; budget=20000
    for q in candidates:
        item=json.dumps({k:q[k] for k in ('question','status','raised_by','timestamp','evidence')},ensure_ascii=False)
        if sum(len(x)+2 for x in cand_items)+len(item)>budget: break
        cand_items.append(item)
    cand_json='['+', '.join(cand_items)+']'
    # Keep enough context for a detailed report while avoiding context-limit 400s on long meetings.
    transcript='\n'.join(f'[{clock(x["timestamp"])}] {x["speaker"]}: {x["text"]}' for x in rows)[-80000:]
    system=('You are SessionMind, an evidence-first meeting analyst. Analyze the complete meeting transcript. Do not invent facts. '
            'Suggestions are not decisions; only treat a decision as confirmed when agreement/confirmation is explicit. '
            'Every extracted item must be traceable to transcript timestamps. Use "Unknown" when ownership/deadline is not stated. '
            'OPEN QUESTIONS: list every open question, unresolved issue, deferred item and unanswered point in the meeting, not just a few. '
            'You are given CANDIDATE OPEN QUESTIONS from a full-transcript scan and live detection. Keep each candidate unless the transcript clearly resolves it '
            '(then status "Answered" and the answer as evidence). You may reword for clarity and merge true duplicates, and you should add any other unresolved question you find. '
            'Status must be one of: Unanswered, Partially answered, Deferred, Answered. '
            'MIND MAP: "mind_map" is a tree for a mind-map view. root = the meeting title. branches = 4 to 8 main themes. '
            'Each theme node has children such as its key points, decisions, action items (with owner), risks and open questions, '
            'written as short labels of at most 12 words. Max 3 levels below the root. Only use what the transcript supports. '
            'Return ONLY valid JSON with this exact shape: '
            '{"executive_summary":"","meeting_overview":"","key_discussion_points":[""],"decisions":[{"decision":"","evidence":""}],'
            '"action_items":[{"owner":"","task":"","deadline":"","status":"Assigned","evidence":""}],'
            '"open_questions":[{"question":"","raised_by":"","timestamp":"","status":"Unanswered","evidence":""}],'
            '"risks":[{"risk":"","evidence":""}],"topics":[{"name":"","summary":"","evidence":""}],'
            '"mind_map":{"root":"","branches":[{"label":"","children":[{"label":"","children":[]}]}]},'
            '"important_moments":[{"timestamp":"","description":"","evidence":""}],"changes_from_previous_meeting":[""],"participants":[""]}. '
            'Keep it concise but comprehensive.')
    user=(f'Meeting: {m["title"]}\nTranscript:\n{transcript}\n\n'
          f'CANDIDATE OPEN QUESTIONS (JSON):\n{cand_json}\n\n'
          f'Previous project context:\n{previous_context(mid)}')
    rep=groq_json(system,user)
    # Safety net: never drop a candidate the model left out.
    rep['open_questions']=merge_questions([q for q in (clean_question(x) for x in (rep.get('open_questions') or [])) if q], candidates)
    for ev in rep.get('action_items',[]):
        ev.setdefault('status','Assigned')
    rep.update({'meeting_id':mid,'title':m['title'],'project':p['name'],'started_at':m['started_at'],'ended_at':m['ended_at'],'screenshot_count':len(s)})
    tree=build_mind_map(rep)
    html_path=DATA/f'report_{mid}.html'; docx_path=DATA/f'report_{mid}.docx'
    render_html(rep,rows,shots,html_path,tree); render_docx(rep,rows,shots,docx_path,BASE,tree)
    c=db(); c.execute('UPDATE meetings SET report=? WHERE id=?',(json.dumps(rep,ensure_ascii=False),mid)); c.execute('INSERT INTO reports(meeting_id,html_path,docx_path,created_at) VALUES(?,?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET html_path=excluded.html_path,docx_path=excluded.docx_path,created_at=excluded.created_at',(mid,str(html_path.relative_to(BASE)),str(docx_path.relative_to(BASE)),now())); c.commit(); c.close()
    return {**rep,'mind_map_tree':tree,'files':{'html':f'/api/reports/{mid}/html','docx':f'/api/reports/{mid}/docx'}}

@app.get('/api/reports/{mid}/docx')
def docx_report(mid:int):
    p=DATA/f'report_{mid}.docx'
    if not p.exists():raise HTTPException(404,'Report not found')
    return FileResponse(p,media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',filename=f'sessionmind_{mid}.docx')
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
