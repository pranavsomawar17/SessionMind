"""Building the final meeting report.

Two separate jobs:

1. Items (decisions, action items, open questions) come from the windowed scan in
   ``extraction.py`` plus whatever the live panel found. They are merged in code, so nothing
   the scan found can be lost, and the model is not asked to re-type them.
2. The narrative (summary, discussion points, risks, mind map, ...) is one model call. If it
   fails, the report is still produced, with the items and a visible warning.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from .. import config, db
from ..providers import get_llm, get_stt
from ..report_render import (build_mind_map, by_time, clean_action, clean_decision, clean_question, clock, merge_items,
                             mind_view, render_docx, render_html)
from .extraction import GENERIC_SPEAKERS, clip, scan_transcript

log = logging.getLogger('sessionmind.report')

NARRATIVE_SYSTEM = (
    'You are SessionMind, an evidence-first meeting analyst. Write the narrative parts of a meeting report from the material provided. '
    'Do not invent facts; use only what the material supports. '
    'The decisions, action items and open questions have ALREADY been extracted and verified (you are shown them). Do not list them again; '
    'use them to write an accurate summary, risks and mind map. '
    'If presentation slides are provided, use them as context and cite them as "Slide N" where relevant. '
    'MIND MAP: "mind_map" describes the meeting. root = meeting title. branches = 3 to 8 main themes. Each theme has children: key points, decisions, '
    'action items (with owner), risks and open questions, as short labels of at most 12 words. Use 2 levels only when the meeting has real structure; '
    'if it is simple, themes can simply have no children. '
    'Write in the language of the transcript. '
    'Return ONLY valid JSON with this exact shape: '
    '{"executive_summary":"","meeting_overview":"","participants":[""],"key_discussion_points":[""],'
    '"risks":[{"risk":"","evidence":""}],"topics":[{"name":"","summary":"","evidence":""}],'
    '"mind_map":{"root":"","branches":[{"label":"","children":[{"label":"","children":[]}]}]},'
    '"important_moments":[{"timestamp":"","description":"","evidence":""}],"changes_from_previous_meeting":[""]}. '
    'Keep it concise but comprehensive.')


def previous_context(mid: int, limit: int) -> str:
    with db.session() as c:
        m = c.execute('SELECT project_id FROM meetings WHERE id=?', (mid,)).fetchone()
        if not m:
            return ''
        rows = c.execute('SELECT title,report FROM meetings WHERE project_id=? AND id<? AND status="ended" ORDER BY id DESC LIMIT 3',
                         (m['project_id'], mid)).fetchall()
    out = []
    for r in rows:
        if not r['report']:
            continue
        try:
            x = json.loads(r['report'])
        except ValueError:
            continue
        out.append({'title': r['title'], 'summary': x.get('executive_summary', ''), 'decisions': x.get('decisions', []),
                    'action_items': x.get('action_items', []), 'open_questions': x.get('open_questions', [])})
    return json.dumps(out, ensure_ascii=False)[:limit]


def hydrate_report(raw: str) -> dict:
    rep = json.loads(raw)
    tree = build_mind_map(rep)
    rep['mind_map_tree'] = tree
    rep['mind_view'] = mind_view(tree)
    return rep


def _event_items(events, rel):
    """Live-panel events as report items. Confirmed ones first so their wording wins a merge."""
    dec, act, qs = [], [], []
    for e in sorted(events, key=lambda e: e['status'] != 'confirmed'):
        if e['status'] == 'dismissed':
            continue
        try:
            detail = json.loads(e['detail']) if e['detail'] else {}
        except ValueError:
            detail = {}
        base = {'evidence': e['evidence'] or '', 'timestamp': detail.get('clock') or clock(rel(e['timestamp']))}
        if e['type'] == 'decision':
            dec.append(clean_decision({**base, 'decision': e['content']}))
        elif e['type'] == 'task':
            act.append(clean_action({**base, 'task': e['content'], 'owner': e['assignee'], 'deadline': e['deadline']}))
        elif e['type'] == 'question':
            qs.append(clean_question({**base, 'question': e['content'], 'status': detail.get('status') or 'Unanswered',
                                      'raised_by': detail.get('raised_by', '')}))
    return [x for x in dec if x], [x for x in act if x], [x for x in qs if x]


def _items_block(dec, act, qs, limit: int) -> str:
    def lines(items, fmt):
        out, used = [], 0
        for it in items:
            s = fmt(it)
            if used + len(s) > limit:
                out.append('- ... (more not shown)')
                break
            out.append(s)
            used += len(s) + 1
        return '\n'.join(out) or '- none'

    return ('EXTRACTED DECISIONS:\n' + lines(dec, lambda x: f"- [{x['timestamp']}] {x['decision']}")
            + '\n\nEXTRACTED ACTION ITEMS:\n' + lines(act, lambda x: f"- [{x['timestamp']}] {x['owner']}: {x['task']} (due: {x['deadline']})")
            + '\n\nEXTRACTED OPEN QUESTIONS:\n' + lines(qs, lambda x: f"- [{x['timestamp']}] {x['question']} ({x['status']})"))


def _material(rows, scan, budget: int) -> tuple[str, str | None]:
    """The transcript if it fits; otherwise section summaries; otherwise its beginning and end."""
    lines = [f"[{clock(x['timestamp'])}] {x['speaker']}: {x['text']}" for x in rows]
    full = '\n'.join(lines)
    if len(full) <= budget:
        return 'TRANSCRIPT:\n' + full, None
    if scan.summaries:
        text = '\n'.join(f'[{c}] {s}' for c, s in scan.summaries)
        return ('SECTION SUMMARIES (the meeting is too long to include in full; each line summarises a few minutes of it):\n'
                + text[:budget], 'The meeting was long, so the summary was written from section summaries rather than the full transcript.')
    half = budget // 2
    return ('TRANSCRIPT (beginning and end only; the middle was too long to include):\n' + full[:half]
            + '\n[... middle omitted ...]\n' + full[-half:],
            'The meeting was long and the summary only saw its beginning and end.')


def _text(v) -> str:
    return v.strip() if isinstance(v, str) else ''


def _str_list(v) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def build_report(mid: int) -> dict:
    llm = get_llm()
    with db.session() as c:
        m = c.execute('SELECT * FROM meetings WHERE id=?', (mid,)).fetchone()
        p = c.execute('SELECT * FROM projects WHERE id=?', (m['project_id'],)).fetchone()
        t = c.execute('SELECT * FROM transcript WHERE meeting_id=? ORDER BY timestamp,id', (mid,)).fetchall()
        s = c.execute('SELECT * FROM screenshots WHERE meeting_id=? ORDER BY timestamp', (mid,)).fetchall()
        e = c.execute('SELECT * FROM events WHERE meeting_id=? ORDER BY timestamp', (mid,)).fetchall()
        sl = c.execute('SELECT slide_no,title,text,notes FROM slides WHERE meeting_id=? ORDER BY slide_no', (mid,)).fetchall()
        mats = c.execute('SELECT kind,filename,detail FROM materials WHERE meeting_id=? ORDER BY id', (mid,)).fetchall()
    if not t:
        raise RuntimeError('No transcript was captured. Make sure the screen-share dialog has "Share audio" enabled, '
                           'or upload a recording, and let the meeting run for at least 10 seconds.')

    try:
        t0 = datetime.fromisoformat(m['started_at']).timestamp()
    except Exception:
        t0 = t[0]['timestamp']

    def rel(ts):
        return max(0.0, float(ts) - t0)

    rows = [{**dict(x), 'abs': x['timestamp'], 'timestamp': rel(x['timestamp'])} for x in t]
    shots = [{**dict(x), 'timestamp': rel(x['timestamp'])} for x in s]
    slides = [dict(x) for x in sl]
    warnings: list[str] = []

    # ---- 1. items: windowed scan + live panel, merged in code
    scan = scan_transcript(llm, rows)
    warnings += scan.warnings()
    live_dec, live_act, live_q = _event_items([dict(x) for x in e], rel)
    confirmed = {x['content'] for x in e if x['status'] == 'confirmed'}
    first = lambda items, key: [i for i in items if i[key] in confirmed]      # noqa: E731
    rest = lambda items, key: [i for i in items if i[key] not in confirmed]   # noqa: E731
    decisions = by_time(merge_items('decision', first(live_dec, 'decision'), scan.decisions, rest(live_dec, 'decision')))
    actions = by_time(merge_items('task', first(live_act, 'task'), scan.actions, rest(live_act, 'task'), owner_key='owner'))
    questions = by_time(merge_items('question', first(live_q, 'question'), scan.questions, rest(live_q, 'question')))

    # ---- 2. narrative: one call, with a fallback so the report always gets produced
    budget = int(llm.max_input_chars * 0.55)
    material, note = _material(rows, scan, budget)
    if note:
        warnings.append(note)
    slide_text = '\n'.join(f'Slide {x["slide_no"]} ({x["title"] or "untitled"}): {x["text"]}' for x in slides)[:int(llm.max_input_chars * 0.15)]
    user = (f'Meeting: {m["title"]}\n\n{material}\n\n'
            + (f'PRESENTATION SLIDES:\n{slide_text}\n\n' if slide_text else '')
            + _items_block(decisions, actions, questions, int(llm.max_input_chars * 0.08))
            + f'\n\nPrevious project context (for "changes from previous meeting"):\n{previous_context(mid, int(llm.max_input_chars * 0.12)) or "none"}')
    try:
        narrative = llm.chat_json(NARRATIVE_SYSTEM, user, max_tokens=llm.report_max_tokens)
        if not isinstance(narrative, dict):
            raise ValueError('the model returned a list instead of an object')
    except Exception as ex:
        log.warning('Narrative generation failed: %s', ex)
        narrative = {}
        warnings.append(f'The AI summary could not be generated ({clip(str(ex), 160)}). The decisions, action items and open '
                        'questions below were still extracted from the transcript.')

    speakers = sorted({r['speaker'] for r in rows if r['speaker'].lower() not in GENERIC_SPEAKERS})
    rep = {
        'executive_summary': _text(narrative.get('executive_summary')) or 'No summary was generated.',
        'meeting_overview': _text(narrative.get('meeting_overview')),
        'participants': _str_list(narrative.get('participants')) or speakers,
        'key_discussion_points': _str_list(narrative.get('key_discussion_points')),
        'risks': [r for r in (narrative.get('risks') or []) if isinstance(r, (dict, str))] if isinstance(narrative.get('risks'), list) else [],
        'topics': [x for x in (narrative.get('topics') or []) if isinstance(x, dict)] if isinstance(narrative.get('topics'), list) else [],
        'mind_map': narrative.get('mind_map') if isinstance(narrative.get('mind_map'), dict) else None,
        'important_moments': [x for x in (narrative.get('important_moments') or []) if isinstance(x, dict)]
        if isinstance(narrative.get('important_moments'), list) else [],
        'changes_from_previous_meeting': _str_list(narrative.get('changes_from_previous_meeting')),
        'decisions': decisions, 'action_items': actions, 'open_questions': questions,
    }
    log.info('Report %s: %d decisions, %d action items, %d open questions', mid, len(decisions), len(actions), len(questions))

    sources = [f'Live capture: {len(t)} transcript lines']
    for mt in mats:
        sources.append(f"{'Recording' if mt['kind'] == 'recording' else 'Slides'}: {mt['filename']} ({mt['detail']})")
    if slides and not any(mt['kind'] == 'slides' for mt in mats):
        sources.append(f'Slides: {len(slides)} slides')
    sources.append(f'Engines: speech-to-text {get_stt().describe()} · analysis {llm.describe()}')
    rep.update({'meeting_id': mid, 'title': m['title'], 'project': p['name'], 'started_at': m['started_at'], 'ended_at': m['ended_at'],
                'screenshot_count': len(s), 'slides': slides, 'sources': sources, 'warnings': warnings})

    tree = build_mind_map(rep)
    data = config.data_dir()
    html_path, docx_path = data / f'report_{mid}.html', data / f'report_{mid}.docx'
    render_html(rep, rows, shots, html_path, tree)
    render_docx(rep, rows, shots, docx_path, config.BASE, tree)
    with db.session() as c:
        c.execute('UPDATE meetings SET report=? WHERE id=?', (json.dumps(rep, ensure_ascii=False), mid))
        c.execute('INSERT INTO reports(meeting_id,html_path,docx_path,created_at) VALUES(?,?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET '
                  'html_path=excluded.html_path,docx_path=excluded.docx_path,created_at=excluded.created_at',
                  (mid, config.rel_to_base(html_path), config.rel_to_base(docx_path), db.now()))
    return {**rep, 'mind_map_tree': tree, 'mind_view': mind_view(tree),
            'files': {'html': f'/api/reports/{mid}/html', 'docx': f'/api/reports/{mid}/docx'}}
