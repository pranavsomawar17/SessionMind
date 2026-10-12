"""Report layout: mind-map tree building, Word (.docx) and HTML rendering.

Kept free of FastAPI/Groq imports so it can be tested on its own.
"""
import html
from pathlib import Path

MAX_TREE_DEPTH = 4
MAX_TREE_CHILDREN = 15


def H(x):
    return html.escape(str(x or ''))


def clock(sec):
    """Meeting-relative time as mm:ss (or h:mm:ss)."""
    try:
        sec = max(0, int(float(sec)))
    except Exception:
        return str(sec or '')
    h, rem = divmod(sec, 3600)
    mi, s_ = divmod(rem, 60)
    return f'{h}:{mi:02d}:{s_:02d}' if h else f'{mi:02d}:{s_:02d}'


def nice_dt(v):
    """ISO timestamp -> '12 Oct 2026, 18:07' in the server's local time zone."""
    from datetime import datetime
    try:
        d = datetime.fromisoformat(str(v))
        if d.tzinfo is not None:
            d = d.astimezone()
        return d.strftime('%d %b %Y, %H:%M')
    except Exception:
        return str(v or '')


def _clip(s, n=160):
    s = ' '.join(str(s or '').split())
    return s if len(s) <= n else s[:n - 1].rstrip() + '…'


# ---------------------------------------------------------------- mind map / tree

def _clean_node(n, depth=1):
    if isinstance(n, dict):
        label = _clip(n.get('label') or n.get('name') or n.get('text'), 120)
        raw_children = n.get('children') or []
    else:
        label = _clip(n, 120)
        raw_children = []
    if not label:
        return None
    kids = []
    if depth < MAX_TREE_DEPTH and isinstance(raw_children, list):
        for c in raw_children[:MAX_TREE_CHILDREN]:
            k = _clean_node(c, depth + 1)
            if k:
                kids.append(k)
    return {'label': label, 'children': kids}


def _fallback_tree(rep):
    """Used when the model returned no usable mind_map (or for older reports).
    Builds the tree from the report's own sections so it is never empty."""
    branches = []
    for t in (rep.get('topics') or [])[:12]:
        if not isinstance(t, dict) or not t.get('name'):
            continue
        kids = []
        if t.get('summary'):
            kids.append({'label': t['summary'], 'children': []})
        branches.append({'label': t['name'], 'children': kids})
    sections = [
        ('Decisions', [d.get('decision') for d in rep.get('decisions', []) if isinstance(d, dict)]),
        ('Action items', [f"{a.get('owner') or 'Unknown'}: {a.get('task') or ''}"
                          for a in rep.get('action_items', []) if isinstance(a, dict)]),
        ('Open questions', [q.get('question') for q in rep.get('open_questions', []) if isinstance(q, dict)]),
        ('Risks / blockers', [r.get('risk') for r in rep.get('risks', []) if isinstance(r, dict)]),
    ]
    for name, items in sections:
        items = [i for i in items if i]
        if items:
            branches.append({'label': name, 'children': [{'label': i, 'children': []} for i in items]})
    return {'label': _clip(rep.get('title') or 'Meeting', 120), 'children': branches}


def build_mind_map(rep):
    """Return a tree {'label', 'children'}: root = meeting, branches = themes,
    children = points/decisions/tasks/questions under each theme."""
    mm = rep.get('mind_map')
    if isinstance(mm, dict):
        branches = []
        for b in (mm.get('branches') or [])[:12]:
            n = _clean_node(b, 1)
            if n:
                branches.append(n)
        if branches:
            return {'label': _clip(mm.get('root') or rep.get('title') or 'Meeting', 120), 'children': branches}
    return _fallback_tree(rep)


def tree_lines(tree):
    """Plain-text tree with box-drawing connectors, for Word and logs."""
    lines = [tree['label']]

    def walk(node, prefix):
        kids = node.get('children') or []
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            lines.append(prefix + ('└── ' if last else '├── ') + k['label'])
            walk(k, prefix + ('    ' if last else '│   '))

    walk(tree, '')
    return lines


# ---------------------------------------------------------------- shared row helpers

def question_rows(rep):
    out = []
    for q in rep.get('open_questions') or []:
        if isinstance(q, dict):
            out.append({
                'question': q.get('question') or '',
                'status': q.get('status') or 'Unanswered',
                'raised': q.get('raised_by') or q.get('timestamp') or '',
                'evidence': q.get('evidence') or '',
            })
        elif q:
            out.append({'question': str(q), 'status': 'Unanswered', 'raised': '', 'evidence': ''})
    return out


def _status_counts(rows):
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    return counts


# ---------------------------------------------------------------- Word (.docx)

def render_docx(rep, transcript, screenshots, path, base_dir, tree=None):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    tree = tree or build_mind_map(rep)
    doc = Document()

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
    normal.paragraph_format.space_after = Pt(4)
    for name, size in (('Heading 1', 16), ('Heading 2', 13), ('Title', 22)):
        st = doc.styles[name]
        st.font.name = 'Calibri'
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)

    # Footer with page number field.
    fp = sec.footer.paragraphs[0]
    fp.add_run('SessionMind report · page ').font.size = Pt(8)
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'PAGE')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = '1'
    r.append(t)
    fld.append(r)
    fp._p.append(fld)

    def para(text='', bold=False, italic=False, size=None, color=None, style=None):
        p = doc.add_paragraph(style=style)
        run = p.add_run(text or '')
        run.bold, run.italic = bold, italic
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
        return p

    def bullets(items, empty='None recorded.'):
        items = [i for i in items if i not in (None, '')]
        if not items:
            para(empty, italic=True, color=(0x66, 0x66, 0x66))
        for i in items:
            doc.add_paragraph(str(i), style='List Bullet')

    def shade(cell, fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), fill)
        tcPr.append(shd)

    def table(headers, rows, widths_cm, empty_text):
        if not rows:
            para(empty_text, italic=True, color=(0x66, 0x66, 0x66))
            return
        tb = doc.add_table(rows=1, cols=len(headers))
        tb.style = 'Table Grid'
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tb.autofit = False
        hdr = tb.rows[0]
        # Repeat the header row on each page.
        trPr = hdr._tr.get_or_add_trPr()
        th = OxmlElement('w:tblHeader')
        th.set(qn('w:val'), 'true')
        trPr.append(th)
        for i, h in enumerate(headers):
            c = hdr.cells[i]
            c.width = Cm(widths_cm[i])
            c.text = ''
            run = c.paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(9.5)
            shade(c, 'E8EDF5')
        for row in rows:
            cells = tb.add_row().cells
            for i, val in enumerate(row):
                cells[i].width = Cm(widths_cm[i])
                cells[i].text = ''
                run = cells[i].paragraphs[0].add_run(str(val or ''))
                run.font.size = Pt(9.5)
        for i, w in enumerate(widths_cm):
            tb.columns[i].width = Cm(w)

    # ---- title block
    doc.add_paragraph(rep.get('title') or 'Meeting', style='Title')
    para(f"SessionMind AI Meeting Report · Project: {rep.get('project', '')}", color=(0x55, 0x55, 0x55))
    para(f"Started: {nice_dt(rep.get('started_at'))}   Ended: {nice_dt(rep.get('ended_at')) if rep.get('ended_at') else '—'}",
         size=9, color=(0x55, 0x55, 0x55))

    doc.add_heading('Executive Summary', level=1)
    para(rep.get('executive_summary') or 'No summary generated.')

    doc.add_heading('Meeting Overview', level=1)
    para(rep.get('meeting_overview') or '')

    doc.add_heading('Participants', level=1)
    bullets(rep.get('participants') or [], empty='Unknown (not stated in transcript).')

    doc.add_heading('Key Discussion Points', level=1)
    bullets(rep.get('key_discussion_points') or [])

    doc.add_heading('Confirmed Decisions', level=1)
    decisions = [d for d in (rep.get('decisions') or []) if isinstance(d, dict)]
    if decisions:
        for d in decisions:
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(d.get('decision') or '').bold = True
            if d.get('evidence'):
                ev = p.add_run(f"  — {d['evidence']}")
                ev.italic = True
                ev.font.size = Pt(9)
    else:
        para('None recorded.', italic=True, color=(0x66, 0x66, 0x66))

    doc.add_heading('Action Items', level=1)
    tasks = [a for a in (rep.get('action_items') or []) if isinstance(a, dict)]
    table(['Owner', 'Task', 'Deadline', 'Status'],
          [[a.get('owner') or 'Unknown', a.get('task') or '', a.get('deadline') or 'Unknown',
            a.get('status') or 'Assigned'] for a in tasks],
          [3.2, 8.6, 2.9, 2.3], 'No action items recorded.')

    qrows = question_rows(rep)
    doc.add_heading('Open Questions', level=1)
    if qrows:
        counts = _status_counts(qrows)
        summary = ' · '.join(f'{v} {k.lower()}' for k, v in sorted(counts.items()))
        para(f'{len(qrows)} question(s) recorded — {summary}.', size=9.5, color=(0x55, 0x55, 0x55))
    table(['#', 'Question', 'Raised', 'Status', 'Evidence'],
          [[i + 1, q['question'], q['raised'], q['status'], q['evidence']] for i, q in enumerate(qrows)],
          [0.9, 6.0, 2.4, 2.6, 5.1], 'No open questions recorded.')

    doc.add_heading('Risks / Blockers', level=1)
    bullets([f"{r.get('risk')} — {r.get('evidence', '')}" if isinstance(r, dict) else str(r)
             for r in (rep.get('risks') or []) if r])

    doc.add_heading('Mind Map (Topic Tree)', level=1)
    intro = para('Root: meeting · branches: themes · leaves: points, decisions, tasks and questions.',
                 size=9, color=(0x55, 0x55, 0x55))
    intro.paragraph_format.keep_with_next = True
    lines = tree_lines(tree)
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = i < len(lines) - 1  # keep the whole tree on one page
        run = p.add_run(line)
        run.font.name = 'Consolas'
        run.font.size = Pt(9.5)
        run.bold = (i == 0)
        run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
    doc.add_paragraph()

    doc.add_heading('Important Moments', level=1)
    moments = [m for m in (rep.get('important_moments') or []) if isinstance(m, dict)]
    bullets([f"{m.get('timestamp', '')} — {m.get('description', '')}"
             + (f" [{m['evidence']}]" if m.get('evidence') else '') for m in moments])

    doc.add_heading('Changes From Previous Meeting', level=1)
    bullets(rep.get('changes_from_previous_meeting') or [], empty='No previous meeting available.')

    doc.add_heading('Meeting Screenshots', level=1)
    shown = 0
    for s in screenshots[:12]:
        p = Path(base_dir) / s['path']
        if not p.exists():
            continue
        try:
            doc.add_picture(str(p), width=Cm(15))
            para(f"Captured at {clock(s['timestamp'])}", size=8, color=(0x77, 0x77, 0x77))
            shown += 1
        except Exception:
            pass
    if not shown:
        para('No screenshots captured.', italic=True, color=(0x66, 0x66, 0x66))

    doc.add_heading('Evidence Transcript', level=1)
    for x in transcript:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        ts = p.add_run(f"[{clock(x['timestamp'])}] {x['speaker']}  ")
        ts.bold = True
        ts.font.size = Pt(9)
        ts.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        body = p.add_run(x['text'])
        body.font.size = Pt(10)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


# ---------------------------------------------------------------- HTML

def _tree_html(node):
    kids = node.get('children') or []
    inner = f'<ul>{"".join(_tree_html(k) for k in kids)}</ul>' if kids else ''
    return f'<li><span>{H(node["label"])}</span>{inner}</li>'


def tree_html(tree):
    return f'<div class="tree"><ul>{_tree_html(tree)}</ul></div>'


HTML_CSS = (
    'body{font:15px system-ui;max-width:1000px;margin:40px auto;color:#18202a;line-height:1.55;padding:0 16px}'
    'h1{margin-bottom:4px}h2{margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:7px}'
    'table{width:100%;border-collapse:collapse}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}'
    'th{background:#eef2f8}'
    'figure{display:inline-block;width:46%;vertical-align:top}img{max-width:100%;border:1px solid #ddd}'
    '.tree ul{list-style:none;margin:0;padding-left:22px;border-left:2px solid #c9d2e0}'
    '.tree>ul{border-left:none;padding-left:0}'
    '.tree li{position:relative;padding:4px 0 4px 14px}'
    '.tree li::before{content:"";position:absolute;left:0;top:14px;width:12px;border-top:2px solid #c9d2e0}'
    '.tree>ul>li::before{display:none}'
    '.tree span{display:inline-block;border:1px solid #aab6c8;border-radius:8px;padding:3px 10px;background:#f7f9fc}'
    '.tree>ul>li>span{font-weight:700;background:#e8eefc}'
)


def render_html(rep, transcript, screenshots, path, tree=None):
    tree = tree or build_mind_map(rep)
    tasks = [a for a in (rep.get('action_items') or []) if isinstance(a, dict)]
    qrows = question_rows(rep)

    task_rows = ''.join(
        f'<tr><td>{H(x.get("owner"))}</td><td>{H(x.get("task"))}</td><td>{H(x.get("deadline"))}</td><td>{H(x.get("status"))}</td></tr>'
        for x in tasks) or '<tr><td colspan="4">No action items recorded.</td></tr>'
    q_rows = ''.join(
        f'<tr><td>{i + 1}</td><td>{H(q["question"])}</td><td>{H(q["raised"])}</td><td>{H(q["status"])}</td><td>{H(q["evidence"])}</td></tr>'
        for i, q in enumerate(qrows)) or '<tr><td colspan="5">No open questions recorded.</td></tr>'
    shots = ''.join(
        f'<figure><img src="/media/{H(Path(x["path"]).name)}"><figcaption>{clock(x["timestamp"])}</figcaption></figure>'
        for x in screenshots)

    def bullets(a, key=None):
        return ''.join('<li>' + H(x if key is None else x.get(key, '')) + '</li>' for x in a) or '<li>None recorded.</li>'

    transcript_html = ''.join(
        f'<p><b>[{clock(x["timestamp"])}] {H(x["speaker"])}</b> — {H(x["text"])}</p>' for x in transcript)
    decisions = ''.join(f'<li><b>{H(x.get("decision"))}</b> — {H(x.get("evidence"))}</li>'
                        for x in rep.get('decisions', []) if isinstance(x, dict)) or '<li>None recorded.</li>'
    risks = ''.join(f'<li>{H(x.get("risk"))} — {H(x.get("evidence"))}</li>'
                    for x in rep.get('risks', []) if isinstance(x, dict)) or '<li>None recorded.</li>'
    moments = ''.join(f'<li><b>{H(x.get("timestamp"))}</b> — {H(x.get("description"))} [{H(x.get("evidence"))}]</li>'
                      for x in rep.get('important_moments', []) if isinstance(x, dict)) or '<li>None recorded.</li>'

    body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{H(rep.get("title"))} - SessionMind</title><style>{HTML_CSS}</style></head><body><h1>{H(rep.get("title"))}</h1><div>SessionMind AI Meeting Report · {H(rep.get("project"))}</div><h2>Executive Summary</h2><p>{H(rep.get("executive_summary"))}</p><h2>Meeting Overview</h2><p>{H(rep.get("meeting_overview"))}</p><h2>Participants</h2><ul>{bullets(rep.get("participants", []))}</ul><h2>Key Discussion Points</h2><ul>{bullets(rep.get("key_discussion_points", []))}</ul><h2>Confirmed Decisions</h2><ul>{decisions}</ul><h2>Action Items</h2><table><tr><th>Owner</th><th>Task</th><th>Deadline</th><th>Status</th></tr>{task_rows}</table><h2>Open Questions ({len(qrows)})</h2><table><tr><th>#</th><th>Question</th><th>Raised</th><th>Status</th><th>Evidence</th></tr>{q_rows}</table><h2>Risks / Blockers</h2><ul>{risks}</ul><h2>Mind Map (Topic Tree)</h2>{tree_html(tree)}<h2>Important Moments</h2><ul>{moments}</ul><h2>Changes From Previous Meeting</h2><ul>{bullets(rep.get("changes_from_previous_meeting", []))}</ul><h2>Meeting Screenshots</h2>{shots or '<p>No screenshots captured.</p>'}<h2>Evidence Transcript</h2>{transcript_html}</body></html>'''
    Path(path).write_text(body, encoding='utf-8')
