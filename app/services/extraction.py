"""Finding action items, decisions and open questions in a transcript.

Why this is built the way it is (these were the reasons items went missing before):

* Small windows.  The model sees ~4-6k characters at a time instead of 24k. Recall drops
  sharply when a model has to list everything from a very long text.
* Context on every side.  Each window also shows a few lines before it (so "Sure, I'll do it"
  can be tied to the earlier "Priya, can you ...?") and after it (so a question answered two
  lines later is not reported as open).
* Cue hints.  A keyword detector flags lines that look like a question or a task. The model is
  told to check every flagged line. This mainly helps smaller local models.
* Grounding.  Every item must point at a real transcript line. Timestamps and evidence are taken
  from that line, not from what the model wrote, and items that do not appear in the transcript
  at all (hallucinations, or the prompt's own example) are dropped.
* No silent failures.  If a window cannot be analysed, a keyword fallback still records the
  obvious questions and requests, marked "Unverified", and the report says so.

This module has no database or web code, so it can be tested on its own.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..report_render import (by_time, clean_action, clean_decision, clean_question, clock, content_tokens,
                             merge_items)

log = logging.getLogger('sessionmind.extraction')

GENERIC_SPEAKERS = {'', 'meeting speaker', 'speaker', 'recording', 'screen'}
QUESTION_STATUSES = ('Unanswered', 'Partially answered', 'Deferred')

# ---------------------------------------------------------------- cue detection

_TAG_QUESTION = re.compile(r"[,\u2014-]\s*(right|okay|ok|yeah|yes|no|isn't it|aren't we|don't you think|you know|correct)\s*\?\s*$", re.I)
_UNCERTAIN = re.compile(
    r"\b(not sure|don't know|do not know|no idea|unclear|tbd|to be decided|to be determined|"
    r"yet to (decide|confirm|finali[sz]e)|haven't (decided|confirmed)|need(s)? to (check|confirm|find out|figure out|decide|clarify|verify)|"
    r"let'?s (revisit|park|discuss (it |this |that )?(later|next))|come back to|circle back|open question|still pending)\b", re.I)
_TASK = re.compile(
    r"\b(i'll|i will|i can (take|do|handle|check|send)|we'll|we will|can you|could you|would you|will you|please|"
    r"need(s)? to|has to|have to|must|should|let'?s|action item|follow[- ]?up|take care of|take (this|that|it) on|"
    r"assigned|responsible|owner|deadline|due|by (monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|eod|end of|next week|next month|the end))\b", re.I)
_STRONG_TASK = re.compile(
    r"\b(can you|could you|will you|would you|i'll|i will|we'll|we will|action item|take care of|follow[- ]?up|"
    r"assigned to|need(s)? to|has to|have to)\b", re.I)


def _plain(text: str) -> str:
    return str(text or '').replace('\u2019', "'").replace('\u2018', "'")


def flags(text: str) -> str:
    """'Q' if the line looks like a question / unresolved point, 'T' if it looks like a task."""
    t = _plain(text)
    out = ''
    if ('?' in t and len(t.split()) >= 3 and not _TAG_QUESTION.search(t)) or _UNCERTAIN.search(t):
        out += 'Q'
    if _TASK.search(t):
        out += 'T'
    return out


def clip(s, n=220) -> str:
    s = ' '.join(str(s or '').split())
    return s if len(s) <= n else s[:n - 1].rstrip() + '\u2026'


# ---------------------------------------------------------------- windows

@dataclass
class Window:
    start: int        # first row analysed (index into rows)
    end: int          # one past the last row analysed
    ctx_before: int   # first row shown as context before
    ctx_after: int    # one past the last row shown as context after


def format_line(n: int, row: dict) -> str:
    tag = ''.join(f' \u2691{c}' for c in flags(row['text']))
    return f"#{n} [{clock(row['timestamp'])}] {row['speaker']}: {row['text']}{tag}"


def make_windows(rows: list[dict], max_chars: int, start_at: int = 0, ctx_before: int = 4, ctx_after: int = 3) -> list[Window]:
    """Split rows[start_at:] into windows of about max_chars, each with context lines on both sides."""
    sizes = [len(format_line(i + 1, r)) + 1 for i, r in enumerate(rows)]
    out, i = [], start_at
    while i < len(rows):
        j, used = i, 0
        while j < len(rows) and (j == i or used + sizes[j] <= max_chars):
            used += sizes[j]
            j += 1
        out.append(Window(i, j, max(0, i - ctx_before), min(len(rows), j + ctx_after)))
        i = j
    return out


def render_window(rows: list[dict], win: Window):
    """-> (prompt text, {line number: row}, set of meaningful words in the window)."""
    by_n: dict[int, dict] = {}

    def block(a, b):
        lines = []
        for i in range(a, b):
            by_n[i + 1] = rows[i]
            lines.append(format_line(i + 1, rows[i]))
        return '\n'.join(lines)

    parts = []
    if win.ctx_before < win.start:
        parts.append('CONTEXT BEFORE (earlier lines, for reference only):\n' + block(win.ctx_before, win.start))
    parts.append('--- ANALYSE FROM HERE ---\n' + block(win.start, win.end) + '\n--- END OF SLICE ---')
    if win.ctx_after > win.end:
        parts.append('CONTEXT AFTER (later lines, only to see whether a question got answered or an item changed):\n'
                     + block(win.end, win.ctx_after))
    words = set(content_tokens(' '.join(f"{r['speaker']} {r['text']}" for r in by_n.values())))
    return '\n\n'.join(parts), by_n, words


# ---------------------------------------------------------------- prompt

SCAN_SYSTEM = '''You are the extraction engine of a meeting-minutes tool. You read one slice of a meeting transcript and list every ACTION ITEM, DECISION and OPEN QUESTION in it.
Recall matters: missing a real item is worse than including a borderline one. Never invent anything that is not in the transcript. Write item text in the language of the transcript.

Transcript lines look like: #12 [03:45] Speaker: text
The speaker label is often generic, so use the names people say in the conversation.
Lines ending in a flag were marked by a keyword detector: \u2691Q = possibly a question or unresolved point, \u2691T = possibly a task. Flags are only hints. Check every flagged line, and also read the unflagged ones.
Only report items from the lines between "ANALYSE FROM HERE" and "END OF SLICE". The CONTEXT sections are there so you can understand references ("that", "yes, I'll do it") and see whether a question was answered later.

ACTION ITEM = anything somebody is expected to do after the meeting:
- a direct request: "Priya, can you send the contract?" -> owner Priya
- a volunteer or commitment: "I'll check with finance" -> owner is the person speaking, if the transcript says who
- an obligation with no owner: "we need to update the roadmap", "someone should email the client" -> owner Unknown
- a request followed by an acceptance ("Can you...?" "Sure.") is ONE action item for the person asked
- a deadline may be anywhere in the sentence: "by Friday", "tomorrow", "next sprint"
Not an action item: work that is already finished, or an idea nobody took on and nobody agreed to.

DECISION = the group agreed on or settled something: "let's go with X", "agreed", "we'll use X", "approved", "we decided". A suggestion nobody agreed to is not a decision.

OPEN QUESTION = anything left unresolved:
- a question nobody answered, or answered only in part
- an answer like "not sure", "I don't know", "I'll find out"
- something postponed: "let's discuss it later", "park that"
- missing information: "we still need the budget number"
- an undecided choice ("A or B?" with no pick)
A question that is fully answered in the lines that follow is NOT open. "Right?" and "okay?" are not questions. One line can be both an action item and an open question.

For every item give "line": the number after # of the line where the item is clearest, and "evidence": a short quote copied from that line.
owner and deadline are "Unknown" when the transcript does not say.
question status is one of Unanswered, Partially answered, Deferred.
Also give "topic" (3-8 words: what is being discussed) and "summary" (one or two sentences about this slice).

Reply with ONLY one JSON object of this shape:
{"decisions":[{"decision":"","line":0,"evidence":""}],
 "action_items":[{"task":"","owner":"","deadline":"","line":0,"evidence":""}],
 "open_questions":[{"question":"","raised_by":"","status":"Unanswered","line":0,"evidence":""}],
 "topic":"","summary":""}
Use empty lists when a category has nothing.

EXAMPLE (only to show the format; never copy it into your answer)
#1 [00:05] Speaker: Priya, can you send the revised contract to the client?
#2 [00:09] Speaker: Sure, I'll send it by Friday.
#3 [00:14] Speaker: Okay. Which vendor are we using for hosting?
#4 [00:19] Speaker: Let's go with Acme, their price is better.
#5 [00:25] Speaker: Agreed. What about the security review though?
#6 [00:30] Speaker: Not sure, we should check with Dan whether it's needed.
Answer:
{"decisions":[{"decision":"Use Acme as the hosting vendor","line":4,"evidence":"Let's go with Acme"}],
 "action_items":[{"task":"Send the revised contract to the client","owner":"Priya","deadline":"Friday","line":2,"evidence":"I'll send it by Friday"},
  {"task":"Check with Dan whether a security review is needed","owner":"Unknown","deadline":"Unknown","line":6,"evidence":"we should check with Dan"}],
 "open_questions":[{"question":"Is a security review needed for the hosting vendor?","raised_by":"Unknown","status":"Unanswered","line":5,"evidence":"What about the security review though?"}],
 "topic":"Hosting vendor choice","summary":"The team picked Acme for hosting and Priya will send the contract; the security review question is open."}'''


def _item_schema(props: dict, required: list[str]) -> dict:
    return {'type': 'object', 'properties': props, 'required': required}


_S, _I = {'type': 'string'}, {'type': 'integer'}
SCAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'decisions': {'type': 'array', 'items': _item_schema({'decision': _S, 'line': _I, 'evidence': _S}, ['decision', 'line', 'evidence'])},
        'action_items': {'type': 'array', 'items': _item_schema(
            {'task': _S, 'owner': _S, 'deadline': _S, 'line': _I, 'evidence': _S}, ['task', 'owner', 'deadline', 'line', 'evidence'])},
        'open_questions': {'type': 'array', 'items': _item_schema(
            {'question': _S, 'raised_by': _S, 'status': {'type': 'string', 'enum': list(QUESTION_STATUSES)}, 'line': _I, 'evidence': _S},
            ['question', 'raised_by', 'status', 'line', 'evidence'])},
        'topic': _S,
        'summary': _S,
    },
    'required': ['decisions', 'action_items', 'open_questions', 'topic', 'summary'],
}


# ---------------------------------------------------------------- grounding

def _support(text: str, words: set[str]) -> float:
    """Share of the text's meaningful words that occur in the window."""
    toks = content_tokens(text)
    if not toks:
        return 1.0
    return sum(1 for t in toks if t in words) / len(toks)


def _mostly_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) >= 0.8


def _line_no(v):
    try:
        return int(float(str(v).strip().lstrip('#')))
    except (TypeError, ValueError):
        return None


def _locate(line_value, text: str, by_n: dict[int, dict]):
    """The transcript row an item came from: by its line number, else by best text match."""
    n = _line_no(line_value)
    if n in by_n:
        return by_n[n]
    toks = set(content_tokens(text))
    if not toks:
        return None
    best, best_score = None, 0.0
    for row in by_n.values():
        score = len(toks & set(content_tokens(row['text']))) / len(toks)
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= 0.5 else None


def _prepare(raw, cleaner, text_key: str, by_n: dict, words: set[str], latin: bool):
    item = cleaner(raw)
    if not item:
        return None
    desc, evidence = item[text_key], item.get('evidence', '')
    # Not in this transcript at all (hallucination, or the prompt's example repeated).
    # An item is kept when its own wording is mostly found in the window, or when its evidence
    # is a near-verbatim quote. Common words alone ("I'll", "Friday") are not enough.
    # Skipped for non-Latin scripts, where the model may describe items in another language.
    if latin:
        described = _support(desc, words) >= 0.4
        quoted = bool(content_tokens(evidence)) and _support(evidence, words) >= 0.8
        if not (described or quoted):
            log.info('Dropped ungrounded item: %s', clip(desc, 80))
            return None
    row = _locate(raw.get('line') if isinstance(raw, dict) else None, evidence or desc, by_n)
    if row is not None:
        item['timestamp'] = clock(row['timestamp'])
        item['abs_ts'] = row.get('abs')
        if not evidence or _support(evidence, set(content_tokens(row['text']))) < 0.5:
            item['evidence'] = clip(row['text'])
        if text_key == 'question' and not item.get('raised_by') and row['speaker'].lower() not in GENERIC_SPEAKERS:
            item['raised_by'] = row['speaker']
    elif item.get('timestamp') and ':' not in item['timestamp']:
        item['timestamp'] = ''
    if text_key == 'question':
        item['status'] = next((s for s in QUESTION_STATUSES if s.lower() == item['status'].lower()), item['status'])
    return item


def _pick_list(res: dict, *keys):
    for k in keys:
        v = res.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [v]
    return []


# ---------------------------------------------------------------- one window

@dataclass
class WindowResult:
    decisions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    summary: str = ''
    topic: str = ''
    dropped: int = 0


def extract_window(llm, rows: list[dict], win: Window, known: list[str] | None = None) -> WindowResult:
    text, by_n, words = render_window(rows, win)
    latin = _mostly_latin(' '.join(r['text'] for r in by_n.values()))
    user = text
    if known:
        user += '\n\nALREADY LISTED (do not repeat these):\n' + '\n'.join(f'- {k}' for k in known[-30:])
    res = llm.chat_json(SCAN_SYSTEM, user, schema=SCAN_SCHEMA, max_tokens=3000)
    if not isinstance(res, dict):
        res = {}
    out = WindowResult(summary=clip(res.get('summary'), 400), topic=clip(res.get('topic'), 80))
    for raw_list, cleaner, key, bucket in (
            (_pick_list(res, 'decisions', 'decision'), clean_decision, 'decision', out.decisions),
            (_pick_list(res, 'action_items', 'tasks', 'actions', 'action_item'), clean_action, 'task', out.actions),
            (_pick_list(res, 'open_questions', 'questions', 'unresolved_questions'), clean_question, 'question', out.questions)):
        for raw in raw_list:
            item = _prepare(raw, cleaner, key, by_n, words, latin)
            if item:
                bucket.append(item)
            else:
                out.dropped += 1
    return out


# ---------------------------------------------------------------- keyword fallback

def heuristic_items(rows: list[dict], win: Window):
    """Last resort when the model failed on a window: obvious questions and requests, marked Unverified."""
    acts, qs = [], []
    for i in range(win.start, win.end):
        row = rows[i]
        text, f = row['text'], flags(row['text'])
        words = len(text.split())
        base = {'timestamp': clock(row['timestamp']), 'evidence': clip(text), 'abs_ts': row.get('abs')}
        if 'Q' in f and '?' in text and words >= 5:
            qs.append(clean_question({**base, 'question': clip(text, 200), 'status': 'Unverified',
                                      'raised_by': '' if row['speaker'].lower() in GENERIC_SPEAKERS else row['speaker']}))
        if 'T' in f and words >= 4 and _STRONG_TASK.search(_plain(text)):
            acts.append(clean_action({**base, 'task': clip(text, 200), 'status': 'Unverified'}))
    return [a for a in acts if a], [q for q in qs if q]


# ---------------------------------------------------------------- whole transcript

@dataclass
class ScanResult:
    decisions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    summaries: list = field(default_factory=list)   # [(clock, one-line summary)]
    windows: int = 0
    failed: int = 0
    dropped: int = 0
    errors: list = field(default_factory=list)

    def warnings(self) -> list[str]:
        if not self.failed:
            return []
        err = self.errors[0] if self.errors else 'unknown error'
        if self.failed >= self.windows:
            return [f'The AI analysis failed for the whole transcript ({err}). The items below come from keyword matching only '
                    'and are marked Unverified. Check your provider settings and generate the report again.']
        return [f'The AI analysis failed for {self.failed} of {self.windows} transcript sections ({err}). Those sections were '
                'checked by keyword matching only; items marked Unverified need a manual look.']


def scan_transcript(llm, rows: list[dict]) -> ScanResult:
    """Analyse the whole transcript window by window and merge the findings."""
    res = ScanResult()
    decs, acts, qs = [], [], []
    for win in make_windows(rows, llm.scan_window_chars):
        res.windows += 1
        try:
            r = extract_window(llm, rows, win)
        except Exception as e:  # provider down, rate limited, unusable reply...
            res.failed += 1
            res.errors.append(clip(str(e), 160))
            log.warning('Extraction failed for lines %s-%s: %s', win.start + 1, win.end, e)
            a, q = heuristic_items(rows, win)
            acts.append(a)
            qs.append(q)
            continue
        decs.append(r.decisions)
        acts.append(r.actions)
        qs.append(r.questions)
        res.dropped += r.dropped
        if r.summary:
            res.summaries.append((clock(rows[win.start]['timestamp']), r.summary))
    res.decisions = by_time(merge_items('decision', *decs))
    res.actions = by_time(merge_items('task', *acts, owner_key='owner'))
    res.questions = by_time(merge_items('question', *qs))
    log.info('Scan: %d window(s), %d failed, %d ungrounded dropped -> %d decisions, %d tasks, %d questions',
             res.windows, res.failed, res.dropped, len(res.decisions), len(res.actions), len(res.questions))
    return res
