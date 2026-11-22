#!/usr/bin/env python3
"""Measure how many of the labelled tasks, decisions and open questions your configured model finds.

    python scripts/eval_extraction.py                       # built-in sample meeting
    python scripts/eval_extraction.py my_meeting.json       # your own labelled transcript
    python scripts/eval_extraction.py --verbose

It uses whatever LLM_PROVIDER / OLLAMA_MODEL / GROQ_* you have in .env, so you can change
the model, re-run, and compare. Matching is fuzzy (shared meaningful words), so rewording is fine.

Labelled file format (see tests/data/sample_meeting.json):
    {"lines": ["text of line 1", ...], "seconds_per_line": 12, "speaker": "Meeting Speaker",
     "expected": {"decisions": ["..."], "action_items": ["..."], "open_questions": ["..."]}}
Write each expected item as a short phrase with its key words (names, objects, numbers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.report_render import content_tokens  # noqa: E402
from app.services.extraction import scan_transcript  # noqa: E402

DEFAULT = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'sample_meeting.json'
FIELDS = (('action_items', 'actions', 'task'), ('decisions', 'decisions', 'decision'), ('open_questions', 'questions', 'question'))


def _overlap(expected: str, found: str) -> float:
    want = set(content_tokens(expected))
    return len(want & set(content_tokens(found))) / len(want) if want else 0.0


def evaluate(llm, data: dict, threshold: float = 0.5) -> dict:
    step = data.get('seconds_per_line', 12)
    rows = [{'timestamp': float(i * step), 'abs': float(i * step), 'speaker': data.get('speaker', 'Meeting Speaker'), 'text': t}
            for i, t in enumerate(data['lines'])]
    started = time.time()
    scan = scan_transcript(llm, rows)
    report = {'seconds': round(time.time() - started, 1), 'windows': scan.windows, 'failed_windows': scan.failed,
              'dropped_ungrounded': scan.dropped, 'categories': {}}
    for label, attr, key in FIELDS:
        found = getattr(scan, attr)
        texts = [f"{x[key]} {x.get('evidence', '')}" for x in found]
        hits, missed, used = [], [], set()
        for exp in data['expected'].get(label, []):
            scores = [(_overlap(exp, t), i) for i, t in enumerate(texts)]
            best, idx = max(scores, default=(0.0, -1))
            (hits if best >= threshold else missed).append(exp)
            if best >= threshold:
                used.add(idx)
        extras = [found[i][key] for i in range(len(found)) if i not in used]
        total = len(data['expected'].get(label, []))
        report['categories'][label] = {'expected': total, 'found_expected': len(hits), 'recall': round(len(hits) / total, 2) if total else None,
                                       'missed': missed, 'extra_items': extras}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=str(DEFAULT))
    ap.add_argument('--verbose', '-v', action='store_true', help='also list extra items the model reported')
    ap.add_argument('--json', help='write the full result to this file')
    args = ap.parse_args()

    from app import providers
    llm = providers.get_llm()
    health = llm.health()
    if not health['ok']:
        sys.exit(f"LLM not ready: {health['detail']}")
    data = json.loads(Path(args.file).read_text(encoding='utf-8'))
    print(f'Model: {llm.describe()}   transcript lines: {len(data["lines"])}')
    rep = evaluate(llm, data)
    print(f"{rep['windows']} window(s) in {rep['seconds']}s, {rep['failed_windows']} failed, {rep['dropped_ungrounded']} ungrounded items dropped\n")
    for label, c in rep['categories'].items():
        pct = 'n/a' if c['recall'] is None else f"{int(c['recall'] * 100)}%"
        print(f"{label:15s} found {c['found_expected']}/{c['expected']}  recall {pct}")
        for m in c['missed']:
            print(f'   MISSED: {m}')
        if args.verbose:
            for e in c['extra_items']:
                print(f'   extra : {e}')
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
