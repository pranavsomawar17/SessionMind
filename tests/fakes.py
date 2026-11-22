"""Fake providers: deterministic stand-ins so the pipeline can be tested without any model."""
import json
import re
from pathlib import Path

from app.providers.base import LLMProvider, ProviderError, STTProvider

SAMPLE = json.loads((Path(__file__).parent / 'data' / 'sample_meeting.json').read_text(encoding='utf-8'))
LINES = SAMPLE['lines']
STEP = SAMPLE['seconds_per_line']

# What a good model would report for the sample meeting, keyed by transcript line number.
SCRIPT = {
    4: ('task', {'task': 'Finish the checkout testing', 'owner': 'Rahul', 'deadline': 'next Wednesday', 'evidence': 'can you finish the checkout testing'}),
    5: ('task', {'task': 'Write up the test report', 'owner': 'Rahul', 'deadline': 'Unknown', 'evidence': "I'll also write up the test report"}),
    7: ('decision', {'decision': 'Launch date is March 15', 'evidence': "let's lock in March 15th"}),
    10: ('question', {'question': 'What are the data residency implications of choosing Acme?', 'raised_by': '', 'status': 'Unanswered',
                      'evidence': 'what happens with our data residency requirements if we pick Acme'}),
    11: ('task', {'task': 'Find out the data residency implications of Acme and email everyone', 'owner': 'Priya', 'deadline': 'Friday',
                  'evidence': "I'll find out and email everyone by Friday"}),
    12: ('decision', {'decision': 'Pick CloudNine for hosting unless the data residency answer changes things',
                      'evidence': "Let's pick CloudNine unless the residency answer changes things"}),
    13: ('question', {'question': 'Is budget approval needed for CloudNine, which is above the original estimate?', 'raised_by': '',
                      'status': 'Unanswered', 'evidence': 'What about budget approval'}),
    14: ('task', {'task': 'Ask finance to approve the CloudNine quote', 'owner': 'Meera', 'deadline': 'Monday', 'evidence': 'could you ask finance to approve the CloudNine quote'}),
    16: ('question', {'question': 'Who owns the blog content migration?', 'raised_by': '', 'status': 'Deferred', 'evidence': 'Who owns the blog content right now'}),
    19: ('task', {'task': 'Legal team to review the new privacy policy', 'owner': 'Legal team', 'deadline': 'Unknown', 'evidence': 'the legal team still has to review the new privacy policy'}),
    20: ('question', {'question': 'Is a cookie consent banner needed for customers in India?', 'raised_by': '', 'status': 'Unanswered',
                      'evidence': 'do we need a cookie consent banner for customers in India'}),
    21: ('task', {'task': 'Check with legal whether a cookie consent banner is needed for India', 'owner': 'Unknown', 'deadline': 'Unknown',
                  'evidence': "Let's check with legal"}),
    22: ('task', {'task': 'Update the project plan with the new dates', 'owner': 'Dan', 'deadline': 'tomorrow', 'evidence': 'please update the project plan with the new dates'}),
    24: ('decision', {'decision': 'Do not migrate the old forum; archive it', 'evidence': 'we decided not to migrate the old forum'}),
}

NARRATIVE = {
    'executive_summary': 'The team agreed on a March 15 launch and CloudNine hosting.',
    'meeting_overview': 'Weekly sync on the website relaunch.',
    'participants': ['Rahul', 'Priya', 'Meera', 'Dan'],
    'key_discussion_points': ['Launch date', 'Hosting vendor', 'Content migration'],
    'risks': [{'risk': 'Data residency is unresolved', 'evidence': 'I don\'t know yet'}],
    'topics': [{'name': 'Hosting', 'summary': 'Acme vs CloudNine', 'evidence': ''}],
    'mind_map': {'root': 'Website relaunch', 'branches': [{'label': 'Launch date', 'children': [{'label': 'March 15', 'children': []}]}]},
    'important_moments': [{'timestamp': '01:12', 'description': 'Launch date locked', 'evidence': ''}],
    'changes_from_previous_meeting': [],
}


def sample_rows(speaker=None):
    sp = speaker or SAMPLE['speaker']
    return [{'timestamp': float(i * STEP), 'abs': 1000.0 + i * STEP, 'speaker': sp, 'text': t} for i, t in enumerate(LINES)]


class ScriptedLLM(LLMProvider):
    """Answers extraction prompts from SCRIPT, for every line number visible in the prompt
    (context lines included, so cross-window duplicates must be merged by the pipeline)."""
    name = 'scripted'
    model = 'test'
    scan_window_chars = 700
    max_input_chars = 60000

    def __init__(self, script=None, fail_scan=None, fail_narrative=False, report_context=False):
        super().__init__()
        self.script = SCRIPT if script is None else script
        self.report_context = report_context  # True = also report items seen only as context (a sloppy model)
        self.fail_scan = fail_scan          # callable(user_text) -> bool
        self.fail_narrative = fail_narrative
        self.scan_calls = 0
        self.narrative_calls = 0
        self.prompts = []

    def _complete(self, system, user, schema, max_tokens):
        self.prompts.append((system, user))
        if system.startswith('You are SessionMind'):
            self.narrative_calls += 1
            if self.fail_narrative:
                raise ProviderError('narrative model offline')
            return json.dumps(NARRATIVE)
        self.scan_calls += 1
        if self.fail_scan and self.fail_scan(user):
            raise ProviderError('scan model offline')
        scope = user
        if not self.report_context:   # a compliant model reports only the lines it was asked to analyse
            scope = user.split('--- ANALYSE FROM HERE ---', 1)[-1].split('--- END OF SLICE ---', 1)[0]
        # Like a real model, read the TEXT of each line (prompt numbering is relative to the slice).
        out = {'decisions': [], 'action_items': [], 'open_questions': [], 'topic': 'Test topic', 'summary': ''}
        seen = []
        for n, text in re.findall(r'^#(\d+) \[[^\]]*\] [^:]*: (.*?)(?: \u2691[QT])*$', scope, re.M):
            seen.append(int(n))
            if text in LINES and (LINES.index(text) + 1) in self.script:
                kind, item = self.script[LINES.index(text) + 1]
                out[{'decision': 'decisions', 'task': 'action_items', 'question': 'open_questions'}[kind]].append({**item, 'line': int(n)})
        if seen:
            out['summary'] = f'Lines {min(seen)}-{max(seen)}.'
        return json.dumps(out)

    def health(self):
        return {'ok': True, 'detail': 'scripted'}


class FakeSTT(STTProvider):
    name = 'fake'
    model = 'fake-1'

    def __init__(self, segments=None):
        self.segments = segments if segments is not None else [{'start': 0.0, 'end': 2.0, 'text': 'Hello from the fake transcriber.'},
                                                               {'start': 2.0, 'end': 4.0, 'text': 'Can you send the notes?'}]

    def transcribe(self, wav):
        return {'text': ' '.join(s['text'] for s in self.segments), 'duration': 4.0, 'segments': self.segments}

    def health(self):
        return {'ok': True, 'detail': 'fake'}
