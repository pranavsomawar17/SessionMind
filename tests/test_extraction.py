import json

import pytest

from app.providers.base import LLMProvider
from app.services import extraction as ex
from tests.fakes import LINES, SAMPLE, ScriptedLLM, sample_rows


# ---------------------------------------------------------------- cues
@pytest.mark.parametrize('text,expected', [
    ('Priya, can you send the contract?', 'QT'),
    ("I'll send it by Friday.", 'T'),
    ('I’ll send it by Friday.', 'T'),                 # curly apostrophe, as Whisper often writes it
    ('Who owns the blog content right now?', 'Q'),
    ("I don't know yet.", 'Q'),
    ("Let's park that and discuss it next week.", 'QT'),
    ('We are about eighty percent done.', ''),
    ('So we ship on Friday, right?', ''),             # tag question is not a real question
])
def test_cue_flags(text, expected):
    assert ''.join(sorted(ex.flags(text))) == ''.join(sorted(expected))


# ---------------------------------------------------------------- windows
def test_windows_cover_every_row_exactly_once_and_respect_the_size():
    rows = sample_rows()
    wins = ex.make_windows(rows, 600)
    assert len(wins) > 2
    covered = [i for w in wins for i in range(w.start, w.end)]
    assert covered == list(range(len(rows)))
    for w in wins:
        assert sum(len(ex.format_line(i + 1, rows[i])) + 1 for i in range(w.start, w.end)) <= 600 or w.end - w.start == 1


def test_windows_have_context_on_both_sides_but_not_past_the_ends():
    rows = sample_rows()
    wins = ex.make_windows(rows, 600)
    first, mid, last = wins[0], wins[1], wins[-1]
    assert first.ctx_before == 0 and mid.ctx_before == mid.start - 4
    assert last.ctx_after == len(rows) and mid.ctx_after == mid.end + 3


def test_live_windows_start_after_the_context_rows():
    rows = sample_rows()
    wins = ex.make_windows(rows, 10_000, start_at=20)
    assert [(w.start, w.end, w.ctx_before) for w in wins] == [(20, 26, 16)]


def test_rendered_window_has_numbers_flags_and_markers():
    rows = sample_rows()
    wins = ex.make_windows(rows, 600)
    text, by_n, _ = ex.render_window(rows, wins[1])
    assert 'CONTEXT BEFORE' in text and '--- ANALYSE FROM HERE ---' in text and 'CONTEXT AFTER' in text
    assert '#4 [00:36]' in text or '#5 [00:48]' in text
    assert '\u2691Q' in text and '\u2691T' in text
    assert set(by_n) == set(range(wins[1].ctx_before + 1, wins[1].ctx_after + 1))


# ---------------------------------------------------------------- full scan with a scripted model
def _scan(llm=None):
    llm = llm or ScriptedLLM()
    return llm, ex.scan_transcript(llm, sample_rows())


def test_scan_finds_every_expected_item():
    llm, res = _scan()
    assert llm.scan_calls >= 3                               # several windows, not one huge call
    assert len(res.decisions) == 3 and len(res.actions) == 7 and len(res.questions) == 4
    owners = sorted(a['owner'] for a in res.actions)
    assert owners == ['Dan', 'Legal team', 'Meera', 'Priya', 'Rahul', 'Rahul', 'Unknown']
    assert res.failed == 0 and res.warnings() == []


def test_items_reported_twice_through_overlapping_context_are_merged():
    llm, res = _scan(ScriptedLLM(report_context=True))
    assert (len(res.decisions), len(res.actions), len(res.questions)) == (3, 7, 4)


def test_action_items_keep_owner_deadline_and_real_timestamp_and_evidence():
    _, res = _scan()
    a = next(x for x in res.actions if x['owner'] == 'Meera')
    assert a['deadline'] == 'Monday'
    assert a['timestamp'] == '02:36'                          # line 14 -> 13 * 12s
    assert 'finance' in a['evidence'].lower()
    unknown = next(x for x in res.actions if x['task'].startswith('Check with legal'))
    assert unknown['owner'] == 'Unknown' and unknown['status'] == 'Needs owner'


def test_questions_keep_status_and_results_are_in_time_order():
    _, res = _scan()
    assert {q['status'] for q in res.questions} == {'Unanswered', 'Deferred'}
    stamps = [x['timestamp'] for x in res.actions]
    assert stamps == sorted(stamps, key=ex.by_time.__globals__['seconds_of'])


def test_window_summaries_are_collected_for_long_meetings():
    _, res = _scan()
    assert len(res.summaries) == res.windows and res.summaries[0][0] == '00:00'


# ---------------------------------------------------------------- grounding
def test_invented_items_are_dropped():
    script = {4: ('task', {'task': 'Send the revised contract to the client', 'owner': 'Priya', 'deadline': 'Friday',
                           'evidence': "I'll send it by Friday"})}      # the prompt's own example, not in this transcript
    _, res = _scan(ScriptedLLM(script=script))
    assert res.actions == [] and res.dropped >= 1


def test_wrong_line_number_is_corrected_from_the_evidence():
    script = {1: ('task', {'task': 'Finish the checkout testing', 'owner': 'Rahul', 'deadline': 'next Wednesday',
                           'evidence': 'can you finish the checkout testing'})}
    llm = ScriptedLLM(script=script)
    orig = llm._complete

    def lying(system, user, schema, max_tokens):
        out = json.loads(orig(system, user, schema, max_tokens))
        for it in out['action_items']:
            it['line'] = 99                                    # model points at a line that is not in the slice
        return json.dumps(out)
    llm._complete = lying
    res = ex.scan_transcript(llm, sample_rows())
    assert [a['timestamp'] for a in res.actions] == ['00:36']  # found line 4 by matching the evidence text


def test_non_latin_transcripts_are_not_filtered_by_word_overlap():
    rows = [{'timestamp': 0.0, 'abs': 0.0, 'speaker': 'Meeting Speaker', 'text': 'राहुल, क्या आप कल तक रिपोर्ट भेज सकते हैं?'}]

    class Hindi(ScriptedLLM):
        def _complete(self, system, user, schema, max_tokens):
            return json.dumps({'decisions': [], 'open_questions': [], 'topic': '', 'summary': '',
                               'action_items': [{'task': 'Send the report by tomorrow', 'owner': 'Rahul', 'deadline': 'tomorrow', 'line': 1,
                                                 'evidence': 'क्या आप कल तक रिपोर्ट भेज सकते हैं'}]})
    res = ex.scan_transcript(Hindi(), rows)
    assert len(res.actions) == 1 and res.actions[0]['owner'] == 'Rahul'


def test_alternative_key_names_from_the_model_are_understood():
    class Sloppy(ScriptedLLM):
        def _complete(self, system, user, schema, max_tokens):
            return json.dumps({'tasks': [{'description': 'Update the project plan with the new dates', 'assignee': 'Dan', 'line': 22,
                                          'evidence': 'please update the project plan'}],
                               'questions': ['Who owns the blog content right now?'], 'decision': {'text': 'lock in March 15th'}})
    res = ex.scan_transcript(Sloppy(), sample_rows())
    assert any(a['owner'] == 'Dan' for a in res.actions)
    assert res.questions and res.decisions


# ---------------------------------------------------------------- failure handling
def test_a_failed_window_is_reported_and_covered_by_keyword_fallback():
    llm = ScriptedLLM(fail_scan=lambda user: '#14 ' in user.split('END OF SLICE')[0])
    _, res = _scan(llm)
    assert 0 < res.failed < res.windows
    assert 'failed for' in res.warnings()[0] and 'Unverified' in res.warnings()[0]
    # the window that failed contains "Meera, could you ask finance..."
    unverified = [a for a in res.actions if a['status'] == 'Unverified']
    assert any('finance' in a['task'] for a in unverified)
    # items from the windows that worked are still there
    assert any(a['owner'] == 'Rahul' for a in res.actions)


def test_total_failure_still_returns_keyword_items_and_a_clear_warning():
    llm = ScriptedLLM(fail_scan=lambda user: True)
    _, res = _scan(llm)
    assert res.failed == res.windows
    assert 'whole transcript' in res.warnings()[0]
    assert any(q['status'] == 'Unverified' for q in res.questions)
    assert res.actions


def test_heuristics_ignore_short_and_tag_questions():
    rows = [{'timestamp': 0, 'speaker': 'A', 'text': 'Right?'}, {'timestamp': 1, 'speaker': 'A', 'text': 'We ship Friday, right?'},
            {'timestamp': 2, 'speaker': 'Priya', 'text': 'Who is going to own the migration plan?'}]
    acts, qs = ex.heuristic_items(rows, ex.Window(0, 3, 0, 3))
    assert [q['question'] for q in qs] == ['Who is going to own the migration plan?']
    assert qs[0]['raised_by'] == 'Priya'


def test_invalid_json_from_the_model_is_retried_once():
    class Flaky(LLMProvider):
        name, model = 'flaky', 'x'
        scan_window_chars = 5000

        def __init__(self):
            super().__init__()
            self.n = 0

        def _complete(self, system, user, schema, max_tokens):
            self.n += 1
            return 'Sure! Here you go' if self.n == 1 else json.dumps({'decisions': [], 'action_items': [], 'open_questions': [], 'topic': '', 'summary': 'ok'})
    llm = Flaky()
    res = ex.scan_transcript(llm, sample_rows()[:3])
    assert res.failed == 0 and llm.n == 2
