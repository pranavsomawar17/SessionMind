from app.report_render import by_time, clean_action, merge_items, same_text, seconds_of


def test_numbers_keep_items_apart():
    assert not same_text('What about item 5?', 'What about item 10?')


def test_rephrased_duplicates_merge():
    assert same_text('Send the revised contract to the client', 'Send the client the revised contract')
    assert same_text('Ask finance to approve the CloudNine quote', 'Ask finance to approve CloudNine quote')


def test_different_items_stay_apart():
    assert not same_text('Update the project plan', 'Write up the test report')


def test_different_owners_are_not_merged():
    a = clean_action({'task': 'Send the weekly update', 'owner': 'Priya'})
    b = clean_action({'task': 'Send the weekly update', 'owner': 'Rahul'})
    assert len(merge_items('task', [a], [b], owner_key='owner')) == 2


def test_same_person_different_spelling_merges():
    a = clean_action({'task': 'Send the weekly update', 'owner': 'Priya'})
    b = clean_action({'task': 'Send the weekly update', 'owner': 'Priya S'})
    assert len(merge_items('task', [a], [b], owner_key='owner')) == 1


def test_duplicate_fills_the_gaps_of_the_first_copy():
    a = clean_action({'task': 'Email the vendor', 'owner': 'unknown', 'timestamp': '01:00'})
    b = clean_action({'task': 'Email the vendor', 'owner': 'Meera', 'deadline': 'Friday', 'evidence': 'I will email the vendor'})
    (m,) = merge_items('task', [a], [b], owner_key='owner')
    assert (m['owner'], m['deadline'], m['status']) == ('Meera', 'Friday', 'Assigned')
    assert m['evidence'] == 'I will email the vendor' and m['timestamp'] == '01:00'


def test_owner_missing_gets_needs_owner_status():
    assert clean_action({'task': 'Update the roadmap'})['status'] == 'Needs owner'
    assert clean_action({'task': 'Update the roadmap', 'owner': 'N/A'})['owner'] == 'Unknown'


def test_by_time_sorts_clock_stamps_and_puts_undated_last():
    items = [{'timestamp': '10:05'}, {'timestamp': ''}, {'timestamp': '01:02'}, {'timestamp': '1:00:00'}]
    assert [i['timestamp'] for i in by_time(items)] == ['01:02', '10:05', '1:00:00', '']
    assert seconds_of('bad') > 10 ** 6
