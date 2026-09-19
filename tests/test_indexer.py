import importlib.util
import os as _os
import sys as _sys
import tempfile
import time
import types
import unittest

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src')

# Load plugin modules as a synthetic package so their relative imports resolve.
_pkg = types.ModuleType('sspkg')
_pkg.__path__ = [SRC]
_sys.modules['sspkg'] = _pkg


def _loadpkg(name):
    key = 'sspkg.' + name
    if key in _sys.modules:
        return _sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _os.path.join(SRC, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'sspkg'
    _sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


indexer = _loadpkg('indexer')
store_mod = _loadpkg('store')
attributes = _loadpkg('attributes')
utils = _loadpkg('utils')
chunker = _loadpkg('chunker')

_FIELDS = [utils.AttrField('gender', 'ss_gender', 'text', 'g'), utils.AttrField('tropes', 'ss_tropes', 'tags', 't')]


class FakeLLM:
    def __init__(self, data=None):
        self.data = data or {}
        self.calls = 0

    def generate_structured_output(self, prompt, schema, instructions='', use_model=''):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(
            data=SimpleNamespace(**{f.name: self.data.get(f.name) for f in _FIELDS}), exception=None, error_details=''
        )


class FailingLLM:
    def generate_structured_output(self, prompt, schema, instructions='', use_model=''):
        from types import SimpleNamespace

        return SimpleNamespace(data=None, exception=RuntimeError('boom'), error_details='boom')


class FakeLLMSequence:
    """Serves queued responses in order (one dict per call), cycling."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.calls = 0

    def generate_structured_output(self, prompt, schema, instructions='', use_model=''):
        from types import SimpleNamespace

        self.calls += 1
        raw = self.responses[self.i % len(self.responses)]
        self.i += 1
        return SimpleNamespace(
            data=SimpleNamespace(**{f.name: raw.get(f.name) for f in _FIELDS}), exception=None, error_details=''
        )


class FakeApi:
    """Minimal newAPI stand-in for reconcile tests."""

    def __init__(self, book_ids):
        self.book_ids = set(book_ids)

    def all_book_ids(self):
        return sorted(self.book_ids)

    def formats(self, bid):
        return ('EPUB',) if bid in self.book_ids else ()

    def format_metadata(self, bid, fmt):
        return {'size': 100, 'mtime': 1234.0}


class FakeWriter:
    """Stands in for the GUI db proxy; records custom-column writes."""

    def __init__(self):
        self.columns = {}
        self.fields = {}
        self._next_num = 1

    @property
    def backend(self):
        class B:

            pass

        b = B()
        b.custom_column_label_map = self.columns
        return b

    def create_custom_column(self, label, name, datatype, is_multiple):
        num = self._next_num
        self._next_num += 1
        self.columns[label] = {'label': label, 'name': name, 'datatype': datatype, 'is_multiple': is_multiple, 'num': num}

    def set_custom_column_metadata(self, num, name=None, label=None, is_editable=None, display=None):
        for col in self.columns.values():
            if col['num'] == num:
                if name is not None:
                    col['name'] = name
                break

    def set_field(self, key, mapping):
        self.fields.setdefault(key, {}).update(mapping)


class TestMtimeToInt(unittest.TestCase):
    def test_float_truncated_not_rounded(self):
        # a stored mtime must never be later than the file's real one
        self.assertEqual(indexer._mtime_to_int(1598092103.694143), 1598092103)
        self.assertEqual(indexer._mtime_to_int(1622500000.5), 1622500000)

    def test_int_and_none(self):
        self.assertEqual(indexer._mtime_to_int(1234), 1234)
        self.assertIsNone(indexer._mtime_to_int(None))

    def test_datetime_truncated(self):
        from datetime import datetime, timezone
        self.assertEqual(indexer._mtime_to_int(datetime.fromtimestamp(1598092103.694143, tz=timezone.utc)), 1598092103)


class TestAttrPhase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_chunks = attributes._chunks_for_book
        attributes._chunks_for_book = lambda s, bid: ['word ' * 40]

    def tearDown(self):
        attributes._chunks_for_book = self._orig_chunks
        self._tmp.cleanup()

    def _make(self, auto=True):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.auto_extract_attributes = auto
        statuses, done = [], []
        writer = FakeWriter()
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: None,
            settings_provider=lambda: settings,
            status_cb=statuses.append,
            attr_writer=writer,
            attr_done_cb=done.append,
        )
        return vs, ix, settings, statuses, done, writer

    def _index_book(self, vs, bid):
        vs.upsert_book(bid, 'EPUB', 1, 'm')
        c = chunker.Chunk(chunk_no=0, text='x' * 3000, chapter_path=['c'], para_start=0, para_end=1, char_offset=0)
        vs.insert_chunk(bid, c, 'm', [0.1] * 8)
        vs.commit(bid)

    def test_persists_and_reports(self):
        vs, ix, settings, statuses, done, writer = self._make()
        llm = FakeLLM({'gender': 'female', 'tropes': ['x']})
        ix._process_attributes([1, 2], settings, llm=llm)
        # source of truth (attrs_raw) is populated for both books
        self.assertEqual(vs.get_attrs(1)['gender'], 'female')
        self.assertEqual(vs.get_attrs(2)['tropes'], ['x'])
        # live status reported per book plus a terminal state
        states = [s['state'] for s in statuses]
        self.assertEqual(states.count('attributes'), 2)
        self.assertIn('attributes_done', states)
        # done callback got (total, errors)
        self.assertEqual(done[0][0], 2)
        self.assertEqual(done[0][1], [])
        # custom-column mirror happened through the writer proxy
        self.assertEqual(writer.fields['#ss_gender'][1], 'female')
        self.assertEqual(writer.fields['#ss_tropes'][2], ['x'])
        vs.close()

    def test_failures_recorded_and_reported(self):
        vs, ix, settings, statuses, done, writer = self._make()
        ix._process_attributes([1], settings, llm=FailingLLM())
        self.assertEqual(done[0][0], 1)
        self.assertEqual(len(done[0][1]), 1)
        self.assertEqual(vs.failed_book_ids('attr'), [1])
        vs.close()

    def test_fulltext_reports_sub_progress(self):
        # fulltext mode makes one LLM call per map group; each call must surface as
        # sub-progress on top of the book-level done/total (like embedding's chunk count)
        vs, ix, settings, statuses, done, writer = self._make()
        settings.attr_mode = 'fulltext'
        settings.attr_context_tokens = 3000  # ~6916 char budget -> [5000,5000,100] splits into 2 groups
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        ix._process_attributes([1], settings, llm=FakeLLM({'gender': 'f'}))
        sub = [s for s in statuses if s.get('sub_total')]
        self.assertEqual([(s['sub_done'], s['sub_total']) for s in sub], [(1, 2), (2, 2)])
        for s in sub:
            self.assertEqual(s['state'], 'attributes')
            self.assertEqual((s['done'], s['total']), (1, 1))
            self.assertEqual(s['book_id'], 1)
        vs.close()

    def test_fulltext_merge_stage_reported(self):
        # parts that disagree on a text field trigger the extra reduce call; it must be
        # reported as its own stage so the GUI doesn't sit on "part N/N"
        vs, ix, settings, statuses, done, writer = self._make()
        settings.attr_mode = 'fulltext'
        settings.attr_context_tokens = 3000  # ~6916 char budget -> [5000,5000,100] splits into 2 groups
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        ix._process_attributes([1], settings, llm=FakeLLMSequence([{'gender': 'male'}, {'gender': 'female'}]))
        merge = [s for s in statuses if s.get('stage') == 'merging']
        self.assertEqual(len(merge), 1)
        self.assertEqual(merge[0]['state'], 'attributes')
        self.assertEqual((merge[0]['done'], merge[0]['total']), (1, 1))
        self.assertEqual(merge[0]['book_id'], 1)
        vs.close()

    def test_pending_excludes_failed(self):
        vs, ix, settings, statuses, done, writer = self._make()
        self._index_book(vs, 1)
        self._index_book(vs, 2)
        vs.set_failed(1, 'attr', 'x')
        self.assertEqual(ix._pending_attr_books(settings), [2])
        vs.close()

    def test_success_clears_prior_failure(self):
        vs, ix, settings, statuses, done, writer = self._make()
        vs.set_failed(1, 'attr', 'old')
        ix._process_attributes([1], settings, llm=FakeLLM({'gender': 'male'}))
        self.assertEqual(vs.failed_book_ids('attr'), [])
        vs.close()

    def test_phase_interrupts_when_dirty_work_appears(self):
        # a re-index queued mid-phase (Book Details context menu) must preempt the
        # phase so the main loop can drain the dirty queue before more LLM calls
        vs, ix, settings, statuses, done, writer = self._make()
        llm = FakeLLM({'gender': 'f'})
        inner = llm.generate_structured_output
        state = {'added': False}

        def gen(*a, **kw):
            r = inner(*a, **kw)
            if not state['added']:
                state['added'] = True
                vs.add_dirty(9, 'reindex')
            return r

        llm.generate_structured_output = gen
        completed = ix._process_attributes([1, 2], settings, llm=llm)
        self.assertFalse(completed)
        # the first book finished before the dirty entry appeared; the second was skipped
        self.assertEqual(vs.get_attrs(1)['gender'], 'f')
        self.assertEqual(vs.get_attrs(2), {})
        self.assertNotIn('attributes_done', [s['state'] for s in statuses])
        self.assertEqual(done, [])
        vs.close()


def _index_book(vs, bid):
    vs.upsert_book(bid, 'EPUB', 1, 'm')
    c = chunker.Chunk(chunk_no=0, text='x' * 3000, chapter_path=['c'], para_start=0, para_end=1, char_offset=0)
    vs.insert_chunk(bid, c, 'm', [0.1] * 8)
    vs.commit(bid)


class TestForcedAttrExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_chunks = attributes._chunks_for_book
        attributes._chunks_for_book = lambda s, bid: ['word ' * 40]

    def tearDown(self):
        attributes._chunks_for_book = self._orig_chunks
        self._tmp.cleanup()

    def _make(self):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        statuses, done = [], []
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: None,
            settings_provider=lambda: settings,
            status_cb=statuses.append,
            attr_writer=FakeWriter(),
            attr_done_cb=done.append,
        )
        return vs, ix, settings, statuses, done

    def test_request_without_book_id_forces_nothing(self):
        vs, ix, *_ = self._make()
        ix.request_attributes()
        self.assertEqual(ix._forced_attrs, set())
        vs.close()

    def test_request_attributes_forces_stored_book(self):
        # a book whose attributes are already stored is not normally pending,
        # but a forced re-extraction must include it in the phase
        vs, ix, settings, *_ = self._make()
        _index_book(vs, 7)
        vs.set_attrs(7, {'gender': 'f', 'tropes': []})
        self.assertEqual(ix._pending_attr_books(settings), [])
        ix.request_attributes(7)
        self.assertEqual(ix._attr_phase_books(settings), [7])
        vs.close()

    def test_forced_book_not_duplicated_when_pending(self):
        vs, ix, settings, *_ = self._make()
        _index_book(vs, 1)  # no attrs stored -> already pending
        ix.request_attributes(1)
        self.assertEqual(ix._attr_phase_books(settings), [1])
        vs.close()

    def test_forced_pending_book_jumps_to_front(self):
        # a book that is normally pending (no attrs yet, e.g. first run) must also
        # jump to the front when explicitly requested — not stay in id order
        vs, ix, settings, *_ = self._make()
        for bid in (1, 2, 3):
            _index_book(vs, bid)
        ix.request_attributes(1)
        self.assertEqual(ix._attr_phase_books(settings), [1, 3, 2])
        vs.close()

    def test_phase_books_forced_front_pending_newest_first(self):
        vs, ix, settings, *_ = self._make()
        for bid in (1, 5, 3):
            _index_book(vs, bid)  # no attrs stored -> pending
        _index_book(vs, 9)
        vs.set_attrs(9, {'gender': 'f', 'tropes': []})  # complete -> not pending
        self.assertEqual(ix._attr_phase_books(settings), [5, 3, 1])
        ix.request_attributes(9)
        self.assertEqual(ix._attr_phase_books(settings), [9, 5, 3, 1])
        vs.close()

    def test_process_drops_forced_book_on_success(self):
        vs, ix, settings, statuses, done = self._make()
        _index_book(vs, 7)
        vs.set_attrs(7, {'gender': 'old', 'tropes': []})
        ix.request_attributes(7)
        ix._process_attributes(ix._attr_phase_books(settings), settings, llm=FakeLLM({'gender': 'male'}))
        self.assertEqual(vs.get_attrs(7)['gender'], 'male')
        self.assertEqual(ix._forced_attrs, set())
        self.assertEqual(done[0][0], 1)
        vs.close()

    def test_failed_forced_book_recorded_and_dropped(self):
        vs, ix, settings, statuses, done = self._make()
        _index_book(vs, 7)
        ix.request_attributes(7)
        ix._process_attributes(ix._attr_phase_books(settings), settings, llm=FailingLLM())
        self.assertEqual(vs.failed_book_ids('attr'), [7])
        self.assertEqual(ix._forced_attrs, set())
        vs.close()

    def test_phase_interrupts_on_new_forced_request(self):
        # a re-extraction requested mid-phase (Book Details context menu) must
        # restart the phase with the forced book first, not wait for it to finish
        vs, ix, settings, statuses, done = self._make()
        _index_book(vs, 1)  # pending
        _index_book(vs, 2)  # pending
        _index_book(vs, 3)
        vs.set_attrs(3, {'gender': 'old', 'tropes': []})  # stored -> not pending
        llm = FakeLLM({'gender': 'f'})
        inner = llm.generate_structured_output
        state = {'forced': False}

        def gen(*a, **kw):
            r = inner(*a, **kw)
            if not state['forced']:
                state['forced'] = True
                ix.request_attributes(3)
            return r

        llm.generate_structured_output = gen
        completed = ix._process_attributes(ix._attr_phase_books(settings), settings, llm=llm)
        self.assertFalse(completed)
        # book 2 (first in descending order) finished before the request arrived
        self.assertEqual(vs.get_attrs(2)['gender'], 'f')
        self.assertEqual(vs.get_attrs(1), {})
        self.assertEqual(done, [])
        # restarting puts the forced book first; book 2 is no longer pending
        self.assertEqual(ix._attr_phase_books(settings), [3, 1])
        completed = ix._process_attributes([3, 1], settings, llm=FakeLLM({'gender': 'f'}))
        self.assertTrue(completed)
        self.assertEqual(vs.get_attrs(3)['gender'], 'f')
        self.assertEqual(ix._forced_attrs, set())
        vs.close()

    def test_forced_failed_book_jumps_front_during_running_phase(self):
        # a book that FAILED attribute extraction is not normally pending; re-extracting
        # it mid-phase must restart the phase with it first (same as any forced request)
        vs, ix, settings, statuses, done = self._make()
        for bid in (10, 20, 30):
            _index_book(vs, bid)  # pending (no attrs)
        X = 5
        _index_book(vs, X)
        vs.set_failed(X, 'attr', 'old failure')
        self.assertEqual(ix._attr_phase_books(settings), [30, 20, 10])
        llm = FakeLLM({'gender': 'f'})
        inner = llm.generate_structured_output
        state = {'forced': False}

        def gen(*a, **kw):
            r = inner(*a, **kw)
            if not state['forced']:
                state['forced'] = True
                ix.request_attributes(X)  # user re-extracts the failed book mid-phase
            return r

        llm.generate_structured_output = gen
        completed = ix._process_attributes(ix._attr_phase_books(settings), settings, llm=llm)
        self.assertFalse(completed)
        # the first book finished before the request arrived; restarting puts X first
        self.assertEqual(vs.get_attrs(30)['gender'], 'f')
        self.assertEqual(ix._attr_phase_books(settings), [X, 20, 10])
        vs.close()

    def test_stale_phase_list_missing_forced_book_restarts(self):
        # a forced book that is in _forced_attrs but NOT in the phase list being processed
        # (the list was built before the request landed) must still trigger a restart —
        # otherwise the request is silently dropped from this pass and only runs later
        vs, ix, settings, statuses, done = self._make()
        for bid in (10, 20):
            _index_book(vs, bid)  # pending
        X = 5
        _index_book(vs, X)
        vs.set_failed(X, 'attr', 'old failure')
        ix.request_attributes(X)  # X is now forced
        # simulate a phase list computed before request_attributes(X): it has no X
        stale_phase = [20, 10]
        llm = FakeLLM({'gender': 'f'})
        completed = ix._process_attributes(stale_phase, settings, llm=llm)
        self.assertFalse(completed)  # must restart to include the forced book first
        self.assertEqual(ix._attr_phase_books(settings), [X, 20, 10])
        vs.close()


class TestReconcile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._vs = None

    def tearDown(self):
        if self._vs is not None:
            self._vs.close()
        self._tmp.cleanup()

    def _make(self, lib_ids):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        self._vs = vs
        settings = utils.Settings()
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: FakeApi(lib_ids),
            settings_provider=lambda: settings,
        )
        return vs, ix

    def test_removed_book_state_cleaned(self):
        vs, ix = self._make([1])
        _index_book(vs, 2)
        vs.set_attrs(2, {'gender': 'f'})
        vs.set_file_info(2, 'EPUB', 100, 1234)
        vs.set_failed(2, 'index', 'x')
        vs.set_failed(2, 'attr', 'y')
        ix.reconcile()
        self.assertFalse(vs.book_is_indexed(2))
        self.assertEqual(vs.get_attrs(2), {})
        self.assertIsNone(vs.get_file_info(2))
        self.assertEqual(vs.failed_book_ids(), [])

    def test_removed_dirty_book_dropped(self):
        vs, ix = self._make([1])
        vs.add_dirty(3)
        ix.reconcile()
        # book 3 vanished from the library; book 1 (present, unindexed) is queued
        self.assertNotIn(3, vs.dirty_book_ids())

    def test_preexisting_orphans_swept(self):
        # rows left behind by older versions: attrs_raw + fileinfo with no books/dirty row
        vs, ix = self._make([1])
        vs.set_attrs(2, {'gender': 'f'})
        vs.set_file_info(2, 'EPUB', 100, 1234)
        ix.reconcile()
        self.assertEqual(vs.get_attrs(2), {})
        self.assertIsNone(vs.get_file_info(2))
        self.assertEqual(vs.attr_book_ids(), [])

    def test_failed_indexing_book_removed(self):
        # a book that failed indexing has no books row, only fileinfo + a failed entry
        vs, ix = self._make([1])
        vs.set_file_info(3, 'EPUB', 100, 1234)
        vs.set_failed(3, 'index', 'x')
        ix.reconcile()
        self.assertIsNone(vs.get_file_info(3))
        self.assertEqual(vs.failed_book_ids(), [])

    def test_failed_unchanged_book_not_requeued(self):
        # a book that failed indexing (e.g. scanned file) stays failed when its
        # file is unchanged; it is only retried via 'Re-index new && failed'
        vs, ix = self._make([1, 3])
        vs.set_file_info(3, 'EPUB', 100, 1234)
        vs.set_failed(3, 'index', 'no meaningful text extracted (0 chars from EPUB)')
        ix.reconcile()
        self.assertNotIn(3, vs.dirty_book_ids())
        self.assertEqual(vs.failed_book_ids('index'), [3])
        vs.close()

    def test_surviving_books_untouched(self):
        vs, ix = self._make([1])
        _index_book(vs, 1)
        vs.set_attrs(1, {'gender': 'm'})
        vs.set_file_info(1, 'EPUB', 100, 1234)
        ix.reconcile()
        self.assertTrue(vs.book_is_indexed(1))
        self.assertEqual(vs.get_attrs(1)['gender'], 'm')
        self.assertEqual(vs.get_file_info(1), {'fmt': 'EPUB', 'size': 100, 'mtime_s': 1234})


class TestPauseResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_chunks = attributes._chunks_for_book
        attributes._chunks_for_book = lambda s, bid: ['word ' * 40]

    def tearDown(self):
        attributes._chunks_for_book = self._orig_chunks
        self._tmp.cleanup()

    def _make(self):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        statuses, done = [], []
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: None,
            settings_provider=lambda: settings,
            status_cb=statuses.append,
            attr_writer=FakeWriter(),
            attr_done_cb=done.append,
        )
        return vs, ix, settings, statuses, done

    def test_pause_resume_toggles_flag(self):
        vs, ix, *_ = self._make()
        self.assertFalse(ix.paused)
        ix.pause()
        self.assertTrue(ix.paused)
        ix.resume()
        self.assertFalse(ix.paused)
        vs.close()

    def test_paused_phase_interrupts_without_done(self):
        vs, ix, settings, statuses, done = self._make()
        ix.pause()
        completed = ix._process_attributes([1, 2], settings, llm=FakeLLM({'gender': 'f'}))
        # interrupted before any book; no terminal state and no done callback
        self.assertFalse(completed)
        self.assertEqual(done, [])
        self.assertNotIn('attributes_done', [s['state'] for s in statuses])
        vs.close()

    def test_resumed_phase_completes(self):
        vs, ix, settings, statuses, done = self._make()
        ix.pause()
        ix.resume()
        completed = ix._process_attributes([1], settings, llm=FakeLLM({'gender': 'f'}))
        self.assertTrue(completed)
        self.assertEqual(done[0][0], 1)
        self.assertIn('attributes_done', [s['state'] for s in statuses])
        vs.close()


class TestResumeReconcile(unittest.TestCase):
    """A book that failed indexing and was then removed from the library must be
    cleaned up when indexing resumes (not before a library switch or calibre restart);
    while paused itself nothing runs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_interval = indexer.Indexer.IDLE_RECONCILE_SECONDS
        indexer.Indexer.IDLE_RECONCILE_SECONDS = 0.2

    def tearDown(self):
        indexer.Indexer.IDLE_RECONCILE_SECONDS = self._orig_interval
        self._tmp.cleanup()

    def test_removed_failed_book_cleaned_on_resume(self):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        api = FakeApi([1])  # book 2 is gone from the library
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: api,
            settings_provider=lambda: settings,
            status_cb=lambda d: None,
            attr_writer=FakeWriter(),
        )
        vs.set_failed(2, 'index', 'embedding failed')
        ix.pause()
        ix.start()
        try:
            # while paused nothing runs: the stale entry must still be there
            time.sleep(0.5)
            self.assertEqual(vs.failed_book_ids(), [2])
            ix.resume()
            deadline = time.time() + 5
            while time.time() < deadline and 2 in vs.failed_book_ids():
                time.sleep(0.05)
        finally:
            ix.stop()
            ix.join(timeout=5)
        # book 2 (gone from the library) is cleaned up; book 1 may have its own
        # failure from the worker picking it up after resume, which is irrelevant here
        self.assertNotIn(2, vs.failed_book_ids())
        vs.close()

    def test_paused_loop_does_not_process_dirty_queue(self):
        # reconcile keeps running while paused, but no indexing work happens
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        api = FakeApi([1])
        ix = indexer.Indexer(
            store=vs,
            get_new_api=lambda: api,
            settings_provider=lambda: settings,
            status_cb=lambda d: None,
            attr_writer=FakeWriter(),
        )
        vs.add_dirty(1)
        ix.pause()
        ix.start()
        try:
            # long enough for several reconcile cycles at the shortened interval
            time.sleep(0.6)
        finally:
            ix.stop()
            ix.join(timeout=5)
        self.assertEqual(vs.dirty_book_ids(), [1])
        self.assertFalse(vs.book_is_indexed(1))
        vs.close()


class TestProcessOneUnexpectedError(unittest.TestCase):
    """An unexpected per-book crash (e.g. a parser bug in a dependency) must fail
    the book and remove it from the dirty queue, not leak into run() where it
    would be retried forever."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_unexpected_exception_fails_book_and_leaves_dirty(self):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        statuses = []
        api = FakeApi([58])
        api.format = lambda bid, fmt, as_path=False: _os.path.join(self._tmp.name, 'fake.epub')
        ix = indexer.Indexer(
            store=vs, get_new_api=lambda: api, settings_provider=lambda: settings, status_cb=statuses.append
        )
        vs.add_dirty(58)
        orig_extract = indexer.extract_book_pages
        orig_chunks = indexer.chunks_from_pages

        def boom(*a, **kw):
            raise RuntimeError('internal error, line 15, column 126')

        try:
            page = '<body><p>' + 'hello world ' * 100 + '</p></body>'
            indexer.extract_book_pages = lambda path, fmt: ('pages', [page])
            indexer.chunks_from_pages = boom
            ix._process_one(58)
        finally:
            indexer.extract_book_pages = orig_extract
            indexer.chunks_from_pages = orig_chunks
        entries = {e['book_id']: e['error'] for e in vs.failed_entries('index')}
        self.assertIn(58, entries)
        self.assertIn('internal error', entries[58])
        self.assertNotIn(58, vs.dirty_book_ids())
        self.assertIn('error', [s['state'] for s in statuses])
        vs.close()


class TestZeroChunkGuard(unittest.TestCase):
    """Extraction that yields no meaningful text, or chunking that loses all of it,
    must fail the book explicitly (visible in the status dialog, retryable via
    'Re-index new && failed') instead of recording a silent 0-chunk success."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, bid=58):
        vs = store_mod.VectorStore(_os.path.join(self._tmp.name, 't.db'), backend='sqlite')
        settings = utils.Settings()
        statuses = []
        api = FakeApi([bid])
        api.format = lambda b, fmt, as_path=False: _os.path.join(self._tmp.name, 'fake.epub')
        ix = indexer.Indexer(
            store=vs, get_new_api=lambda: api, settings_provider=lambda: settings, status_cb=statuses.append
        )
        vs.add_dirty(bid)
        return vs, ix, statuses

    def test_no_text_fails_book(self):
        vs, ix, statuses = self._make()
        orig = indexer.extract_book_pages
        try:
            indexer.extract_book_pages = lambda path, fmt: ('pages', ['', '   '])
            ix._process_one(58)
        finally:
            indexer.extract_book_pages = orig
        entries = {e['book_id']: e['error'] for e in vs.failed_entries('index')}
        self.assertIn(58, entries)
        self.assertIn('no meaningful text extracted (0 chars from EPUB)', entries[58])
        self.assertNotIn(58, vs.dirty_book_ids())
        self.assertFalse(vs.book_is_indexed(58))
        # file info recorded so reconcile does not auto-retry an unchanged file
        self.assertEqual(vs.get_file_info(58)['fmt'], 'EPUB')
        self.assertIn('error', [s['state'] for s in statuses])
        vs.close()

    def test_tiny_text_fails_book(self):
        vs, ix, statuses = self._make()
        orig = indexer.extract_book_pages
        try:
            page = '<body><p>' + 'word ' * 40 + '</p></body>'
            indexer.extract_book_pages = lambda path, fmt: ('pages', [page])
            ix._process_one(58)
        finally:
            indexer.extract_book_pages = orig
        entries = {e['book_id']: e['error'] for e in vs.failed_entries('index')}
        self.assertIn(58, entries)
        self.assertIn('no meaningful text extracted', entries[58])
        self.assertNotIn(58, vs.dirty_book_ids())
        self.assertFalse(vs.book_is_indexed(58))
        vs.close()

    def test_text_present_but_no_chunks_fails_book(self):
        # the book-41 failure mode: real text extracted, chunking lost it all
        vs, ix, statuses = self._make()
        orig_e, orig_c = indexer.extract_book_pages, indexer.chunks_from_pages
        try:
            page = '<body><p>' + 'word ' * 200 + '</p></body>'
            indexer.extract_book_pages = lambda path, fmt: ('pages', [page])
            indexer.chunks_from_pages = lambda *a, **kw: []
            ix._process_one(58)
        finally:
            indexer.extract_book_pages, indexer.chunks_from_pages = orig_e, orig_c
        entries = {e['book_id']: e['error'] for e in vs.failed_entries('index')}
        self.assertIn(58, entries)
        self.assertIn('chunking produced no chunks from', entries[58])
        self.assertNotIn(58, vs.dirty_book_ids())
        self.assertFalse(vs.book_is_indexed(58))
        vs.close()

    def test_text_above_threshold_still_indexes(self):
        vs, ix, statuses = self._make()
        embed_client_mod = _loadpkg('embed_client')

        class FakeEmbed:
            def __init__(self, **kw):
                pass

            def embed_batched(self, texts, batch_size=1, concurrency=1, progress=None):
                return [[0.1] * 8 for _ in texts]

        orig_e, orig_cl = indexer.extract_book_pages, embed_client_mod.EmbedClient
        try:
            page = '<body><p>' + 'word ' * 200 + '</p></body>'
            indexer.extract_book_pages = lambda path, fmt: ('pages', [page])
            embed_client_mod.EmbedClient = FakeEmbed
            ix._process_one(58)
        finally:
            indexer.extract_book_pages = orig_e
            embed_client_mod.EmbedClient = orig_cl
        self.assertEqual(vs.failed_book_ids(), [])
        self.assertTrue(vs.book_is_indexed(58))
        self.assertNotIn(58, vs.dirty_book_ids())
        self.assertIn('done', [s['state'] for s in statuses])
        vs.close()


if __name__ == '__main__':
    unittest.main()
