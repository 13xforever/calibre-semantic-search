import os as _os
import sys as _sys
import json
import tempfile
import types
import importlib.util
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

    @property
    def backend(self):
        class B:

            pass

        b = B()
        b.custom_column_label_map = self.columns
        return b

    def create_custom_column(self, label, name, datatype, is_multiple):
        self.columns[label] = {'label': label}

    def set_field(self, key, mapping):
        self.fields.setdefault(key, {}).update(mapping)


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
        failed = json.loads(vs.get_meta('attr_failed', '{}'))
        self.assertIn('1', failed)
        vs.close()

    def test_pending_excludes_failed(self):
        vs, ix, settings, statuses, done, writer = self._make()
        self._index_book(vs, 1)
        self._index_book(vs, 2)
        vs.set_meta('attr_failed', json.dumps({'1': {'error': 'x'}}))
        self.assertEqual(ix._pending_attr_books(settings), [2])
        vs.close()

    def test_success_clears_prior_failure(self):
        vs, ix, settings, statuses, done, writer = self._make()
        vs.set_meta('attr_failed', json.dumps({'1': {'error': 'old'}}))
        ix._process_attributes([1], settings, llm=FakeLLM({'gender': 'male'}))
        failed = json.loads(vs.get_meta('attr_failed', '{}'))
        self.assertNotIn('1', failed)
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
        failed = json.loads(vs.get_meta('attr_failed', '{}'))
        self.assertIn('7', failed)
        self.assertEqual(ix._forced_attrs, set())
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
        vs.set_meta('failed', json.dumps({'2': {'error': 'x'}}))
        vs.set_meta('attr_failed', json.dumps({'2': {'error': 'y'}}))
        ix.reconcile()
        self.assertFalse(vs.book_is_indexed(2))
        self.assertEqual(vs.get_attrs(2), {})
        self.assertIsNone(vs.get_file_info(2))
        self.assertNotIn('2', json.loads(vs.get_meta('failed', '{}')))
        self.assertNotIn('2', json.loads(vs.get_meta('attr_failed', '{}')))

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
        vs.set_meta('failed', json.dumps({'3': {'error': 'x'}}))
        ix.reconcile()
        self.assertIsNone(vs.get_file_info(3))
        self.assertNotIn('3', json.loads(vs.get_meta('failed', '{}')))

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


if __name__ == '__main__':
    unittest.main()
