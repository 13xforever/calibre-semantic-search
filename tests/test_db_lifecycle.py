"""End-to-end store scenarios on small, reproducible databases.

Three scenarios, each exercising the full operation surface:
1. a sqlite DB built with the legacy v0 schema, migrated to the latest schema version
2. a brand-new sqlite DB (created at the latest schema)
3. a lancedb backend (no migrations; created at the latest shape)

For each: insert embeddings for a few books, attributes, search, dirty queue,
file info, removal of stale data (removed book + abandoned model table), and
persistence across a reopen.
"""
import importlib.util
import math
import os
import os as _os
import sqlite3
import sys as _sys
import tempfile
import types
import unittest
from dataclasses import dataclass

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


store = _loadpkg('store')


@dataclass
class _C:
    chunk_no: int
    text: str
    chapter_path: list = None
    para_start: int = 0
    para_end: int = 0
    char_offset: int = 0


# Legacy v0 layout (pre-migration): old books/dirty/attrs_raw/meta shapes plus the
# single fat chunks table. Mirrors the structure of a real pre-migration library DB.
LEGACY_V0_SCHEMA = '''
CREATE TABLE IF NOT EXISTS books(
    id INTEGER PRIMARY KEY,
    fmt TEXT NOT NULL,
    indexed_at REAL,
    n_chunks INTEGER DEFAULT 0,
    model TEXT,
    dim INTEGER
);
CREATE TABLE IF NOT EXISTS dirty(
    book_id INTEGER PRIMARY KEY,
    fmt TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT 'added',
    added_at REAL
);
CREATE TABLE IF NOT EXISTS attrs_raw(
    book_id INTEGER PRIMARY KEY,
    json TEXT NOT NULL DEFAULT '{}',
    fields TEXT NOT NULL DEFAULT '',
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    chunk_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    chapter_path TEXT NOT NULL DEFAULT '',
    para_start INTEGER NOT NULL DEFAULT 0,
    para_end INTEGER NOT NULL DEFAULT 0,
    char_offset INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id, chunk_no);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
'''


class _LifecycleOps:
    """The operation suite shared by all three scenarios."""

    MODEL = 'lifecycle-model'
    DIM = 8
    BOOKS = ((901, 'EPUB', 3), (902, 'PDF', 2), (903, 'LRF', 4))

    def _vec(self, bid, i):
        if bid == 901 and i == 0:  # aligned with the query [1]*DIM -> deterministic top hit
            return store.l2_normalize([1.0] * self.DIM)
        return store.l2_normalize([math.cos(0.3 * (bid * 10 + i) + t) for t in range(self.DIM)])

    def _index(self, s, bid, fmt, n, model):
        chunks = [_C(i, f'book {bid} chunk {i}', ['Ch', f'S{i}'], i, i + 1, i * 10) for i in range(n)]
        for i, c in enumerate(chunks):
            s.insert_chunk(bid, c, model, self._vec(bid, i))
        s.commit(bid)
        s.upsert_book(bid, fmt, n, model)

    def _run_ops(self, s):
        q = [1.0] * self.DIM
        # -- insert embeddings for a few books ------------------------------------
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n, self.MODEL)
        reg = {b['id']: b for b in s.indexed_books()}
        for bid, fmt, n in self.BOOKS:
            self.assertTrue(s.book_is_indexed(bid))
            self.assertEqual(reg[bid]['fmt'], fmt)
            self.assertEqual(reg[bid]['n_chunks'], n)
            self.assertEqual(reg[bid]['model'], store.normalize_model(self.MODEL))
        # -- search (min_score=-1.0: cosine range, so every book's best chunk qualifies) --
        res = s.search(q, limit=9, min_score=-1.0)
        self.assertEqual(len(res), 3)  # one result per book
        self.assertEqual((res[0].book_id, res[0].chunk_no), (901, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(res[0].text, 'book 901 chunk 0')
        self.assertEqual(res[0].chapter_path, ['Ch', 'S0'])
        self.assertEqual(res[0].fmt, 'EPUB')
        for a, b in zip(res, res[1:]):
            self.assertGreaterEqual(a.score, b.score)
        # each result really is its book's best chunk (cross-check via search_book)
        for r in res:
            best = s.search_book(q, r.book_id, min_score=-1.0)[0]
            self.assertEqual(r.chunk_no, best.chunk_no)
            self.assertAlmostEqual(r.score, best.score, places=5)
        mid = (res[1].score + res[2].score) / 2.0
        top2 = s.search(q, limit=9, min_score=mid)
        self.assertEqual([r.book_id for r in top2], [r.book_id for r in res[:2]])
        self.assertEqual(s.search(q, limit=9, min_score=res[0].score + 0.001), [])
        self.assertEqual(s.book_chunks_text(903), [f'book 903 chunk {i}' for i in range(4)])
        # -- attributes ---------------------------------------------------------------
        s.set_attrs(901, {'title': 'Ops Book One', 'tags': ['a', 'b']})
        s.set_attrs(903, {'title': 'Ops Book Three'})
        self.assertEqual(s.get_attrs(901), {'title': 'Ops Book One', 'tags': ['a', 'b']})
        self.assertTrue({901, 903} <= set(s.attr_book_ids()))
        s.clear_attrs(903)
        self.assertEqual(s.get_attrs(903), {})
        # -- dirty queue ------------------------------------------------------------------
        s.add_dirty(904, 'added')
        self.assertIn(904, s.dirty_book_ids())
        s.remove_dirty(904)
        self.assertNotIn(904, s.dirty_book_ids())
        # -- failed books ---------------------------------------------------------------
        s.wipe_failed('index')  # start from a known state (scenario fixtures may seed rows)
        s.wipe_failed('attr')
        self.assertEqual(s.failed_book_ids(), [])
        s.set_failed(901, 'index', 'boom')
        s.set_failed(902, 'attr', 'llm down')
        self.assertEqual(s.failed_book_ids(), [901, 902])
        self.assertEqual(s.failed_book_ids('index'), [901])
        self.assertEqual(s.failed_entries('attr'), [{'book_id': 902, 'kind': 'attr', 'error': 'llm down'}])
        s.set_failed(901, 'index', 'boom again')  # upsert: same row, new error
        self.assertEqual([e['error'] for e in s.failed_entries('index')], ['boom again'])
        s.clear_failed(902)  # drops every kind for the book
        self.assertEqual(s.failed_book_ids(), [901])
        # -- persisted indexing status --------------------------------------------------------
        s.set_meta('indexing_status', 'running')
        self.assertEqual(s.get_meta('indexing_status'), 'running')
        # -- file info -----------------------------------------------------------------------
        s.set_file_info(901, 'EPUB', 123456, 1788528000)
        fi = s.get_file_info(901)
        self.assertEqual((fi['fmt'], fi['size'], fi['mtime_s']), ('EPUB', 123456, 1788528000))
        self.assertIn(901, s.file_info_book_ids())
        s.clear_file_info(901)
        self.assertNotIn(901, s.file_info_book_ids())
        # -- remove stale data reflecting removed books -----------------------------------------
        s.clear_book(902)
        self.assertFalse(s.book_is_indexed(902))
        self.assertEqual(s.book_chunks_text(902), [])
        self.assertNotIn(902, {r.book_id for r in s.search(q, limit=50, min_score=-1.0)})
        # a model abandoned after re-indexing: its table must be droppable as stale
        self._index(s, 905, 'EPUB', 2, 'stale-model')
        s.clear_book(905)
        self._index(s, 905, 'EPUB', 2, self.MODEL)
        self.assertEqual(s.cleanup_stale_models(self.MODEL), 1)
        self.assertEqual(s.search(q, limit=10, model='stale-model'), [])
        # -- manual chunk deletion --------------------------------------------------------
        s.delete_chunk(905, 0)
        self.assertEqual(s.book_chunks_text(905), ['book 905 chunk 1'])
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(reg[905]['n_chunks'], 1)
        # the book is still found through its surviving chunk
        self.assertIn(905, {r.book_id for r in s.search(q, limit=10, min_score=-1.0)})

    def _assert_persisted(self, s):
        q = [1.0] * self.DIM
        reg = {b['id'] for b in s.indexed_books()}
        self.assertTrue({901, 903, 905} <= reg)
        self.assertNotIn(902, reg)
        res = s.search(q, limit=9)
        self.assertEqual((res[0].book_id, res[0].chunk_no), (901, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(s.book_chunks_text(903), [f'book 903 chunk {i}' for i in range(4)])
        # the manually deleted chunk stays gone across a close/reopen, count included
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(reg[905]['n_chunks'], 1)
        self.assertEqual(s.book_chunks_text(905), ['book 905 chunk 1'])
        self.assertEqual(s.get_attrs(901)['title'], 'Ops Book One')
        self.assertEqual(s.get_attrs(903), {})
        self.assertNotIn(904, s.dirty_book_ids())
        self.assertNotIn(901, s.file_info_book_ids())
        self.assertEqual(s.failed_entries('index'), [{'book_id': 901, 'kind': 'index', 'error': 'boom again'}])
        self.assertEqual(s.failed_book_ids('attr'), [])
        self.assertEqual(s.get_meta('indexing_status'), 'running')


class TestSqliteLegacyV0(_LifecycleOps, unittest.TestCase):
    """Scenario 1: a small sqlite DB in the legacy v0 schema, migrated to the latest version."""

    LEGACY_MODEL = 'Legacy-Embedding-8B-GGUF'  # normalizes to 'legacy_embedding_8b'
    LEGACY_DIM = 4

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        self.tmp.cleanup()

    def _build_legacy_db(self):
        conn = sqlite3.connect(self.path)
        conn.executescript(LEGACY_V0_SCHEMA)
        m, d = self.LEGACY_MODEL, self.LEGACY_DIM
        for bid, fmt, n in ((1, 'EPUB', 2), (2, 'PDF', 1), (3, 'LRF', 0)):
            conn.execute('INSERT INTO books(id, fmt, indexed_at, n_chunks, model, dim) VALUES(?,?,?,?,?,?)',
                         (bid, fmt, 1788528000.0 + bid, n, m, d if n else 0))
        for bid, fmt in ((50, 'AZW3'), (51, 'EPUB')):  # dirty books absent from books, like real dbs
            conn.execute('INSERT INTO dirty(book_id, fmt, reason, added_at) VALUES(?,?,?,?)',
                         (bid, fmt, 'added', 1788528100.5))
        conn.execute("INSERT INTO attrs_raw(book_id, json, fields) VALUES(1,'{\"title\":\"Legacy Book One\"}','title')")
        for bid, val in ((1, 'EPUB|717102|1598092103.694143'), (2, 'PDF|1737973|1209541068.0')):
            conn.execute('INSERT INTO meta(key, value) VALUES(?,?)', (f'fileinfo:{bid}', val))
        # legacy failure records as meta JSON blobs (the pre-v4 layout)
        conn.execute("INSERT INTO meta(key, value) VALUES('failed','{\"3\": {\"error\": \"no meaningful text extracted\", \"at\": 1788528101.0}}')")
        conn.execute("INSERT INTO meta(key, value) VALUES('attr_failed','{\"1\": {\"error\": \"llm timeout\", \"at\": 1788528102.0}}')")
        self._legacy_texts = {}
        for bid, n in ((1, 2), (2, 1)):
            for cn in range(n):
                if bid == 1 and cn == 0:
                    vec = store.l2_normalize([1.0] * self.LEGACY_DIM)
                else:
                    vec = store.l2_normalize([math.cos(0.5 * (bid * 10 + cn) + t) for t in range(self.LEGACY_DIM)])
                text = f'legacy book {bid} chunk {cn}'
                self._legacy_texts[(bid, cn)] = text
                conn.execute(
                    'INSERT INTO chunks(book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector) '
                    'VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (bid, cn, text, f'Chapter {cn + 1}', cn * 3, cn * 3 + 2, cn * 400, m, self.LEGACY_DIM, store.vec_to_blob(vec, f32=True)))
        conn.commit()
        conn.close()

    def test_v0_migrates_to_latest_and_all_ops_work(self):
        self._build_legacy_db()
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        c = s.meta.conn
        with s.meta._lock:
            uv = c.execute('PRAGMA user_version').fetchone()[0]
            models = {r[0] for r in c.execute('SELECT model FROM models')}
            fmts = {r[0] for r in c.execute('SELECT fmt FROM formats')}
        # stage 1: open migrated meta to v1; chunk tables still legacy
        self.assertEqual(uv, 1)
        self.assertEqual(models, {'legacy_embedding_8b'})
        self.assertEqual(fmts, {'EPUB', 'PDF', 'LRF', 'AZW3'})
        # legacy mtime is truncated to whole seconds, never rounded into the future
        self.assertEqual(s.get_file_info(1), {'fmt': 'EPUB', 'size': 717102, 'mtime_s': 1598092103})
        self.assertEqual(s.get_file_info(2)['mtime_s'], 1209541068)
        self.assertEqual(s.meta_keys('fileinfo:'), [])
        self.assertEqual(sorted(s.dirty_book_ids()), [50, 51])
        self.assertEqual(s.get_attrs(1), {'title': 'Legacy Book One'})
        # migrated DB has no stored pause state -> starts paused by default
        self.assertEqual(s.get_meta('indexing_status', 'paused'), 'paused')
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(set(reg), {1, 2, 3})
        self.assertEqual((reg[1]['fmt'], reg[1]['n_chunks'], reg[1]['model']), ('EPUB', 2, 'legacy_embedding_8b'))
        with s.meta._lock:
            n = c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        self.assertEqual(n, 3)  # fat table untouched at v1
        self.assertEqual(s.backend._chunk_tables(), [])
        self.assertTrue(s.needs_finalize())
        # the legacy failure blobs stay in meta until finalize runs v4
        self.assertIsNotNone(s.get_meta('failed'))
        self.assertIsNotNone(s.get_meta('attr_failed'))

        # stage 2: finalize splits + slims the chunk tables (v2) and converts the
        # vectors to half floats (v3) in the same pass
        s.finalize_schema()
        with s.meta._lock:
            uv = c.execute('PRAGMA user_version').fetchone()[0]
            av = c.execute('PRAGMA auto_vacuum').fetchone()[0]
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            slim_cols = {r[1] for r in c.execute('PRAGMA table_info(chunks_legacy_embedding_8b)')}
            blob_len = c.execute('SELECT LENGTH(vector) FROM chunks_legacy_embedding_8b LIMIT 1').fetchone()[0]
        self.assertEqual(uv, store.SCHEMA_VERSION)
        self.assertEqual(av, 2)
        self.assertNotIn('chunks', tables)
        self.assertIn('chunks_legacy_embedding_8b', tables)
        self.assertEqual(slim_cols, {'id', 'book_id', 'chunk_no', 'text_z', 'chapter_path', 'vector'})
        self.assertEqual(blob_len, 2 * self.LEGACY_DIM)  # f32 -> f16: two bytes per component
        # v4 moved the legacy failure blobs into the failed table and deleted the keys
        self.assertEqual(s.failed_book_ids('index'), [3])
        self.assertEqual(s.failed_entries('attr'), [{'book_id': 1, 'kind': 'attr', 'error': 'llm timeout'}])
        self.assertIsNone(s.get_meta('failed'))
        self.assertIsNone(s.get_meta('attr_failed'))
        self.assertFalse(s.needs_finalize())
        res = s.search([1.0] * self.LEGACY_DIM, limit=5)
        self.assertEqual((res[0].book_id, res[0].chunk_no), (1, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(s.book_chunks_text(1), [self._legacy_texts[(1, i)] for i in range(2)])

        # full operation suite on the migrated db (coexists with the legacy model's table)
        self._run_ops(s)

        # stage 3: reopen stays at the latest schema, everything persisted
        s.close()
        self.s = store.VectorStore(self.path, backend='sqlite')
        with self.s.meta._lock:
            uv = self.s.meta.conn.execute('PRAGMA user_version').fetchone()[0]
        self.assertEqual(uv, store.SCHEMA_VERSION)
        self.assertFalse(self.s.needs_finalize())
        self._assert_persisted(self.s)


class TestSqliteFresh(_LifecycleOps, unittest.TestCase):
    """Scenario 2: a brand-new sqlite DB is created at the latest schema version."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        self.tmp.cleanup()

    def test_fresh_db_is_latest_and_all_ops_work(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        with s.meta._lock:
            uv = s.meta.conn.execute('PRAGMA user_version').fetchone()[0]
            av = s.meta.conn.execute('PRAGMA auto_vacuum').fetchone()[0]
        self.assertEqual(uv, store.SCHEMA_VERSION)
        self.assertEqual(av, 2)
        self.assertFalse(s.needs_finalize())
        # fresh DB has no stored pause state -> starts paused by default
        self.assertEqual(s.get_meta('indexing_status', 'paused'), 'paused')

        self._run_ops(s)

        s.close()
        self.s = store.VectorStore(self.path, backend='sqlite')
        self.assertFalse(self.s.needs_finalize())
        self._assert_persisted(self.s)


class TestLanceDb(_LifecycleOps, unittest.TestCase):
    """Scenario 3: the lancedb backend has no migrations; it is created at the latest shape.

    Not skipped when lancedb is missing: the suite must exercise this code path even
    though most users run the sqlite backend. Without the package, VectorStore raises
    a clear install error and the test fails loudly instead of passing silently."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        self.tmp.cleanup()

    def test_lancedb_all_ops_work(self):
        s = store.VectorStore(self.path, backend='lancedb')
        self.s = s
        self.assertEqual(s.backend_name, 'lancedb')
        self.assertFalse(s.needs_finalize())
        # lancedb keeps its meta in the same sqlite file -> same default
        self.assertEqual(s.get_meta('indexing_status', 'paused'), 'paused')

        self._run_ops(s)

        # data now exists -> the ANN index is a pending finalize stage
        self.assertEqual(s.pending_stages(), ['index'])
        stages = []
        s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
        self.assertTrue(any(st == 'index' for st, _ in stages))
        cfg = s.backend._our_index(s.backend._open_table(self.MODEL))
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.index_type, store.LanceVectorBackend.INDEX_TYPE)
        self.assertEqual(s.pending_stages(), [])
        self.assertFalse(s.needs_finalize())

        s.close()
        self.s = store.VectorStore(self.path, backend='lancedb')
        self._assert_persisted(self.s)

    def test_lancedb_index_merge_tail(self):
        s = store.VectorStore(self.path, backend='lancedb')
        self.s = s
        for i in range(4):
            s.insert_chunk(1, _C(i, f'chunk {i}', ['Ch', f'S{i}'], i, i + 1, i * 10), self.MODEL, store.l2_normalize([1.0] * self.DIM))
        s.commit(1)
        s.upsert_book(1, 'EPUB', 4, self.MODEL)
        s.finalize_schema()
        self.assertEqual(s.pending_stages(), [])
        orig = store.LanceVectorBackend.MERGE_TAIL_ROWS
        store.LanceVectorBackend.MERGE_TAIL_ROWS = 3
        try:
            for i in range(4, 7):
                s.insert_chunk(1, _C(i, f'chunk {i}', ['Ch', f'S{i}'], i, i + 1, i * 10), self.MODEL, store.l2_normalize([1.0] * self.DIM))
            s.commit(1)
            # three unindexed rows reach the (shrunk) threshold -> a merge is pending
            self.assertEqual(s.pending_stages(), ['index'])
            stages = []
            s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
            self.assertTrue(any('updating' in d for _, d in stages))
            self.assertEqual(s.pending_stages(), [])
        finally:
            store.LanceVectorBackend.MERGE_TAIL_ROWS = orig
        res = s.search([1.0] * self.DIM, limit=10, min_score=-1.0)
        self.assertEqual([r.book_id for r in res], [1])  # one row per book even with 7 tied chunks

    def test_lancedb_dir_is_hidden(self):
        s = store.VectorStore(self.path, backend='lancedb')
        self.s = s
        d = s._lancedb_dir()
        self.assertEqual(os.path.basename(d), '.semantic-search.lancedb')
        if os.name == 'nt':
            import ctypes

            attr = ctypes.windll.kernel32.GetFileAttributesW(d)
            self.assertNotEqual(attr, -1)
            self.assertTrue(attr & 0x2)


class _MigrateOps:
    """Shared helpers for the backend/codec migration scenarios."""

    MODEL = 'migrate-model'
    DIM = 8
    BOOKS = ((601, 'EPUB', 3), (602, 'PDF', 2))

    def _vec(self, bid, i):
        if bid == 601 and i == 0:  # aligned with the query [1]*DIM -> deterministic top hit
            return store.l2_normalize([1.0] * self.DIM)
        return store.l2_normalize([math.cos(0.3 * (bid * 10 + i) + t) for t in range(self.DIM)])

    def _index(self, s, bid, fmt, n):
        chunks = [_C(i, f'book {bid} chunk {i}', ['Ch', f'S{i}']) for i in range(n)]
        for i, c in enumerate(chunks):
            s.insert_chunk(bid, c, self.MODEL, self._vec(bid, i))
        s.commit(bid)
        s.upsert_book(bid, fmt, n, self.MODEL)

    def _check(self, s):
        q = [1.0] * self.DIM
        res = s.search(q, limit=5, min_score=-1.0)
        self.assertEqual(len(res), 2)  # one result per book
        self.assertEqual((res[0].book_id, res[0].chunk_no), (601, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(res[0].text, 'book 601 chunk 0')
        self.assertEqual(s.book_chunks_text(602), [f'book 602 chunk {i}' for i in range(2)])

    def _codec_name(self, s):
        return store.json.loads(s.get_meta(store.TEXT_CODEC_KEY))['name']


class TestCodecRecompress(_MigrateOps, unittest.TestCase):
    """The sqlite text codec follows the stored codec (meta['text_codec']).

    A compression switch confirmed in settings is recorded as a meta['recompress']
    marker that finalize_schema() executes as an in-place conversion. The test env
    has zstandard installed; unavailability is simulated by patching
    store._module_available (the real module stays importable, which mirrors an
    in-session uninstall where the already-loaded module still works)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None
        self._orig_avail = store._module_available

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        store._module_available = self._orig_avail
        self.tmp.cleanup()

    def _avail(self, zstandard_ok):
        real = self._orig_avail
        store._module_available = lambda name: False if (name == 'zstandard' and not zstandard_ok) else real(name)

    def test_switch_zstd_to_zlib_and_back(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(self._codec_name(s), 'zstd')  # zstandard is installed in the test env
        self.assertEqual(s.stored_codec(), 'zstd')
        self._check(s)

        # confirm a switch to zlib in settings: a marker appears and the conversion runs
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zlib'))
        self.assertEqual(s.pending_stages(), ['codec'])
        stages = []
        s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
        self.assertEqual(self._codec_name(s), 'zlib')
        self.assertIsNone(s.get_meta(store.RECOMPRESS_KEY))
        self.assertTrue(any(st == 'codec' for st, _ in stages))
        self.assertFalse(s.needs_finalize())
        self._check(s)

        # reopen: the stored codec sticks — no conversion back to zstd
        s.close()
        self.s = store.VectorStore(self.path, backend='sqlite')
        self.assertEqual(self.s.stored_codec(), 'zlib')
        self.assertEqual(self.s.pending_stages(), [])
        self.assertFalse(self.s.needs_finalize())
        self._check(self.s)

        # confirm a switch back to zstd: converts back in place
        self.s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zstd'))
        self.assertEqual(self.s.pending_stages(), ['codec'])
        self.s.finalize_schema()
        self.assertEqual(self._codec_name(self.s), 'zstd')
        self._check(self.s)

    def test_dict_drift_recompresses_on_open(self):
        # a plugin update ships a new default dictionary: on open, the stored zstd
        # record holding the old one schedules an in-place re-compression
        from unittest import mock

        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(self._codec_name(s), 'zstd')  # zstandard is installed in the test env
        self.assertIsNone(s.get_meta(store.RECOMPRESS_KEY))  # no drift while the dict matches
        s.close()
        self.s = None
        new_dict = b'updated default compression dictionary' * 16
        with mock.patch.object(store, '_load_default_dict', return_value=new_dict):
            s = store.VectorStore(self.path, backend='sqlite')
            self.s = s
            self.assertEqual(store.json.loads(s.get_meta(store.RECOMPRESS_KEY))['target'], 'zstd')
            self.assertEqual(s.pending_stages(), ['codec'])
            self.assertTrue(s.needs_finalize())
            stages = []
            s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
            self.assertIsNone(s.get_meta(store.RECOMPRESS_KEY))
            spec = store.json.loads(s.get_meta(store.TEXT_CODEC_KEY))
            self.assertEqual(spec['name'], 'zstd')
            self.assertEqual(spec['dictionary'], store.base64.b64encode(new_dict).decode('ascii'))
            self.assertTrue(any(st == 'codec' for st, _ in stages))
            self.assertFalse(s.needs_finalize())
            self._check(s)  # the text round-trips through the new dictionary

    def test_dict_drift_dataless_updates_spec_directly(self):
        # a dataless library gets no marker — its record is simply updated to the new
        # default dictionary, and data arriving afterwards uses it
        from unittest import mock

        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        old_dict = b'outdated default compression dictionary' * 16
        s.set_meta(
            store.TEXT_CODEC_KEY,
            store.json.dumps({'name': 'zstd', 'dictionary': store.base64.b64encode(old_dict).decode('ascii')}),
        )
        s.close()
        self.s = None
        new_dict = b'updated default compression dictionary' * 16
        with mock.patch.object(store, '_load_default_dict', return_value=new_dict):
            s = store.VectorStore(self.path, backend='sqlite')
            self.s = s
            self.assertIsNone(s.get_meta(store.RECOMPRESS_KEY))
            spec = store.json.loads(s.get_meta(store.TEXT_CODEC_KEY))
            self.assertEqual(spec['dictionary'], store.base64.b64encode(new_dict).decode('ascii'))
            self.assertEqual(s.pending_stages(), [])
            self.assertFalse(s.needs_finalize())
            for bid, fmt, n in self.BOOKS:
                self._index(s, bid, fmt, n)
            self._check(s)

    def test_dict_drift_keeps_inflight_marker(self):
        # a conversion already in flight (any target) is not reset by drift detection
        from unittest import mock

        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zlib'))  # a user-confirmed switch in flight
        s.close()
        self.s = None
        new_dict = b'updated default compression dictionary' * 16
        with mock.patch.object(store, '_load_default_dict', return_value=new_dict):
            s = store.VectorStore(self.path, backend='sqlite')
            self.s = s
            self.assertEqual(store.json.loads(s.get_meta(store.RECOMPRESS_KEY))['target'], 'zlib')

    def test_reopen_zstd_db_without_zstandard_blocks(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(self._codec_name(s), 'zstd')
        s.close()
        self.s = None
        self._avail(False)
        try:
            with self.assertRaises(store.MissingDependencyError) as cm:
                store.VectorStore(self.path, backend='sqlite')
            self.assertEqual(cm.exception.dep, 'zstandard')
            self.assertTrue(cm.exception.has_data)
        finally:
            self._avail(True)

    def test_inflight_zstd_marker_without_zstandard_blocks(self):
        # stored as zlib with a pending conversion to zstd: the open store is fine...
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zlib'))
        s.finalize_schema()
        self.assertEqual(self._codec_name(s), 'zlib')
        # ...and so is a fresh open while the marker still targets zstd
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zstd'))
        s.close()
        self.s = None
        self._avail(False)
        try:
            with self.assertRaises(store.MissingDependencyError) as cm:
                store.VectorStore(self.path, backend='sqlite')
            self.assertEqual(cm.exception.dep, 'zstandard')
            self.assertTrue(cm.exception.has_data)
        finally:
            self._avail(True)

    def test_fresh_library_never_blocks_on_codec(self):
        # no data yet: the codec is created on demand with whatever is available
        self._avail(False)
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(self._codec_name(s), 'zlib')
        self._check(s)

    def test_dataless_codec_choice_persists(self):
        # a dataless library's pick becomes the codec its data will be stored with
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        self.assertEqual(self._codec_name(s), 'zstd')  # zstandard is installed in the test env
        s.close()
        self.s = None
        ms = store.MetaStore(self.path)
        try:
            store.write_codec_choice(ms, 'zlib')  # what the settings dialog does for a dataless library
        finally:
            ms.close()
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        self.assertEqual(self._codec_name(s), 'zlib')
        self.assertIsNone(s.stored_codec())  # no chunk data yet
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(self._codec_name(s), 'zlib')  # no forced conversion to zstd
        self.assertEqual(s.pending_stages(), [])
        self._check(s)

    def test_dataless_zstd_record_survives_uninstall(self):
        # a dataless library whose record says zstd is healed to zlib when the package
        # is gone — it never blocks, because there is no stored data to protect
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        self.assertEqual(self._codec_name(s), 'zstd')
        s.close()
        self.s = None
        self._avail(False)
        try:
            s = store.VectorStore(self.path, backend='sqlite')
            self.s = s
            self.assertIsNone(s.stored_codec())
            self.assertEqual(self._codec_name(s), 'zlib')  # healed on open
        finally:
            self._avail(True)


class TestBackendMigration(_MigrateOps, unittest.TestCase):
    """Cross-backend transfer (sqlite <-> lancedb), resumable and orphan-safe."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        self.tmp.cleanup()

    def test_sqlite_to_lancedb_and_back(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        self.assertEqual(s.get_meta(store.BACKEND_KEY), 'sqlite')
        # the settings dialog flips the per-library meta; the migration runs on open
        s.set_meta(store.BACKEND_KEY, 'lancedb')
        s.close()
        size_before = os.path.getsize(self.path)  # close checkpointed the WAL into the main file

        s = store.VectorStore(self.path)
        self.s = s
        self.assertEqual(s.backend_name, 'sqlite')
        self.assertEqual(s.want_backend, 'lancedb')
        self.assertEqual(s.pending_stages(), ['backend'])
        self.assertTrue(s.needs_finalize())
        stages = []
        s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
        self.assertEqual(stages[0], ('backend', 'moving data to the lancedb backend'))
        # the freshly transferred data is indexed in the same pass (live-evaluated stage)
        self.assertTrue(any(st == 'index' for st, _ in stages))
        self.assertEqual(s.backend_name, 'lancedb')
        self.assertEqual(s.get_meta(store.BACKEND_KEY), 'lancedb')
        self.assertIsNone(s.get_meta(store.MIGRATE_KEY))
        self.assertFalse(s._sqlite_has_chunks())  # source storage dropped
        self.assertTrue(s._lancedb_has_data())
        self.assertFalse(s.needs_finalize())
        self._check(s)
        # dropping the chunk tables must actually reclaim their pages: finalize ends with
        # a full VACUUM, so no freelist bloat and the physical file has shrunk
        with s.meta._lock:
            free = s.meta.conn.execute('PRAGMA freelist_count').fetchone()[0]
        self.assertEqual(free, 0)
        self.assertLess(os.path.getsize(self.path), size_before)

        # reopen: stable, no stages
        s.close()
        self.s = store.VectorStore(self.path)
        self.assertEqual(self.s.backend_name, 'lancedb')
        self.assertEqual(self.s.pending_stages(), [])
        self._check(self.s)

        # switch back to sqlite
        self.s.set_meta(store.BACKEND_KEY, 'sqlite')
        self.s.close()
        self.s = store.VectorStore(self.path)
        self.assertEqual(self.s.backend_name, 'lancedb')
        self.assertEqual(self.s.want_backend, 'sqlite')
        self.assertEqual(self.s.pending_stages(), ['backend'])
        self.s.finalize_schema()
        self.assertEqual(self.s.backend_name, 'sqlite')
        self.assertFalse(self.s._lancedb_has_data())  # lancedb dir removed
        self.assertTrue(self.s._sqlite_has_chunks())
        self._check(self.s)

    def test_interrupted_transfer_resumes(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        model = store.normalize_model(self.MODEL)
        table = store.model_table_name(model)

        # simulate a crash mid-transfer: book 601 already moved to lancedb, progress
        # recorded, but the meta flip and source drop never happened
        lance = store.LanceVectorBackend(s.meta)
        items = s._backend_book_items('sqlite', s.backend, model, table, 601)
        lance.insert_chunks(601, [(c, v, self.MODEL) for c, v in items])
        s.set_meta(store.BACKEND_KEY, 'lancedb')
        s.set_meta(store.MIGRATE_KEY, store.json.dumps({'src': 'sqlite', 'dst': 'lancedb', 'models_done': [], 'books_done': [601]}))
        s.close()

        s = store.VectorStore(self.path)
        self.s = s
        # the partial lancedb data decides where the store opens, and the marker keeps
        # the transfer pending (and protects the sqlite source from the orphan sweep)
        self.assertEqual(s.backend_name, 'lancedb')
        self.assertEqual(s.want_backend, 'lancedb')
        self.assertTrue(s._sqlite_has_chunks())  # not swept while in flight
        # the partial lancedb data also lacks its vector index -> both stages pending
        self.assertEqual(s.pending_stages(), ['backend', 'index'])
        s.finalize_schema()
        self.assertEqual(s.backend_name, 'lancedb')
        self.assertIsNone(s.get_meta(store.MIGRATE_KEY))
        self.assertFalse(s._sqlite_has_chunks())
        self._check(s)  # both books survived the "crash"

    def test_orphan_storage_swept(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        # move everything to lancedb...
        s.set_meta(store.BACKEND_KEY, 'lancedb')
        s.close()
        self.s = store.VectorStore(self.path)
        self.s.finalize_schema()
        # ...then leave stale sqlite chunk data behind (an abandoned source remnant)
        with self.s.meta._lock:
            self.s.meta.conn.executescript(store.chunks_table_sql(store.model_table_name(store.normalize_model(self.MODEL))))
            self.s.meta.conn.commit()
        self.assertTrue(self.s._sqlite_has_chunks())
        self.s.close()
        # reopen sweeps it: no transfer in flight, so the extra storage is garbage
        self.s = store.VectorStore(self.path)
        self.assertEqual(self.s.backend_name, 'lancedb')
        self.assertFalse(self.s._sqlite_has_chunks())
        self._check(self.s)


class TestFailedTableUpgrade(_MigrateOps, unittest.TestCase):
    """A settled v3 library still carries the legacy failure JSON in meta; only the
    deferred v4 step runs on finalize (no chunk work is pending)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        self.tmp.cleanup()

    def test_settled_v3_db_migrates_failed_blobs(self):
        s = store.VectorStore(self.path, backend='sqlite')
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        # simulate a library written by the pre-v4 code: restore the legacy failure
        # JSON blobs and roll user_version back to 3 (the chunk tables are already
        # settled, so v2/v3 have nothing left to do)
        with s.meta._lock:
            c = s.meta.conn
            c.execute("INSERT INTO meta(key, value) VALUES('failed','{\"601\": {\"error\": \"no meaningful text extracted\", \"at\": 1.0}}')")
            c.execute("INSERT INTO meta(key, value) VALUES('attr_failed','{\"602\": {\"error\": \"llm timeout\", \"at\": 2.0}}')")
            c.execute('PRAGMA user_version=3')
            c.commit()
        s.close()

        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        # no chunk work is pending, yet the meta schema must keep 'schema' pending
        # (this is what makes the GUI run finalize at all)
        self.assertEqual(s.pending_stages(), ['schema'])
        self.assertTrue(s.needs_finalize())
        stages = []
        s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
        with s.meta._lock:
            uv = s.meta.conn.execute('PRAGMA user_version').fetchone()[0]
        self.assertEqual(uv, store.SCHEMA_VERSION)
        self.assertEqual(s.failed_book_ids('index'), [601])
        self.assertEqual(s.failed_entries('attr'), [{'book_id': 602, 'kind': 'attr', 'error': 'llm timeout'}])
        self.assertIsNone(s.get_meta('failed'))
        self.assertIsNone(s.get_meta('attr_failed'))
        # only the v4 step ran in this pass — no chunk-table migration
        self.assertEqual(stages, [('schema', 'moving failed-book records to their own table')])
        self._check(s)  # data intact

    def test_latest_db_has_no_schema_work(self):
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        for bid, fmt, n in self.BOOKS:
            self._index(s, bid, fmt, n)
        stages = []
        s.finalize_schema(progress=lambda st, d: stages.append((st, d)))
        self.assertEqual(stages, [])
        self.assertEqual(s.pending_stages(), [])


class TestBlockedStates(unittest.TestCase):
    """Pre-flight blocks: stored data that the installed packages cannot read."""

    DIM = 8

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'semantic-search.db')
        self.s = None
        self._orig_avail = store._module_available

    def tearDown(self):
        if self.s is not None:
            self.s.close()
        store._module_available = self._orig_avail
        self.tmp.cleanup()

    def _avail(self, name, ok):
        real = self._orig_avail
        store._module_available = lambda n: False if (n == name and not ok) else real(n)

    def test_lancedb_data_without_package_blocks(self):
        s = store.VectorStore(self.path, backend='lancedb')
        self.s = s
        vec = store.l2_normalize([1.0] * self.DIM)
        s.insert_chunk(1, _C(0, 'hello', ['Ch']), 'block-model', vec)
        s.commit(1)
        s.upsert_book(1, 'EPUB', 1, 'block-model')
        s.close()
        self.s = None
        self._avail('lancedb', False)
        try:
            with self.assertRaises(store.MissingDependencyError) as cm:
                store.VectorStore(self.path)
            self.assertEqual(cm.exception.dep, 'lancedb')
            self.assertTrue(cm.exception.has_data)
        finally:
            self._avail('lancedb', True)

    def test_fresh_lancedb_default_without_package_blocks_without_data(self):
        self._avail('lancedb', False)
        try:
            with self.assertRaises(store.MissingDependencyError) as cm:
                store.VectorStore(self.path, backend='lancedb')
            self.assertEqual(cm.exception.dep, 'lancedb')
            self.assertFalse(cm.exception.has_data)
        finally:
            self._avail('lancedb', True)

    def test_pending_switch_to_unavailable_backend_does_not_block(self):
        # data in sqlite stays readable while the wanted backend's package is missing;
        # the migration simply waits (and fails with a clear error when attempted)
        s = store.VectorStore(self.path, backend='sqlite')
        self.s = s
        vec = store.l2_normalize([1.0] * self.DIM)
        s.insert_chunk(1, _C(0, 'hello', ['Ch']), 'block-model', vec)
        s.commit(1)
        s.upsert_book(1, 'EPUB', 1, 'block-model')
        s.set_meta(store.BACKEND_KEY, 'lancedb')
        self._avail('lancedb', False)
        try:
            # the open store is fine...
            self.assertEqual(s.backend_name, 'sqlite')
            self.assertEqual(s.pending_stages(), ['backend'])
            # ...and a fresh open of it is fine too (data readable, switch pending)
            s.close()
            self.s = store.VectorStore(self.path)
            self.assertEqual(self.s.backend_name, 'sqlite')
            self.assertEqual(self.s.want_backend, 'lancedb')
            with self.assertRaises(store.MissingDependencyError):
                self.s.finalize_schema()  # the migration itself names the missing package
        finally:
            self._avail('lancedb', True)


class TestPerLibraryIndependence(_MigrateOps, unittest.TestCase):
    """Two libraries in different backends, open at once, do not interfere."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_a = os.path.join(self.tmp.name, 'liba.db')
        self.path_b = os.path.join(self.tmp.name, 'libb.db')
        self.sa = None
        self.sb = None

    def tearDown(self):
        for s in (self.sa, self.sb):
            if s is not None:
                s.close()
        self.tmp.cleanup()

    def test_two_backends_side_by_side(self):
        sa = store.VectorStore(self.path_a, backend='sqlite')
        sb = store.VectorStore(self.path_b, backend='lancedb')
        self.sa, self.sb = sa, sb
        for s, bid in ((sa, 701), (sb, 702)):
            chunks = [_C(i, f'book {bid} chunk {i}', ['Ch']) for i in range(2)]
            for i, c in enumerate(chunks):
                s.insert_chunk(bid, c, self.MODEL, store.l2_normalize([math.cos(0.3 * (bid + i) + t) for t in range(self.DIM)]))
            s.commit(bid)
            s.upsert_book(bid, 'EPUB', 2, self.MODEL)
        ra = sa.search([1.0] * self.DIM, limit=5, min_score=-1.0)
        rb = sb.search([1.0] * self.DIM, limit=5, min_score=-1.0)
        self.assertEqual({r.book_id for r in ra}, {701})
        self.assertEqual({r.book_id for r in rb}, {702})
        self.assertEqual(sa.backend_name, 'sqlite')
        self.assertEqual(sb.backend_name, 'lancedb')


if __name__ == '__main__':
    unittest.main()
