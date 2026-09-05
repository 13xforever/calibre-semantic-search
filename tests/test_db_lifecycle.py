"""End-to-end store scenarios on small, reproducible databases.

Three scenarios, each exercising the full operation surface:
1. a sqlite DB built with the legacy v0 schema, migrated to the latest schema version
2. a brand-new sqlite DB (created at the latest schema)
3. a lancedb backend (no migrations; created at the latest shape)

For each: insert embeddings for a few books, attributes, search, dirty queue,
file info, removal of stale data (removed book + abandoned model table), and
persistence across a reopen.
"""
import math
import os
import sqlite3
import tempfile
import types
import importlib.util
import unittest
from dataclasses import dataclass

import os as _os, sys as _sys
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
        # -- search (min_score=-1.0: cosine range, so every stored chunk is returned) --
        res = s.search(q, limit=9, min_score=-1.0)
        self.assertEqual(len(res), 9)
        self.assertEqual((res[0].book_id, res[0].chunk_no), (901, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(res[0].text, 'book 901 chunk 0')
        self.assertEqual(res[0].chapter_path, ['Ch', 'S0'])
        self.assertEqual(res[0].fmt, 'EPUB')
        for a, b in zip(res, res[1:]):
            self.assertGreaterEqual(a.score, b.score)
        mid = (res[1].score + res[2].score) / 2.0
        top2 = s.search(q, limit=9, min_score=mid)
        self.assertEqual([(r.book_id, r.chunk_no) for r in top2], [(r.book_id, r.chunk_no) for r in res[:2]])
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

    def _assert_persisted(self, s):
        q = [1.0] * self.DIM
        reg = {b['id'] for b in s.indexed_books()}
        self.assertTrue({901, 903, 905} <= reg)
        self.assertNotIn(902, reg)
        res = s.search(q, limit=9)
        self.assertEqual((res[0].book_id, res[0].chunk_no), (901, 0))
        self.assertGreater(res[0].score, 0.999)
        self.assertEqual(s.book_chunks_text(903), [f'book 903 chunk {i}' for i in range(4)])
        self.assertEqual(s.get_attrs(901)['title'], 'Ops Book One')
        self.assertEqual(s.get_attrs(903), {})
        self.assertNotIn(904, s.dirty_book_ids())
        self.assertNotIn(901, s.file_info_book_ids())


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
        conn.execute("INSERT INTO meta(key, value) VALUES('attr_failed','{}')")
        conn.execute("INSERT INTO meta(key, value) VALUES('failed','{}')")
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
                    (bid, cn, text, f'Chapter {cn + 1}', cn * 3, cn * 3 + 2, cn * 400, m, self.LEGACY_DIM, store.vec_to_blob(vec)))
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
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(set(reg), {1, 2, 3})
        self.assertEqual((reg[1]['fmt'], reg[1]['n_chunks'], reg[1]['model']), ('EPUB', 2, 'legacy_embedding_8b'))
        with s.meta._lock:
            n = c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        self.assertEqual(n, 3)  # fat table untouched at v1
        self.assertEqual(s.backend._chunk_tables(), [])
        self.assertTrue(s.needs_finalize())

        # stage 2: finalize splits + slims the chunk tables to v2
        s.finalize_schema()
        with s.meta._lock:
            uv = c.execute('PRAGMA user_version').fetchone()[0]
            av = c.execute('PRAGMA auto_vacuum').fetchone()[0]
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            slim_cols = {r[1] for r in c.execute('PRAGMA table_info(chunks_legacy_embedding_8b)')}
        self.assertEqual(uv, 2)
        self.assertEqual(av, 2)
        self.assertNotIn('chunks', tables)
        self.assertIn('chunks_legacy_embedding_8b', tables)
        self.assertEqual(slim_cols, {'id', 'book_id', 'chunk_no', 'text_z', 'chapter_path', 'vector'})
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
        self.assertEqual(uv, 2)
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

        self._run_ops(s)

        s.close()
        self.s = store.VectorStore(self.path, backend='lancedb')
        self._assert_persisted(self.s)


if __name__ == '__main__':
    unittest.main()
