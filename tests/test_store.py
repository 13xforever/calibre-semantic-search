import importlib
import importlib.util
import math
import os
import os as _os
import random
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
_v2 = importlib.import_module('sspkg.migrations.v2')


# v0 layout (pre-migration): used to build legacy DBs for the migration tests.
LEGACY_META_SCHEMA = '''
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
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
'''


@dataclass
class _C:
    chunk_no: int
    text: str
    chapter_path: list = None
    para_start: int = 0
    para_end: int = 0
    char_offset: int = 0


class _SearchBookMixin:
    """search_book checks shared by the sqlite and lancedb backend test classes."""

    def _check_search_book(self):
        s = self.s
        dim = 8
        for b in (11, 12):
            chunks = [_C(i, f'book {b} chunk {i}', ['ch'], i, i + 1, i) for i in range(4)]
            vecs = [store.l2_normalize([1.0 - 0.05 * i] + [0.1] * (dim - 1)) for i in range(4)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'sb-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', 4, 'sb-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        res = s.search_book(q, 11, model='sb-model')
        # every chunk of the book, best first, no cap
        self.assertEqual(len(res), 4)
        self.assertTrue(all(r.book_id == 11 for r in res))
        self.assertEqual([r.chunk_no for r in res], [0, 1, 2, 3])  # a_i decreases with i
        scores = [r.score for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res[0].fmt, 'EPUB')
        self.assertIn('book 11 chunk 0', res[0].text)
        # min_score keeps only the book's top chunks
        mid = (res[1].score + res[2].score) / 2.0
        top2 = s.search_book(q, 11, min_score=mid, model='sb-model')
        self.assertEqual([r.chunk_no for r in top2], [0, 1])
        # unknown book -> empty; another book's chunks never leak in
        self.assertEqual(s.search_book(q, 999, model='sb-model'), [])
        other = s.search_book(q, 12, model='sb-model')
        self.assertTrue(other and all(r.book_id == 12 for r in other))


class _DeleteChunkMixin:
    """delete_chunk checks shared by the sqlite and lancedb backend test classes."""

    def _check_delete_chunk(self):
        s = self.s
        dim = 8
        for b in (21, 22):
            chunks = [_C(i, f'book {b} chunk {i}', ['ch'], i, i + 1, i) for i in range(4)]
            vecs = [store.l2_normalize([1.0 - 0.05 * i] + [0.1] * (dim - 1)) for i in range(4)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'del-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', 4, 'del-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        # delete one chunk: it leaves the results and the registry count follows
        s.delete_chunk(21, 1)
        self.assertEqual([r.chunk_no for r in s.search_book(q, 21, model='del-model')], [0, 2, 3])
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(reg[21]['n_chunks'], 3)
        # the book is still found through its surviving chunks (its best one was kept)
        top = s.search(q, limit=10, min_score=-1.0, model='del-model')
        self.assertIn(21, {r.book_id for r in top})
        self.assertEqual(s.book_chunks_text(21), ['book 21 chunk 0', 'book 21 chunk 2', 'book 21 chunk 3'])
        # deleting a chunk that is not there changes nothing (counts never go negative)
        s.delete_chunk(22, 99)
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(reg[22]['n_chunks'], 4)
        # delete every chunk of a book: the count hits zero and the book drops out of
        # search, but it stays registered as indexed (a manual deletion is not a failure)
        for i in range(4):
            s.delete_chunk(22, i)
        reg = {b['id']: b for b in s.indexed_books()}
        self.assertEqual(reg[22]['n_chunks'], 0)
        self.assertTrue(s.book_is_indexed(22))
        self.assertNotIn(22, {r.book_id for r in s.search(q, limit=10, min_score=-1.0, model='del-model')})
        self.assertEqual(s.search_book(q, 22, model='del-model'), [])


class TestVectorStore(_SearchBookMixin, _DeleteChunkMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'test.db')
        self.s = store.VectorStore(self.path)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def _chunks(self):
        from dataclasses import dataclass

        @dataclass
        class C:
            chunk_no: int
            text: str
            chapter_path: list = None
            para_start: int = 0
            para_end: int = 0
            char_offset: int = 0

        return [C(i, f'text of chunk {i}', ['Ch', f'S{i}'], i, i + 3, i * 10) for i in range(5)]

    def test_roundtrip_and_search(self):
        s = self.s
        chunks = self._chunks()
        # deterministic pseudo-vectors: chunk i is close to query vector q
        import math

        dim = 8
        vecs = []
        for i in range(5):
            v = [math.cos(i * 0.3 + t) for t in range(dim)]
            vecs.append(v)
        q = [1.0] * dim  # closest to chunk 0 (all cos small positive)
        for c, v in zip(chunks, vecs):
            s.insert_chunk(1, c, 'test-model', store.l2_normalize(v))
        s.commit()
        s.upsert_book(1, 'EPUB', 5, 'test-model')

        self.assertTrue(s.book_is_indexed(1))
        books = s.indexed_books()
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]['fmt'], 'EPUB')
        self.assertEqual(books[0]['n_chunks'], 5)

        # one result per book: the single book's best chunk, whatever limit asks for
        results = s.search(q, limit=3)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.book_id, 1)
        self.assertEqual(r.fmt, 'EPUB')
        self.assertIn('text of chunk', r.text)
        self.assertEqual(r.chapter_path, ['Ch', f'S{r.chunk_no}'])

    def test_search_returns_best_per_book(self):
        s = self.s
        # 3 books x 3 chunks; a decreases globally so every score is distinct and
        # book b's best chunk is its first one (query e_0, score monotone in a)
        dim = 8
        for b in range(1, 4):
            chunks = [_C(i, f'book {b} chunk {i}', ['ch'], i, i + 1, i) for i in range(3)]
            vecs = [store.l2_normalize([1.0 - 0.001 * ((b - 1) * 3 + i)] + [0.1] * (dim - 1)) for i in range(3)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'pm-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', 3, 'pm-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        res = s.search(q, limit=10, model='pm-model')
        self.assertEqual([r.book_id for r in res], [1, 2, 3])
        for r in res:
            self.assertEqual(r.chunk_no, 0)  # each book's best chunk
        scores = [r.score for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # a smaller limit truncates the books, never duplicates them
        top2 = s.search(q, limit=2, model='pm-model')
        self.assertEqual([r.book_id for r in top2], [1, 2])

    def test_search_min_score_filters(self):
        s = self.s
        # two books; the threshold lands between the books' best chunks while book 1's
        # worst chunk is already below it (book-level filtering, not chunk-level)
        dim = 8
        for b in range(1, 3):
            chunks = [_C(i, f'book {b} chunk {i}', ['ch'], i, i + 1, i) for i in range(3)]
            vecs = [store.l2_normalize([1.0 - 0.05 * ((b - 1) * 3 + i)] + [0.1] * (dim - 1)) for i in range(3)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'pm-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', 3, 'pm-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        all_res = s.search(q, limit=10, model='pm-model')
        self.assertEqual([r.book_id for r in all_res], [1, 2])
        bests = {r.book_id: r.score for r in all_res}
        mid = (bests[1] + bests[2]) / 2.0
        top1 = s.search(q, limit=10, min_score=mid, model='pm-model')
        self.assertEqual([r.book_id for r in top1], [1])
        # threshold above the best score returns nothing
        self.assertEqual(s.search(q, limit=10, min_score=bests[1] + 0.001, model='pm-model'), [])

    def test_search_book(self):
        self._check_search_book()

    def test_delete_chunk(self):
        self._check_delete_chunk()

    def test_dim_mismatch_isolated(self):
        s = self.s
        chunks = self._chunks()[:1]
        for c in chunks:
            s.insert_chunk(7, c, 'model-a', store.l2_normalize([1.0, 0.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(7, 'EPUB', 1, 'model-a')
        # query with different dim -> no results (no cross-dim search)
        self.assertEqual(s.search([1.0] * 8), [])

    def test_dirty_queue(self):
        s = self.s
        s.enqueue_indexing(1)
        s.enqueue_indexing(2)
        # newest first: descending book id (calibre assigns ids in insertion order)
        self.assertEqual(s.queued_indexing_book_ids(), [2, 1])
        s.dequeue_indexing(1)
        self.assertEqual(s.queued_indexing_book_ids(), [2])

    def test_clear_book(self):
        s = self.s
        chunks = self._chunks()[:2]
        for c in chunks:
            s.insert_chunk(9, c, 'm', store.l2_normalize([1.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(9, 'EPUB', 2, 'm')
        self.assertTrue(s.book_is_indexed(9))
        s.clear_book(9)
        self.assertFalse(s.book_is_indexed(9))
        self.assertEqual(s.search([1.0] * 3), [])

    def test_meta(self):
        s = self.s
        self.assertIsNone(s.get_meta('k'))
        s.set_meta('k', 'v1')
        s.set_meta('k', 'v2')
        self.assertEqual(s.get_meta('k'), 'v2')
        s.delete_meta('k')
        self.assertIsNone(s.get_meta('k'))
        s.delete_meta('missing')  # no error
        s.set_meta('fileinfo:1', 'EPUB|1|2')
        s.set_meta('other', 'x')
        self.assertEqual(s.meta_keys('fileinfo:'), ['fileinfo:1'])

    def test_attrs_clear(self):
        s = self.s
        self.assertEqual(s.get_attrs(4), {})
        s.set_attrs(4, {'gender': 'f'})
        self.assertEqual(s.get_attrs(4)['gender'], 'f')
        s.set_attrs(5, {'gender': 'm'})
        self.assertEqual(sorted(s.attr_book_ids()), [4, 5])
        s.clear_attrs(4)
        self.assertEqual(s.get_attrs(4), {})
        self.assertEqual(s.attr_book_ids(), [5])

    def test_enqueue_indexing_many(self):
        s = self.s
        s.enqueue_indexing_many([1, 2, 3])
        self.assertEqual(s.queued_indexing_book_ids(), [3, 2, 1])
        # upsert keeps the highest priority tier: no duplicates on re-enqueue
        s.enqueue_indexing(2)
        self.assertEqual(sorted(s.queued_indexing_book_ids()), [1, 2, 3])
        s.dequeue_indexing(2)
        self.assertEqual(sorted(s.queued_indexing_book_ids()), [1, 3])
        s.enqueue_indexing_many([])  # empty is a no-op, not an error
        self.assertEqual(sorted(s.queued_indexing_book_ids()), [1, 3])

    def test_queue_indexing_priority(self):
        s = self.s
        # default-tier books are ordered newest-first; a per-book (high) tier jumps ahead
        s.enqueue_indexing(1)
        s.enqueue_indexing(2)
        s.enqueue_indexing(3, store.PRIORITY_BOOK)
        self.assertEqual(s.queued_indexing_book_ids(), [3, 2, 1])
        # re-enqueueing a high-tier book at the default tier must not downgrade it
        s.enqueue_indexing(3)
        self.assertEqual(s.queued_indexing_book_ids(), [3, 2, 1])

    def test_attr_queue(self):
        s = self.s
        # whole-library (default) batch first, then a per-book force jumps to the front
        s.enqueue_attrs_all([1, 2, 3])
        self.assertEqual(s.queued_attrs_book_ids(), [3, 2, 1])
        s.enqueue_attrs(2)
        self.assertEqual(s.queued_attrs_book_ids(), [2, 3, 1])
        # a whole-library enqueue never downgrades an existing per-book row
        s.enqueue_attrs_all([2])
        self.assertEqual(s.queued_attrs_book_ids(), [2, 3, 1])
        s.dequeue_attrs(2)
        self.assertEqual(s.queued_attrs_book_ids(), [3, 1])

    def test_attrs_fields(self):
        s = self.s
        self.assertEqual(s.attrs_fields(), {})
        s.set_attrs(4, {'gender': 'f', 'tropes': ['x']})
        s.set_attrs(5, {})
        got = s.attrs_fields()
        self.assertEqual(got[4], frozenset({'gender', 'tropes'}))
        self.assertEqual(got[5], frozenset())  # stored-but-empty still counts as a row
        s.clear_attrs(4)
        self.assertEqual(s.attrs_fields(), {5: frozenset()})


class TestSqlitePerModel(unittest.TestCase):
    """sqlite backend: per-model tables, legacy migration, stale cleanup, batched search."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'test.db')
        self.s = store.VectorStore(self.path, backend='sqlite')

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def _tables(self):
        rows = self.s.backend.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return sorted(r[0] for r in rows if r[0].startswith('chunks_'))

    def _index_book(self, s, book_id, model, dim, n=3):
        chunks = [_C(i, f'{model} chunk {i}', ['ch'], i, i + 1, i) for i in range(n)]
        vecs = [[math.cos(i * 0.7 + t) for t in range(dim)] for i in range(n)]
        for c, v in zip(chunks, vecs):
            s.insert_chunk(book_id, c, model, store.l2_normalize(v))
        s.commit()
        s.upsert_book(book_id, 'EPUB', n, model)

    def test_one_table_per_model(self):
        s = self.s
        self._index_book(s, 1, 'alpha-model', 8)
        self._index_book(s, 2, 'beta_model', 8)
        self.assertEqual(self._tables(), ['chunks_alpha_model', 'chunks_beta_model'])

    def test_search_scoped_to_model(self):
        s = self.s
        self._index_book(s, 1, 'alpha-model', 8)
        self._index_book(s, 2, 'beta_model', 8)
        a = s.search([1.0] * 8, limit=10, model='alpha-model')
        b = s.search([1.0] * 8, limit=10, model='beta_model')
        self.assertTrue(a and all(r.book_id == 1 for r in a))
        self.assertTrue(b and all(r.book_id == 2 for r in b))
        both = s.search([1.0] * 8, limit=10)
        self.assertEqual({r.book_id for r in both}, {1, 2})

    def test_search_unknown_model_empty(self):
        s = self.s
        self._index_book(s, 1, 'alpha-model', 8)
        self.assertEqual(s.search([1.0] * 8, model='no-such-model'), [])

    def test_dim_mismatch_skips_table(self):
        s = self.s
        self._index_book(s, 7, 'model-a', 4)
        self.assertEqual(s.search([1.0] * 8), [])

    def test_stale_model_table_dropped(self):
        s = self.s
        self._index_book(s, 1, 'old-model', 8)
        self.assertIn('chunks_old_model', self._tables())
        # re-index the book under a new model (what "Re-index all books" does per book)
        s.clear_book(1)
        self._index_book(s, 1, 'new-model', 8)
        self.assertEqual(s.cleanup_stale_models('new-model'), 1)
        self.assertEqual(self._tables(), ['chunks_new_model'])
        res = s.search([1.0] * 8, limit=10, model='new-model')
        self.assertTrue(res and all(r.book_id == 1 for r in res))

    def test_stale_cleanup_keeps_in_use_models(self):
        s = self.s
        self._index_book(s, 1, 'alpha-model', 8)
        self._index_book(s, 2, 'beta_model', 8)
        self.assertEqual(s.cleanup_stale_models('alpha-model'), 0)
        self.assertEqual(len(self._tables()), 2)

    def test_legacy_chunks_table_migrated(self):
        self.s.close()
        for suffix in ('', '-wal', '-shm'):  # setUp already created a fresh db at this path
            p = self.path + suffix
            if os.path.exists(p):
                os.remove(p)
        # build a db in the old single-table layout (v0 meta schema + legacy chunks)
        conn = sqlite3.connect(self.path)
        conn.executescript(LEGACY_META_SCHEMA)
        conn.execute(
            '''CREATE TABLE chunks(
                id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_no INTEGER NOT NULL,
                text TEXT NOT NULL, chapter_path TEXT NOT NULL DEFAULT '',
                para_start INTEGER NOT NULL DEFAULT 0, para_end INTEGER NOT NULL DEFAULT 0,
                char_offset INTEGER NOT NULL DEFAULT 0, model TEXT NOT NULL, dim INTEGER NOT NULL,
                vector BLOB NOT NULL)'''
        )
        conn.execute(
            'INSERT INTO chunks(book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector) '
            'VALUES(?,?,?,?,?,?,?,?,?,?)',
            (1, 0, 'legacy text', 'ch', 0, 2, 0, 'old-model', 4, store.vec_to_blob([1.0, 0.0, 0.0, 0.0], f32=True)),
        )
        conn.execute('INSERT INTO books(id, fmt, indexed_at, n_chunks, model, dim) VALUES(?,?,?,?,?,?)', (1, 'EPUB', 0.0, 1, 'old-model', 4))
        conn.execute("INSERT INTO meta(key, value) VALUES('fileinfo:1', 'EPUB|100|1598092103.694143')")
        conn.commit()
        conn.close()

        s = store.VectorStore(self.path, backend='sqlite')
        try:
            # legacy mtime is truncated to whole seconds, never rounded into the future
            self.assertEqual(s.get_file_info(1), {'fmt': 'EPUB', 'size': 100, 'mtime_s': 1598092103})
            self.assertTrue(s.needs_finalize())
            with s.meta._lock:
                ac_before = s.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            ac_during = {}
            orig_slim = _v2.slim_table

            def slim_probe(backend, t):
                with s.meta._lock:
                    ac_during['v'] = s.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
                return orig_slim(backend, t)
            _v2.slim_table = slim_probe
            try:
                s.finalize_schema()
            finally:
                _v2.slim_table = orig_slim
            self.assertFalse(s.needs_finalize())
            self.assertEqual(ac_during.get('v'), 0)  # auto-checkpoint disabled during migration
            with s.meta._lock:
                ac_after = s.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            self.assertEqual(ac_after, ac_before)  # ...and restored afterwards
            self.assertEqual(s.backend._chunk_tables(), ['chunks_old_model'])
            with s.meta._lock:
                uv = s.meta.conn.execute('PRAGMA user_version').fetchone()[0]
                blob_len = s.meta.conn.execute('SELECT LENGTH(vector) FROM chunks_old_model LIMIT 1').fetchone()[0]
            self.assertEqual(uv, store.SCHEMA_VERSION)  # v3: the f32 blobs were converted to half floats
            self.assertEqual(blob_len, 2 * 4)  # two bytes per component
            res = s.search([1.0] * 4, limit=5)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0].book_id, 1)
            self.assertEqual(res[0].text, 'legacy text')
            self.assertEqual(s.book_chunks_text(1), ['legacy text'])
        finally:
            s.close()

    def test_batched_search_matches_bruteforce(self):
        # 2500 rows whose scores are analytically distinct and decreasing in row order:
        # v_i = [a_i, 0.1, ..., 0.1] normalized, query = e_0 -> score = a_i / |v_i|
        s = self.s
        dim = 64
        n_books, n_chunks = 5, 500
        for b in range(1, n_books + 1):
            chunks = [_C(i, f'book {b} chunk {i} ' + 'x' * 20, [], i, i + 1, i) for i in range(n_chunks)]
            vecs = [store.l2_normalize([1.0 - 0.001 * (b - 1) * n_chunks - 0.001 * i] + [0.1] * (dim - 1)) for i in range(n_chunks)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'bf-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', n_chunks, 'bf-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        rows = s.backend.conn.execute('SELECT book_id, chunk_no, vector FROM chunks_bf_model ORDER BY id').fetchall()
        qa = q.tolist() if hasattr(q, 'tolist') else list(q)
        scored = []
        for i, (bid, cno, blob) in enumerate(rows):
            va = store.blob_to_vec(blob)
            scored.append((sum(a * b for a, b in zip(qa, va)), i, bid, cno))

        def expected_top(limit, min_score):
            # per-book best (first row wins ties, like the scan), then top books
            per_book = {}
            for e in scored:
                if e[0] < min_score:
                    continue
                cur = per_book.get(e[2])
                if cur is None or e[0] > cur[0]:
                    per_book[e[2]] = e
            sel = sorted(per_book.values(), key=lambda e: (-e[0], e[2]))[:limit]
            return [(e[2], e[3]) for e in sel]

        orig_min = store.SEARCH_MIN_BUDGET
        try:
            # even with the largest possible row size this is < 2500, so search must batch
            store.SEARCH_MIN_BUDGET = 96 * 1024
            self.assertLess(store.SEARCH_MIN_BUDGET // (dim * 6 + 64), len(rows))
            # between book 3's and book 4's best chunks: keeps exactly books 1-3
            mid = (scored[2 * n_chunks][0] + scored[3 * n_chunks][0]) / 2.0
            for limit, min_score in ((2, -1.0), (7, -1.0), (10, mid)):
                got = s.search(q, limit=limit, min_score=min_score, model='bf-model')
                self.assertEqual([(r.book_id, r.chunk_no) for r in got], expected_top(limit, min_score))
        finally:
            store.SEARCH_MIN_BUDGET = orig_min


class TestVecHelpers(unittest.TestCase):
    def test_normalize_unit_length(self):
        v = store.l2_normalize([3.0, 4.0])
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_blob_roundtrip(self):
        v = [0.1, -0.2, 0.3]
        blob = store.vec_to_blob(v)
        back = store.blob_to_vec(blob)
        for a, b in zip(v, list(back)):
            self.assertAlmostEqual(a, b, places=3)  # half-precision rounding

    def test_blob_roundtrip_f32(self):
        v = [0.1, -0.2, 0.3]
        blob = store.vec_to_blob(v, f32=True)
        back = store.blob_to_vec(blob, False)
        for a, b in zip(v, list(back)):
            self.assertAlmostEqual(a, b, places=7)

    def test_zero_vector(self):
        v = store.l2_normalize([0.0, 0.0])
        self.assertEqual(list(v), [0.0, 0.0])


class TestHalfLut(unittest.TestCase):
    """The no-numpy decode path: the LUT must be exact for every half bit pattern."""

    @staticmethod
    def _naive(blob):
        # independent reference: the original per-bit double-precision decode
        out = []
        for i in range(0, len(blob), 2):
            h = blob[i] | (blob[i + 1] << 8)
            sign = -1.0 if h & 0x8000 else 1.0
            exp = (h >> 10) & 0x1F
            mant = h & 0x3FF
            if exp == 0x1F:
                v = float('inf') if mant == 0 else float('nan')
            elif exp == 0:
                v = mant * 2.0 ** -24
            else:
                v = (mant + 1024) * 2.0 ** (exp - 25)
            out.append(sign * v)
        return out

    def test_random_blobs_match_naive_reference(self):
        rnd = random.Random(1)
        for _ in range(64):
            n = rnd.randrange(1, 512)
            blob = bytes(rnd.randrange(256) for _ in range(2 * n))
            got = store._half_bytes_to_floats(blob)
            want = self._naive(blob)
            self.assertEqual(len(got), len(want))
            for a, b in zip(got, want):
                if math.isnan(a) and math.isnan(b):
                    continue
                self.assertEqual(a, b)  # exact: the f32 LUT entries are lossless

    def test_special_values(self):
        # +1.0, -1.0, max finite (65504), +inf, nan, zero, smallest subnormal
        blob = bytes.fromhex('003C 00BC FF7B 007C 007E 0000 0100'.replace(' ', ''))
        got = store._half_bytes_to_floats(blob)
        self.assertEqual(got[0], 1.0)
        self.assertEqual(got[1], -1.0)
        self.assertEqual(got[2], 65504.0)
        self.assertEqual(got[3], float('inf'))
        self.assertTrue(math.isnan(got[4]))
        self.assertEqual(got[5], 0.0)
        self.assertEqual(got[6], 2.0 ** -24)

    def test_empty_blob(self):
        self.assertEqual(store._half_bytes_to_floats(b''), [])


class TestVectorStoreLance(_SearchBookMixin, _DeleteChunkMixin, unittest.TestCase):
    """Runtime coverage for the LanceDB backend (forced, not auto).

    Not skipped when lancedb is missing: without the package, VectorStore raises a
    clear install error and the tests fail loudly instead of passing silently."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'test.db')
        self.s = store.VectorStore(self.path, backend='lancedb')

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_backend_selected(self):
        self.assertEqual(self.s.backend_name, 'lancedb')

    def test_roundtrip_search_and_chunks_text(self):
        from dataclasses import dataclass

        @dataclass
        class C:
            chunk_no: int
            text: str
            chapter_path: list = None
            para_start: int = 0
            para_end: int = 0
            char_offset: int = 0

        s = self.s
        dim = 8
        # two books; per-book bests well separated (a drops 0.2 per book, 0.02 per chunk)
        for b in (1, 2):
            chunks = [C(i, f'text of chunk {i} of book {b}', ['Ch', f'S{i}'], i, i + 3, i * 10) for i in range(5)]
            vecs = [store.l2_normalize([1.0 - 0.2 * (b - 1) - 0.02 * i] + [0.1] * (dim - 1)) for i in range(5)]
            for c, v in zip(chunks, vecs):
                s.insert_chunk(b, c, 'test-model', v)
            s.commit()
            s.upsert_book(b, 'EPUB', 5, 'test-model')

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        results = s.search(q, limit=3)
        self.assertEqual(len(results), 2)  # one result per book, not per chunk
        self.assertEqual([r.book_id for r in results], [1, 2])
        for a, b in zip(results, results[1:]):
            self.assertGreaterEqual(a.score, b.score)
        for r in results:
            self.assertEqual(r.chunk_no, 0)  # each book's best chunk

        self.assertEqual(s.book_chunks_text(1), [f'text of chunk {i} of book 1' for i in range(5)])

        s.clear_book(1)
        s.clear_book(2)
        self.assertFalse(s.book_is_indexed(1))
        self.assertEqual(s.book_chunks_text(1), [])
        self.assertEqual(s.search(q), [])

    def test_search_book(self):
        self._check_search_book()

    def test_delete_chunk(self):
        self._check_delete_chunk()


class TestDefaultDictLoading(unittest.TestCase):
    def setUp(self):
        self._saved = getattr(store, 'get_resources', None)

    def tearDown(self):
        if self._saved is None:
            del store.get_resources
        else:
            store.get_resources = self._saved

    def test_zip_branch_reads_injected_resource(self):
        seen = []

        def fake_get_resources(name):
            seen.append(name)
            return b'zipdict'

        store.get_resources = fake_get_resources
        self.assertEqual(store._load_default_dict(), b'zipdict')
        self.assertEqual(seen, ['assets/default_compression_dict.bin'])

    def test_falls_back_to_file_when_resource_missing(self):
        store.get_resources = lambda name: None
        with open(store.DEFAULT_DICT_PATH, 'rb') as f:
            expected = f.read()
        self.assertEqual(store._load_default_dict(), expected)


class TestNumpyFallbackEquivalence(unittest.TestCase):
    """The numpy paths and the pure-Python fallbacks must produce the same results
    on the same data (store.np is patched to force each path; real dim 4096)."""

    def setUp(self):
        import numpy

        self.np = numpy
        self._saved_np = store.np

    def tearDown(self):
        store.np = self._saved_np

    def _vectors(self, n, dim, seed=7):
        rnd = random.Random(seed)
        return [[rnd.random() for _ in range(dim)] for _ in range(n)]

    def test_vec_to_blob_identical(self):
        vecs = self._vectors(16, 4096)
        store.np = self.np
        blobs_np = [store.vec_to_blob(v) for v in vecs]
        store.np = None
        blobs_py = [store.vec_to_blob(v) for v in vecs]
        self.assertEqual(blobs_np, blobs_py)

    def test_blob_to_vec_identical(self):
        vecs = self._vectors(8, 4096)
        store.np = None
        blobs = [store.vec_to_blob(v) for v in vecs]
        store.np = self.np
        got_np = [list(map(float, store.blob_to_vec(b))) for b in blobs]
        store.np = None
        got_py = [store.blob_to_vec(b) for b in blobs]
        self.assertEqual(got_np, got_py)

    def test_l2_normalize_equivalent(self):
        vecs = self._vectors(8, 4096)
        store.np = self.np
        norm_np = [list(map(float, store.l2_normalize(v))) for v in vecs]
        store.np = None
        norm_py = [store.l2_normalize(v) for v in vecs]
        for a, b in zip(norm_np, norm_py):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, delta=1e-6)
        for lst in norm_np + norm_py:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in lst)), 1.0, places=5)

    def test_l2_normalize_zero_vector(self):
        store.np = self.np
        z_np = list(map(float, store.l2_normalize([0.0] * 8)))
        store.np = None
        z_py = store.l2_normalize([0.0] * 8)
        self.assertEqual(z_np, [0.0] * 8)
        self.assertEqual(z_py, [0.0] * 8)

    def test_score_rows_equivalent(self):
        backend = object.__new__(store.SqliteVectorBackend)  # _score_rows does not use self
        dim = 4096
        vecs = self._vectors(256, dim)
        qv_src = self._vectors(1, dim, seed=99)[0]
        # stored vectors are unit-norm in production (normalized at insert time),
        # so scores land in [-1, 1] and both paths agree to float32 precision
        store.np = None
        rows = [(i, store.vec_to_blob(store.l2_normalize(v))) for i, v in enumerate(vecs)]  # last column is the vector blob
        store.np = self.np
        scores_np = backend._score_rows(rows, store.l2_normalize(qv_src), True)  # rows hold f16 blobs
        store.np = None
        scores_py = backend._score_rows(rows, store.l2_normalize(qv_src), True)
        self.assertEqual(len(scores_np), len(scores_py))
        for a, b in zip(scores_np, scores_py):
            self.assertAlmostEqual(a, b, delta=1e-5)


class TestFinalizeCancel(unittest.TestCase):
    """finalize_schema(cancel=...) stops at the next checkpoint and stays resumable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'test.db')
        self.s = store.VectorStore(self.path)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_pre_set_cancel_raises_before_any_work(self):
        import threading

        ev = threading.Event()
        ev.set()
        with self.assertRaises(store.FinalizeCancelled):
            self.s.finalize_schema(cancel=ev)

    def test_uncancelled_run_completes(self):
        # a fresh store has nothing pending: the run is a no-op and must not raise
        self.s.finalize_schema()
        self.assertFalse(self.s.needs_finalize())

    def test_cancel_aborts_recompress_stage_resumably(self):
        import threading
        from dataclasses import dataclass

        @dataclass
        class C:
            chunk_no: int
            text: str
            chapter_path: list = None

        s = self.s
        for i in range(10):
            s.insert_chunk(1, C(i, f'text {i}', []), 'm', store.l2_normalize([1.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(1, 'EPUB', 10, 'm')
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zlib'))
        ev = threading.Event()
        ev.set()
        with self.assertRaises(store.FinalizeCancelled):
            s.finalize_schema(cancel=ev)
        # the marker survives: a later open re-runs the conversion from scratch (nothing committed)
        self.assertIsNotNone(s.get_meta(store.RECOMPRESS_KEY))

    def test_cancel_mid_conversion_rolls_back_and_reruns(self):
        # one transaction, one end commit: a cancel mid-stage discards every update,
        # the marker keeps its original progress, and a re-run converts from scratch
        import threading
        from dataclasses import dataclass
        from unittest import mock

        @dataclass
        class C:
            chunk_no: int
            text: str
            chapter_path: list = None

        s = self.s
        for i in range(10):
            s.insert_chunk(1, C(i, f'text {i}', []), 'm', store.l2_normalize([1.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(1, 'EPUB', 10, 'm')
        s.set_meta(store.RECOMPRESS_KEY, store.recompress_marker('zlib'))
        ev = threading.Event()

        def progress(stage, detail):
            if stage == 'codec':
                ev.set()  # cancel from the first progress report onward

        with mock.patch.object(s, '_recompress_batch', return_value=3):
            with self.assertRaises(store.FinalizeCancelled):
                s.finalize_schema(progress=progress, cancel=ev)
        prog = store.json.loads(s.get_meta(store.RECOMPRESS_KEY))
        self.assertEqual(prog['last_id'], 0)  # nothing was ever committed
        self.assertEqual(s.stored_codec(), 'zstd')  # the codec flip did not happen
        # re-run: the conversion completes from scratch and the text survives
        s.finalize_schema()
        self.assertIsNone(s.get_meta(store.RECOMPRESS_KEY))
        self.assertEqual(s.stored_codec(), 'zlib')
        res = s.search([1.0, 0.0, 0.0], limit=5, min_score=-1.0)
        self.assertEqual(res[0].text, 'text 0')


if __name__ == '__main__':
    unittest.main()
