import math
import os
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

store = load('store')


@dataclass
class _C:
    chunk_no: int
    text: str
    chapter_path: list = None
    para_start: int = 0
    para_end: int = 0
    char_offset: int = 0


class TestVectorStore(unittest.TestCase):
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
            s.insert_chunk(1, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, 'test-model', dim, store.l2_normalize(v))
        s.commit()
        s.upsert_book(1, 'EPUB', 5, 'test-model', dim)

        self.assertTrue(s.book_is_indexed(1))
        books = s.indexed_books()
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]['fmt'], 'EPUB')
        self.assertEqual(books[0]['n_chunks'], 5)

        results = s.search(q, limit=3)
        self.assertEqual(len(results), 3)
        # scores descending
        for a, b in zip(results, results[1:]):
            self.assertGreaterEqual(a.score, b.score)
        # top result is book 1 with valid fields
        r = results[0]
        self.assertEqual(r.book_id, 1)
        self.assertEqual(r.fmt, 'EPUB')
        self.assertIn('text of chunk', r.text)
        self.assertEqual(r.chapter_path, ['Ch', f'S{r.chunk_no}'])

    def test_search_min_score_filters(self):
        s = self.s
        chunks = self._chunks()
        import math

        dim = 8
        vecs = [[math.cos(i * 0.3 + t) for t in range(dim)] for i in range(5)]
        q = [1.0] * dim
        for c, v in zip(chunks, vecs):
            s.insert_chunk(1, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, 'test-model', dim, store.l2_normalize(v))
        s.commit()
        s.upsert_book(1, 'EPUB', 5, 'test-model', dim)

        all_res = s.search(q, limit=10)
        self.assertEqual(len(all_res), 5)
        scores = [r.score for r in all_res]
        # threshold between the 2nd and 3rd best score keeps exactly the top 2
        mid = (scores[1] + scores[2]) / 2.0
        top2 = s.search(q, limit=10, min_score=mid)
        self.assertEqual([r.chunk_no for r in top2], [r.chunk_no for r in all_res[:2]])
        # threshold above the best score returns nothing
        self.assertEqual(s.search(q, limit=10, min_score=scores[0] + 0.001), [])

    def test_dim_mismatch_isolated(self):
        s = self.s
        chunks = self._chunks()[:1]
        for c in chunks:
            s.insert_chunk(7, c, c.text, [], 0, 0, 0, 'model-a', 4, store.l2_normalize([1.0, 0.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(7, 'EPUB', 1, 'model-a', 4)
        # query with different dim -> no results (no cross-dim search)
        self.assertEqual(s.search([1.0] * 8), [])

    def test_dirty_queue(self):
        s = self.s
        s.add_dirty(1, 'EPUB', 'added')
        s.add_dirty(2, 'MOBI', 'changed')
        self.assertEqual(sorted(s.dirty_book_ids()), [1, 2])
        s.remove_dirty(1)
        self.assertEqual(s.dirty_book_ids(), [2])

    def test_clear_book(self):
        s = self.s
        chunks = self._chunks()[:2]
        for c in chunks:
            s.insert_chunk(9, c, c.text, [], 0, 0, 0, 'm', 3, store.l2_normalize([1.0, 0.0, 0.0]))
        s.commit()
        s.upsert_book(9, 'EPUB', 2, 'm', 3)
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
            s.insert_chunk(book_id, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, model, dim, store.l2_normalize(v))
        s.commit()
        s.upsert_book(book_id, 'EPUB', n, model, dim)

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
        self.assertEqual(s.cleanup_stale_models(), 1)
        self.assertEqual(self._tables(), ['chunks_new_model'])
        res = s.search([1.0] * 8, limit=10, model='new-model')
        self.assertTrue(res and all(r.book_id == 1 for r in res))

    def test_stale_cleanup_keeps_in_use_models(self):
        s = self.s
        self._index_book(s, 1, 'alpha-model', 8)
        self._index_book(s, 2, 'beta_model', 8)
        self.assertEqual(s.cleanup_stale_models(), 0)
        self.assertEqual(len(self._tables()), 2)

    def test_legacy_chunks_table_migrated(self):
        self.s.close()
        # build a db in the old single-table layout
        conn = sqlite3.connect(self.path)
        conn.executescript(store.META_SCHEMA)
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
            (1, 0, 'legacy text', 'ch', 0, 2, 0, 'old-model', 4, store.vec_to_blob([1.0, 0.0, 0.0, 0.0])),
        )
        conn.execute('INSERT INTO books(id, fmt, indexed_at, n_chunks, model, dim) VALUES(?,?,?,?,?,?)', (1, 'EPUB', 0.0, 1, 'old-model', 4))
        conn.commit()
        conn.close()

        s = store.VectorStore(self.path, backend='sqlite')
        try:
            self.assertEqual(s.backend._chunk_tables(), ['chunks_old_model'])
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
                s.insert_chunk(b, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, 'bf-model', dim, v)
            s.commit()
            s.upsert_book(b, 'EPUB', n_chunks, 'bf-model', dim)

        q = store.l2_normalize([1.0] + [0.0] * (dim - 1))
        rows = s.backend.conn.execute('SELECT book_id, chunk_no, vector FROM chunks_bf_model ORDER BY id').fetchall()
        qa = q.tolist() if hasattr(q, 'tolist') else list(q)
        scored = []
        for i, (bid, cno, blob) in enumerate(rows):
            va = store.blob_to_vec(blob)
            scored.append((sum(a * b for a, b in zip(qa, va)), i, bid, cno))

        def expected_top(limit, min_score):
            sel = sorted((e for e in scored if e[0] >= min_score), key=lambda e: (-e[0], e[1]))
            return [(e[2], e[3]) for e in sel[:limit]]

        orig_min = store.SEARCH_MIN_BUDGET
        try:
            # even with the largest possible row size this is < 2500, so search must batch
            store.SEARCH_MIN_BUDGET = 96 * 1024
            self.assertLess(store.SEARCH_MIN_BUDGET // (dim * 4 + 64), len(rows))
            mid = (scored[100][0] + scored[101][0]) / 2.0
            for limit, min_score in ((7, -1.0), (100, 0.0), (3000, mid)):
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
            self.assertAlmostEqual(a, b, places=5)

    def test_zero_vector(self):
        v = store.l2_normalize([0.0, 0.0])
        self.assertEqual(list(v), [0.0, 0.0])


def _lancedb_available():
    try:
        import lancedb  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_lancedb_available(), 'lancedb not installed')
class TestVectorStoreLance(unittest.TestCase):
    """Runtime coverage for the LanceDB backend (forced, not auto)."""

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
        import math
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
        chunks = [C(i, f'text of chunk {i}', ['Ch', f'S{i}'], i, i + 3, i * 10) for i in range(5)]
        dim = 8
        vecs = [[math.cos(i * 0.3 + t) for t in range(dim)] for i in range(5)]
        for c, v in zip(chunks, vecs):
            s.insert_chunk(1, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, 'test-model', dim, store.l2_normalize(v))
        s.commit()
        s.upsert_book(1, 'EPUB', 5, 'test-model', dim)

        results = s.search([1.0] * dim, limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].book_id, 1)
        for a, b in zip(results, results[1:]):
            self.assertGreaterEqual(a.score, b.score)

        self.assertEqual(s.book_chunks_text(1), [f'text of chunk {i}' for i in range(5)])

        s.clear_book(1)
        self.assertFalse(s.book_is_indexed(1))
        self.assertEqual(s.book_chunks_text(1), [])
        self.assertEqual(s.search([1.0] * dim), [])


if __name__ == '__main__':
    unittest.main()
