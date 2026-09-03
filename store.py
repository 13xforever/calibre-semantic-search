'''Vector store: SQLite metadata + pluggable vector backend.

Public API (used by indexer/dialog/attributes):
  VectorStore(db_path, backend='auto') -> facade with:
    add_dirty / dirty_book_ids / remove_dirty
    book_is_indexed / indexed_books / clear_book / upsert_book
    insert_chunk / commit
    search(query_vec, limit, min_score) -> list[SearchResult]
    book_chunks_text(book_id) -> list[str]
    get_meta / set_meta / close

Backends:
  'sqlite'  - vectors in a local SQLite file (always available; numpy speeds it up)
  'lancedb' - vectors in LanceDB (optional dependency, lazy import)
  'auto'    - lancedb if importable, else sqlite
'''

from __future__ import annotations

import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass

try:
    import numpy as np  # optional, big speedup for search
except ImportError:
    np = None


@dataclass
class SearchResult:
    book_id: int
    fmt: str
    chunk_no: int
    text: str
    chapter_path: list[str]
    para_start: int
    para_end: int
    char_offset: int
    score: float

    @property
    def chapter_label(self) -> str:
        return ' > '.join(self.chapter_path)


def vec_to_blob(vec) -> bytes:
    if np is not None:
        return np.asarray(vec, dtype='<f4').tobytes()
    n = len(vec)
    return struct.pack('<%df' % n, *vec)


def blob_to_vec(blob: bytes):
    if np is not None:
        return np.frombuffer(blob, dtype='<f4')
    import array

    a = array.array('f')
    a.frombytes(blob)
    return list(a)


def l2_normalize(vec):
    if np is not None:
        arr = np.asarray(vec, dtype='<f4')
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr
        return (arr / norm).astype('<f4')
    s = sum(x * x for x in vec) ** 0.5
    if s == 0.0:
        return list(vec)
    return [x / s for x in vec]


META_SCHEMA = '''
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

CHUNKS_SCHEMA = '''
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
'''


class MetaStore:
    """SQLite bookkeeping: dirty queue, indexed-book registry, key/value meta."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        with self._lock:
            self.conn.executescript(META_SCHEMA)
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    # -- meta ------------------------------------------------------------------

    def get_meta(self, key: str, default=None):
        with self._lock:
            row = self.conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str):
        with self._lock:
            self.conn.execute(
                'INSERT INTO meta(key, value) VALUES(?, ?) '
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (key, value),
            )
            self.conn.commit()

    # -- dirty queue -------------------------------------------------------------

    def add_dirty(self, book_id: int, fmt: str, reason: str = 'added'):
        with self._lock:
            self.conn.execute(
                'INSERT INTO dirty(book_id, fmt, reason, added_at) VALUES(?,?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET fmt=excluded.fmt, reason=excluded.reason, added_at=excluded.added_at',
                (book_id, fmt, reason, time.time()),
            )
            self.conn.commit()

    def dirty_book_ids(self):
        with self._lock:
            rows = self.conn.execute('SELECT book_id FROM dirty').fetchall()
        return [r[0] for r in rows]

    def remove_dirty(self, book_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM dirty WHERE book_id=?', (book_id,))
            self.conn.commit()

    # -- books registry ------------------------------------------------------------

    def clear_book(self, book_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM books WHERE id=?', (book_id,))
            self.conn.commit()

    def book_is_indexed(self, book_id: int) -> bool:
        with self._lock:
            row = self.conn.execute('SELECT 1 FROM books WHERE id=?', (book_id,)).fetchone()
        return row is not None

    def indexed_books(self):
        with self._lock:
            rows = self.conn.execute('SELECT id, fmt, n_chunks, model, dim, indexed_at FROM books').fetchall()
        return [
            {'id': r[0], 'fmt': r[1], 'n_chunks': r[2], 'model': r[3], 'dim': r[4], 'indexed_at': r[5]} for r in rows
        ]

    def upsert_book(self, book_id: int, fmt: str, n_chunks: int, model: str, dim: int):
        with self._lock:
            self.conn.execute(
                'INSERT INTO books(id, fmt, indexed_at, n_chunks, model, dim) VALUES(?,?,?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET fmt=excluded.fmt, indexed_at=excluded.indexed_at, '
                'n_chunks=excluded.n_chunks, model=excluded.model, dim=excluded.dim',
                (book_id, fmt, time.time(), n_chunks, model, dim),
            )
            self.conn.commit()


class SqliteVectorBackend:
    """Vectors + chunk text in the same SQLite file."""

    name = 'sqlite'

    def __init__(self, meta: MetaStore):
        self.meta = meta
        self.conn = meta.conn  # share connection/lock
        with meta._lock:
            self.conn.executescript(CHUNKS_SCHEMA)
            self.conn.commit()

    def delete_book(self, book_id: int):
        with self.meta._lock:
            self.conn.execute('DELETE FROM chunks WHERE book_id=?', (book_id,))

    def insert_chunks(self, book_id: int, items):
        """items: list of (chunk, vector) already normalized."""
        rows = [
            (book_id, c.chunk_no, c.text, ' > '.join(c.chapter_path), c.para_start, c.para_end, c.char_offset, model, dim, vec_to_blob(v))
            for c, v, model, dim in items
        ]
        with self.meta._lock:
            self.conn.executemany(
                'INSERT INTO chunks(book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector) '
                'VALUES(?,?,?,?,?,?,?,?,?,?)',
                rows,
            )

    def commit(self):
        with self.meta._lock:
            self.conn.commit()

    def book_chunks_text(self, book_id: int):
        with self.meta._lock:
            rows = self.conn.execute('SELECT text FROM chunks WHERE book_id=? ORDER BY chunk_no', (book_id,)).fetchall()
        return [r[0] for r in rows]

    def search(self, query_vec, limit: int, min_score: float) -> list[SearchResult]:
        qv = l2_normalize(query_vec)
        dim = qv.shape[0] if np is not None else len(qv)
        with self.meta._lock:
            rows = self.conn.execute(
                'SELECT c.book_id, c.chunk_no, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, b.fmt, c.vector '
                'FROM chunks c LEFT JOIN books b ON b.id=c.book_id WHERE c.dim=?',
                (dim,),
            ).fetchall()
        if not rows:
            return []
        qblob = vec_to_blob(qv)
        score_map = {}
        if np is not None:
            q = np.frombuffer(qblob, dtype='<f4')
            for i, r in enumerate(rows):
                v = np.frombuffer(r[8], dtype='<f4')
                score_map[i] = float(np.dot(q, v))
        else:
            import array

            qa = array.array('f')
            qa.frombytes(qblob)
            for i, r in enumerate(rows):
                va = array.array('f')
                va.frombytes(r[8])
                s = 0.0
                for a, b in zip(qa, va):
                    s += a * b
                score_map[i] = s
        order = sorted(score_map, key=lambda i: score_map[i], reverse=True)
        out = []
        for i in order:
            s = score_map[i]
            if s < min_score:
                break
            r = rows[i]
            out.append(
                SearchResult(
                    book_id=r[0],
                    fmt=r[7] or '',
                    chunk_no=r[1],
                    text=r[2],
                    chapter_path=[p for p in r[3].split(' > ') if p],
                    para_start=r[4],
                    para_end=r[5],
                    char_offset=r[6],
                    score=s,
                )
            )
            if len(out) >= limit:
                break
        return out


class LanceVectorBackend:
    """Vectors + chunk text in LanceDB (optional dependency).

    One table per embedding model so dimension changes never mix vectors.
    """

    name = 'lancedb'

    def __init__(self, meta: MetaStore):
        import lancedb  # lazy: optional dependency

        self.meta = meta
        db_path = os.path.splitext(meta.db_path)[0] + '-lancedb'
        self._db = lancedb.connect(db_path)
        self._tables: dict[str, object] = {}

    def _table_name(self, model: str) -> str:
        import re

        slug = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_').lower() or 'model'
        return f'chunks_{slug}'[:80]

    def _table_names(self):
        res = self._db.list_tables()
        tables = getattr(res, 'tables', None)  # newer lancedb returns a page object
        if tables is None:
            return list(res)
        return list(tables)

    def _get_table(self, model: str, dim: int):
        name = self._table_name(model)
        t = self._tables.get(name)
        if t is not None:
            return t
        if name in self._table_names():
            t = self._db.open_table(name)
        else:
            import pyarrow as pa

            schema = pa.schema(
                [
                    ('book_id', pa.int64()),
                    ('chunk_no', pa.int32()),
                    ('text', pa.string()),
                    ('chapter_path', pa.string()),
                    ('para_start', pa.int32()),
                    ('para_end', pa.int32()),
                    ('char_offset', pa.int64()),
                    ('dim', pa.int32()),
                    ('vector', pa.list_(pa.float32(), dim)),
                ]
            )
            t = self._db.create_table(name, schema=schema)
        self._tables[name] = t
        return t

    def _all_tables(self):
        names = [self._table_name(m) for m in {i['model'] for i in self.meta.indexed_books() if i['model']}]
        out = []
        for n in set(names):
            if n in self._table_names():
                out.append(self._db.open_table(n))
        return out

    def delete_book(self, book_id: int):
        for t in self._all_tables():
            t.delete(f'book_id = {int(book_id)}')

    def insert_chunks(self, book_id: int, items):
        first_model = items[0][2]
        dim = items[0][3]
        t = self._get_table(first_model, dim)
        rows = []
        for c, v, model, dim_ in items:
            vec = v.tolist() if np is not None else list(v)
            rows.append(
                (
                    book_id,
                    c.chunk_no,
                    c.text,
                    ' > '.join(c.chapter_path),
                    c.para_start,
                    c.para_end,
                    c.char_offset,
                    dim_,
                    vec,
                )
            )
        data = [
            {
                'book_id': r[0],
                'chunk_no': r[1],
                'text': r[2],
                'chapter_path': r[3],
                'para_start': r[4],
                'para_end': r[5],
                'char_offset': r[6],
                'dim': r[7],
                'vector': r[8],
            }
            for r in rows
        ]
        t.add(data)

    def commit(self):
        pass  # lancedb commits per add()

    def book_chunks_text(self, book_id: int):
        out = []
        for t in self._all_tables():
            try:
                df = t.search().where(f'book_id = {int(book_id)}').limit(10_000_000).to_pandas()
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            df = df.sort_values('chunk_no')
            out.extend(str(x) for x in df['text'])
        return out

    def search(self, query_vec, limit: int, min_score: float) -> list[SearchResult]:
        qv = l2_normalize(query_vec)
        vec = qv.tolist() if np is not None else list(qv)
        dim = len(vec)
        candidates = []
        for t in self._all_tables():
            try:
                df = t.search(vec).limit(limit * 3).to_pandas()
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            candidates.append(df)
        if not candidates:
            return []
        import pandas as pd

        df = pd.concat(candidates, ignore_index=True)
        fmt_map = {i['id']: i['fmt'] for i in self.meta.indexed_books()}
        out = []
        for _, r in df.iterrows():
            s = float(r['_distance'])
            # lancedb returns L2 distance by default; monotonic cosine-like score for unit vectors
            score = 1.0 / (1.0 + s)
            if score < min_score:
                continue
            out.append(
                SearchResult(
                    book_id=int(r['book_id']),
                    fmt=fmt_map.get(int(r['book_id']), ''),
                    chunk_no=int(r['chunk_no']),
                    text=str(r['text']),
                    chapter_path=[p for p in str(r['chapter_path']).split(' > ') if p],
                    para_start=int(r['para_start']),
                    para_end=int(r['para_end']),
                    char_offset=int(r['char_offset']),
                    score=score,
                )
            )
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]


class VectorStore:
    """Facade combining MetaStore + a vector backend."""

    def __init__(self, db_path: str, backend: str = 'auto'):
        self.db_path = db_path
        self.meta = MetaStore(db_path)
        self.backend_name = self._select_backend(backend)
        if self.backend_name == 'lancedb':
            self.backend = LanceVectorBackend(self.meta)
        else:
            self.backend = SqliteVectorBackend(self.meta)
        self._pending: list[tuple[int, object]] = []  # (book_id, [(chunk, vec, model, dim)])

    def _select_backend(self, want: str) -> str:
        if want in ('sqlite', 'lancedb'):
            if want == 'lancedb':
                try:
                    import lancedb  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "vector backend 'lancedb' selected but the 'lancedb' package is not installed. "
                        "Install it (pip install lancedb) or choose 'sqlite' in settings."
                    )
            return want
        if want == 'auto':
            try:
                import lancedb  # noqa: F401

                return 'lancedb'
            except ImportError:
                return 'sqlite'
        raise ValueError(f'unknown vector backend: {want!r}')

    # -- pass-throughs -------------------------------------------------------------

    def close(self):
        self.meta.close()

    def get_meta(self, key, default=None):
        return self.meta.get_meta(key, default)

    def set_meta(self, key, value):
        self.meta.set_meta(key, value)

    def add_dirty(self, book_id, fmt, reason='added'):
        self.meta.add_dirty(book_id, fmt, reason)

    def dirty_book_ids(self):
        return self.meta.dirty_book_ids()

    def remove_dirty(self, book_id):
        self.meta.remove_dirty(book_id)

    def book_is_indexed(self, book_id):
        return self.meta.book_is_indexed(book_id)

    def indexed_books(self):
        return self.meta.indexed_books()

    # -- indexing --------------------------------------------------------------------

    def clear_book(self, book_id: int):
        self.backend.delete_book(book_id)
        self.meta.clear_book(book_id)

    def insert_chunk(self, book_id, chunk, text, chapter_path, para_start, para_end, char_offset, model, dim, vector):
        # normalize the Chunk-like object's fields (indexer passes a chunk with .text etc.)
        c = chunk
        self._pending.append((book_id, (c, vector, model, dim)))

    def commit(self, book_id: int | None = None):
        """Flush pending chunks. book_id filters when given."""
        if not self._pending:
            return
        by_book: dict[int, list] = {}
        rest = []
        for bid, item in self._pending:
            if book_id is None or bid == book_id:
                by_book.setdefault(bid, []).append(item)
            else:
                rest.append((bid, item))
        self._pending = rest
        for bid, items in by_book.items():
            self.backend.insert_chunks(bid, items)
        self.backend.commit()

    def upsert_book(self, book_id, fmt, n_chunks, model, dim):
        self.meta.upsert_book(book_id, fmt, n_chunks, model, dim)

    # -- search ------------------------------------------------------------------------

    def search(self, query_vec, limit: int = 20, min_score: float = 0.0) -> list[SearchResult]:
        return self.backend.search(query_vec, limit, min_score)

    def book_chunks_text(self, book_id: int):
        return self.backend.book_chunks_text(book_id)
