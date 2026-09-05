'''Vector store: SQLite metadata + pluggable vector backend.

Public API (used by indexer/dialog/attributes):
  VectorStore(db_path, backend='auto') -> facade with:
    add_dirty / dirty_book_ids / remove_dirty
    book_is_indexed / indexed_books / clear_book / upsert_book
    insert_chunk / commit
    search(query_vec, limit, min_score, model=None) -> list[SearchResult]
    book_chunks_text(book_id) -> list[str]
    get_meta / set_meta / delete_meta / meta_keys(prefix)
    set_attrs / get_attrs / clear_attrs / attr_book_ids
    cleanup_stale_models()
    close

Backends (both keep one chunk table per embedding model):
  'sqlite'  - vectors in a local SQLite file (always available; numpy speeds it up)
  'lancedb' - vectors in LanceDB (optional dependency, lazy import)
  'auto'    - lancedb if importable, else sqlite
'''

from __future__ import annotations

import heapq
import json
import os
import re
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


def model_table_name(model: str) -> str:
    """Chunk-table name for an embedding model (shared by both backends)."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', model or '').strip('_').lower() or 'model'
    return f'chunks_{slug}'[:80]


# Memory budget for the batched sqlite search: half of the available RAM, with a floor.
SEARCH_MIN_BUDGET = 10 * 1024 * 1024


def available_ram_bytes():
    """Best-effort available RAM in bytes; None when it cannot be determined."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == 'nt':
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


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

def chunks_table_sql(name: str) -> str:
    """DDL for one per-model chunk table (identifier comes from model_table_name)."""
    return f'''
CREATE TABLE IF NOT EXISTS {name}(
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
CREATE INDEX IF NOT EXISTS idx_{name}_book ON {name}(book_id);
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

    def delete_meta(self, key: str):
        with self._lock:
            self.conn.execute('DELETE FROM meta WHERE key=?', (key,))
            self.conn.commit()

    def meta_keys(self, prefix: str = ''):
        with self._lock:
            if prefix:
                rows = self.conn.execute('SELECT key FROM meta WHERE key LIKE ?', (prefix + '%',)).fetchall()
            else:
                rows = self.conn.execute('SELECT key FROM meta').fetchall()
        return [r[0] for r in rows]

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

    # -- attributes ---------------------------------------------------------------

    def set_attrs(self, book_id: int, values: dict):
        with self._lock:
            self.conn.execute(
                'INSERT INTO attrs_raw(book_id, json, fields, updated_at) VALUES(?,?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET json=excluded.json, fields=excluded.fields, updated_at=excluded.updated_at',
                (book_id, json.dumps(values), ','.join(values.keys()), time.time()),
            )
            self.conn.commit()

    def get_attrs(self, book_id: int):
        with self._lock:
            row = self.conn.execute('SELECT json FROM attrs_raw WHERE book_id=?', (book_id,)).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row[0] or '{}')
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def clear_attrs(self, book_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM attrs_raw WHERE book_id=?', (book_id,))
            self.conn.commit()

    def attr_book_ids(self):
        with self._lock:
            rows = self.conn.execute('SELECT book_id FROM attrs_raw').fetchall()
        return [r[0] for r in rows]


class SqliteVectorBackend:
    """Vectors + chunk text in the same SQLite file, one table per embedding model."""

    name = 'sqlite'

    def __init__(self, meta: MetaStore):
        self.meta = meta
        self.conn = meta.conn  # share connection/lock
        self._migrate_legacy_chunks()

    # -- table management ------------------------------------------------------

    def _chunk_tables_locked(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return sorted(r[0] for r in rows if r[0].startswith('chunks_'))

    def _chunk_tables(self) -> list[str]:
        with self.meta._lock:
            return self._chunk_tables_locked()

    def _table_exists(self, name: str) -> bool:
        with self.meta._lock:
            row = self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        return row is not None

    def _ensure_table(self, model: str) -> str:
        name = model_table_name(model)
        with self.meta._lock:
            self.conn.executescript(chunks_table_sql(name))
        return name

    def _migrate_legacy_chunks(self):
        """Move rows from the pre-per-model single `chunks` table into per-model tables."""
        if not self._table_exists('chunks'):
            return
        cols = 'book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector'
        with self.meta._lock:
            models = [r[0] for r in self.conn.execute('SELECT DISTINCT model FROM chunks')]
            for m in models:
                name = model_table_name(m or '')
                self.conn.executescript(chunks_table_sql(name))
                if m is None:
                    self.conn.execute(f'INSERT INTO {name}({cols}) SELECT {cols} FROM chunks WHERE model IS NULL')
                else:
                    self.conn.execute(f'INSERT INTO {name}({cols}) SELECT {cols} FROM chunks WHERE model=?', (m,))
            self.conn.execute('DROP TABLE chunks')
            self.conn.commit()

    def _table_model(self, name: str):
        with self.meta._lock:
            row = self.conn.execute(f'SELECT model FROM {name} LIMIT 1').fetchone()
        return row[0] if row else None

    def drop_stale_models(self) -> int:
        """Drop chunk tables whose model no longer has any indexed book (e.g. after a model switch)."""
        in_use = {b['model'] for b in self.meta.indexed_books() if b['model']}
        dropped = 0
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                row = self.conn.execute(f'SELECT model FROM {t} LIMIT 1').fetchone()
                if (row[0] if row else None) not in in_use:
                    self.conn.execute(f'DROP TABLE {t}')
                    dropped += 1
            if dropped:
                self.conn.commit()
        return dropped

    # -- writes ------------------------------------------------------------------

    def delete_book(self, book_id: int):
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                self.conn.execute(f'DELETE FROM {t} WHERE book_id=?', (book_id,))

    def insert_chunks(self, book_id: int, items):
        """items: list of (chunk, vector, model, dim) already normalized."""
        name = self._ensure_table(items[0][2])
        rows = [
            (book_id, c.chunk_no, c.text, ' > '.join(c.chapter_path), c.para_start, c.para_end, c.char_offset, model, dim, vec_to_blob(v))
            for c, v, model, dim in items
        ]
        with self.meta._lock:
            self.conn.executemany(
                f'INSERT INTO {name}(book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector) '
                'VALUES(?,?,?,?,?,?,?,?,?,?)',
                rows,
            )

    def commit(self):
        with self.meta._lock:
            self.conn.commit()

    def book_chunks_text(self, book_id: int):
        out = []
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                rows = self.conn.execute(f'SELECT chunk_no, text FROM {t} WHERE book_id=?', (book_id,)).fetchall()
                out.extend(rows)
        out.sort(key=lambda r: r[0])
        return [r[1] for r in out]

    # -- search --------------------------------------------------------------------

    def _search_budget(self) -> int:
        free = available_ram_bytes()
        if free is None:
            free = SEARCH_MIN_BUDGET * 4  # unknown: assume a modest amount of headroom
        return max(SEARCH_MIN_BUDGET, free // 2)

    def _table_dim(self, name: str):
        with self.meta._lock:
            row = self.conn.execute(f'SELECT dim FROM {name} LIMIT 1').fetchone()
        return row[0] if row else None

    def _avg_text_chars(self, name: str) -> int:
        with self.meta._lock:
            row = self.conn.execute(f'SELECT AVG(LENGTH(text)) FROM (SELECT text FROM {name} LIMIT 200)').fetchone()
        return int(row[0]) if row and row[0] is not None else 1024

    def _score_rows(self, rows, qv) -> list[float]:
        """Dot products of each row's vector (last column) with the query vector."""
        if np is not None:
            m = np.empty((len(rows), len(qv)), dtype='<f4')
            for i, r in enumerate(rows):
                m[i] = np.frombuffer(r[-1], dtype='<f4')
            return list(m @ np.asarray(qv, dtype='<f4'))
        import array

        qa = array.array('f')
        qa.frombytes(vec_to_blob(qv))
        out = []
        for r in rows:
            va = array.array('f')
            va.frombytes(r[-1])
            s = 0.0
            for a, b in zip(qa, va):
                s += a * b
            out.append(s)
        return out

    def search(self, query_vec, limit: int, min_score: float, model: str | None = None) -> list[SearchResult]:
        """Top-k by cosine similarity.

        Reads vectors in RAM-budgeted batches (half the available free RAM, with a
        SEARCH_MIN_BUDGET floor) and keeps only a top-`limit` heap, so memory stays
        bounded no matter how many chunks are stored.
        """
        qv = l2_normalize(query_vec)
        dim = qv.shape[0] if np is not None else len(qv)
        tables = [model_table_name(model)] if model is not None else self._chunk_tables()
        budget = self._search_budget()
        top: list[tuple[float, int, SearchResult]] = []  # min-heap of (score, seq, result)
        seq = 0
        for t in tables:
            if not self._table_exists(t) or self._table_dim(t) != dim:
                continue
            row_bytes = dim * 4 + self._avg_text_chars(t) + 64
            batch = max(1, budget // max(1, row_bytes))
            last_id = 0
            while True:
                with self.meta._lock:
                    rows = self.conn.execute(
                        f'SELECT c.id, c.book_id, c.chunk_no, c.text, c.chapter_path, c.para_start, c.para_end, '
                        f'c.char_offset, b.fmt, c.vector FROM {t} c LEFT JOIN books b ON b.id=c.book_id '
                        'WHERE c.id>? ORDER BY c.id LIMIT ?',
                        (last_id, batch),
                    ).fetchall()
                if not rows:
                    break
                last_id = rows[-1][0]
                for r, s in zip(rows, self._score_rows(rows, qv)):
                    if s < min_score:
                        continue
                    heapq.heappush(
                        top,
                        (
                            s,
                            seq,
                            SearchResult(
                                book_id=r[1],
                                fmt=r[8] or '',
                                chunk_no=r[2],
                                text=r[3],
                                chapter_path=[p for p in r[4].split(' > ') if p],
                                para_start=r[5],
                                para_end=r[6],
                                char_offset=r[7],
                                score=s,
                            ),
                        ),
                    )
                    seq += 1
                    if len(top) > limit:
                        heapq.heappop(top)
        top.sort(key=lambda e: (-e[0], e[1]))
        return [r for _, _, r in top]


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
        return model_table_name(model)

    def _open_table(self, model: str):
        name = self._table_name(model)
        if name not in self._table_names():
            return None
        t = self._db.open_table(name)
        self._tables[name] = t
        return t

    def drop_stale_models(self) -> int:
        in_use = {i['model'] for i in self.meta.indexed_books() if i['model']}
        keep = {self._table_name(m) for m in in_use}
        dropped = 0
        for name in self._table_names():
            if name not in keep:
                try:
                    self._db.drop_table(name)
                except Exception:
                    continue
                self._tables.pop(name, None)
                dropped += 1
        return dropped

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

    def search(self, query_vec, limit: int, min_score: float, model: str | None = None) -> list[SearchResult]:
        qv = l2_normalize(query_vec)
        vec = qv.tolist() if np is not None else list(qv)
        dim = len(vec)
        candidates = []
        tables = [self._open_table(model)] if model is not None else self._all_tables()
        tables = [t for t in tables if t is not None]
        for t in tables:
            try:
                df = t.search(vec).metric('cosine').limit(limit * 3).to_pandas()
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
            # cosine distance = 1 - similarity; stored vectors are normalized, so this is the
            # same cosine-similarity scale as the sqlite backend (dot product of unit vectors)
            score = 1.0 - s
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

    def delete_meta(self, key):
        self.meta.delete_meta(key)

    def meta_keys(self, prefix=''):
        return self.meta.meta_keys(prefix)

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

    def set_attrs(self, book_id, values):
        self.meta.set_attrs(book_id, values)

    def get_attrs(self, book_id):
        return self.meta.get_attrs(book_id)

    def clear_attrs(self, book_id):
        self.meta.clear_attrs(book_id)

    def attr_book_ids(self):
        return self.meta.attr_book_ids()

    # -- search ------------------------------------------------------------------------

    def search(self, query_vec, limit: int = 20, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        return self.backend.search(query_vec, limit, min_score, model)

    def book_chunks_text(self, book_id: int):
        return self.backend.book_chunks_text(book_id)

    def cleanup_stale_models(self):
        """Drop chunk tables left over from embedding models no book is indexed with."""
        return self.backend.drop_stale_models()
