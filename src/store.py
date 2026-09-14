'''Vector store: SQLite metadata + pluggable vector backend.

Public API (used by indexer/dialog/attributes):
  VectorStore(db_path, backend=None) -> facade with:
     add_dirty / add_dirty_many / dirty_book_ids / remove_dirty
     book_is_indexed / indexed_books / clear_book / upsert_book
     insert_chunk / commit
      search(query_vec, limit, min_score, model=None) -> list[SearchResult]
      search_book(query_vec, book_id, min_score=0.0, model=None) -> list[SearchResult]
      book_chunks_text(book_id) -> list[str]
     get_meta / set_meta / delete_meta / meta_keys(prefix)
      set_attrs / get_attrs / all_attrs / clear_attrs / attr_book_ids / attrs_fields
    set_failed / clear_failed / wipe_failed / failed_book_ids / failed_entries
    cleanup_stale_models(current_model=None)
    close

Backends (both keep one chunk table per embedding model):
  'sqlite'  - vectors in a local SQLite file (always available; numpy speeds it up)
  'lancedb' - vectors in LanceDB (optional dependency, lazy import)

The backend choice is stored PER LIBRARY in meta['vector_backend'] (this file's
MetaStore always exists, for both backends). The `backend` argument is only a
default for libraries that have no data yet; a library with existing data keeps
it where the data lives until the user picks another backend in settings, which
VectorStore.finalize_schema() then carries out as a resumable cross-backend
transfer. Chunk-text compression works the same way: meta['text_codec'] records
the codec the chunks are actually stored with (zstd+dict or zlib), and a switch
confirmed in settings is recorded as a meta['recompress'] marker that
finalize_schema() executes as an in-place, resumable conversion — the marker is
deleted when it finishes. Opening a store whose data cannot be read with the
currently installed packages raises MissingDependencyError.

Schema versioning: the meta tables (books/dirty/attrs_raw/failed plus the
models/formats/file_info registries) are versioned with PRAGMA user_version and
migrated structurally on open. The sqlite chunk tables migrate from the legacy
single `chunks` table to slim per-model tables whose text lives in a compressed
`text_z` BLOB; the codec is recorded once in meta['text_codec'] and used verbatim
by both reads and writes. At user_version 3 the vector blobs become half floats
(2 bytes per component, ~lossless for normalized embeddings); earlier versions
store single precision, and the width is derived from user_version, never stored
per row. v4 moves failed-book records out of meta JSON into the `failed` table;
it runs deferred at the end of the schema stage, because its user_version bump
must follow the v3 blob-format flip (both gate on >= 3). Each migration step
lives in the migrations/ package (one module per version step) and is imported
at its call site below.

LanceDB tables store half floats natively; once a table holds enough rows,
finalize_schema() builds an IVF_HNSW_SQ vector index for it ('index' stage) and
keeps it merged with the appended rows (the state lives in the dataset manifest).
'''

from __future__ import annotations

import base64
import heapq
import json
import operator
import os
import re
import shutil
import sqlite3
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import timedelta

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
    score: float

    @property
    def chapter_label(self) -> str:
        return ' > '.join(self.chapter_path)


class MissingDependencyError(RuntimeError):
    """A library's search data cannot be read because an optional package is missing.

    Raised by VectorStore.__init__ (which then closes its MetaStore). `dep` is the
    package name; `has_data` says whether the library actually holds chunk data,
    which decides how the GUI explains the block (and whether switching backends
    is possible at all)."""

    def __init__(self, dep: str, has_data: bool = False, want_backend: str | None = None):
        self.dep = dep
        self.has_data = has_data
        self.want_backend = want_backend
        super().__init__(f"this library's search data needs the '{dep}' package, which is not installed")


class FinalizeCancelled(Exception):
    """finalize_schema was asked to stop (e.g. the library is being switched away).

    Raised at a stage checkpoint when its cancel event is set. No data is lost:
    every stage records resumable progress in meta before it does work."""


def _module_available(name: str) -> bool:
    """True when `name` can be imported right now (find_spec: no import side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def vec_to_blob(vec, f32: bool = False) -> bytes:
    """Pack a vector as little-endian IEEE floats: binary16 by default (the v3+
    storage format), binary32 when `f32` (pre-v3 rows and legacy fixtures)."""
    if f32:
        if np is not None:
            return np.asarray(vec, dtype='<f4').tobytes()
        n = len(vec)
        return struct.pack('<%df' % n, *vec)
    if np is not None:
        return np.asarray(vec, dtype='<f2').tobytes()
    return _floats_to_half_bytes(vec)


def blob_to_vec(blob: bytes, f16: bool = True):
    """Decode a vector blob to float32 values (half blobs are upcast exactly)."""
    if np is not None:
        arr = np.frombuffer(blob, dtype='<f2' if f16 else '<f4')
        return arr.astype('<f4') if f16 else arr
    if f16:
        return _half_bytes_to_floats(blob)
    import array

    a = array.array('f')
    a.frombytes(blob)
    return list(a)


def _floats_to_half_bytes(vec) -> bytes:
    """Pure-Python float -> IEEE binary16, round-to-nearest-even (byte-identical to
    numpy's '<f2' cast). Converts from the full double precision in one step —
    rounding via an intermediate float32 would disagree at rare boundaries. Only
    used when numpy is unavailable."""
    out = bytearray()
    for x in vec:
        f = struct.unpack('<Q', struct.pack('<d', float(x)))[0]
        sign = ((f >> 63) & 1) << 15
        exp = (f >> 52) & 0x7FF
        mant = f & ((1 << 52) - 1)
        if exp == 0x7FF:  # inf / nan
            out += struct.pack('<H', sign | (0x7E00 if mant else 0x7C00))
            continue
        if exp == 0:  # zero or f64 subnormal: far below half's range, flushes to +-0
            out += struct.pack('<H', sign)
            continue
        e = exp - 1023  # true exponent
        m = mant | (1 << 52)  # 53-bit mantissa with the implicit leading bit
        if e > 15:  # overflow
            out += struct.pack('<H', sign | 0x7C00)
            continue
        if e < -14:  # half subnormal (or underflow to zero): K = m * 2^(e-28)
            k = _round_shift(m, 28 - e)
            if k == 1024:  # rounded up into the smallest normal
                h = sign | (1 << 10)
            else:
                h = sign | k
        else:  # normal half: keep the top 10 mantissa bits
            k = _round_shift(m - (1 << 52), 42)
            if k == 1024:  # carry into the exponent
                h = sign | ((e + 16) << 10)
            else:
                h = sign | ((e + 15) << 10) | k
        out += struct.pack('<H', h)
    return bytes(out)


def _round_shift(m: int, s: int) -> int:
    """m >> s with round-to-nearest-even."""
    r = m >> s
    rem = m & ((1 << s) - 1)
    half = 1 << (s - 1)
    if rem > half or (rem == half and r & 1):
        r += 1
    return r


def _half_bits_to_float(h: int) -> float:
    """One IEEE binary16 bit pattern to float (exact)."""
    sign = -1.0 if h & 0x8000 else 1.0
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0x1F:
        v = float('inf') if mant == 0 else float('nan')
    elif exp == 0:
        v = mant * 2.0 ** -24
    else:
        v = (mant + 1024) * 2.0 ** (exp - 25)
    return sign * v


_HALF_LUT = None


def _half_lut():
    """Lazy half->float table for all 65536 patterns (exact: every half value is
    representable in f32, so the array('f') entries are lossless)."""
    global _HALF_LUT
    if _HALF_LUT is None:
        import array

        a = array.array('f')
        a.fromlist([_half_bits_to_float(h) for h in range(65536)])
        _HALF_LUT = a
    return _HALF_LUT


def _half_bytes_to_floats(blob: bytes) -> list[float]:
    """Pure-Python IEEE binary16 -> float (exact). Only used when numpy is unavailable."""
    n = len(blob) // 2
    lut = _half_lut()
    return [lut[h] for h in struct.unpack('<%dH' % n, blob)]


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


def normalize_model(model: str) -> str:
    """Canonical embedding-model name (table names, models registry, keep-sets).

    Strips a quantization suffix (after ':'), a vendor prefix (before the first
    '/'), and a trailing '-GGUF' marker, then lowercases and underscores the rest.
    """
    s = (model or '').strip()
    if ':' in s:
        s = s.split(':', 1)[0]
    if '/' in s:
        s = s.split('/', 1)[1]
    s = re.sub(r'-gguf$', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_') or 'model'


def model_table_name(model: str) -> str:
    """Chunk-table name for a (normalized) embedding model (shared by both backends)."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', model or '').strip('_').lower() or 'model'
    return f'chunks_{slug}'[:80]


# Memory budget for the batched sqlite search: a quarter of the available RAM, with a floor.
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


# -- text compression -----------------------------------------------------------

TEXT_CODEC_KEY = 'text_codec'
ZSTD_LEVEL = 3
DEFAULT_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'default_compression_dict.bin')

# Per-library choices and in-flight migration progress, all in the meta table:
BACKEND_KEY = 'vector_backend'  # 'sqlite' | 'lancedb' — where this library's chunks should live
MIGRATE_KEY = 'backend_migrate'  # JSON progress of an in-flight cross-backend transfer
RECOMPRESS_KEY = 'recompress'  # JSON progress of a pending/in-flight sqlite codec conversion (written when the user confirms a compression switch)
VEC_MIGRATE_KEY = 'vec_migrate'  # JSON progress of an in-flight f32->f16 vector conversion (migrations.v3; deleted with the final user_version flip)


def _set_hidden(path):
    if os.name != 'nt':
        return
    import ctypes

    try:
        k32 = ctypes.windll.kernel32
        attr = k32.GetFileAttributesW(path)
        if attr != -1:
            k32.SetFileAttributesW(path, attr | 0x2)
    except Exception:
        pass


def lancedb_dir_for(db_path):
    """The library's hidden LanceDB directory next to the SQLite file.

    The dot prefix hides it on unix; on Windows FILE_ATTRIBUTE_HIDDEN is set as
    well."""
    base = os.path.splitext(os.path.basename(db_path))[0]
    d = os.path.join(os.path.dirname(db_path), '.' + base + '.lancedb')
    _set_hidden(d)
    return d


def _load_default_dict() -> bytes:
    # Installed plugins load from the ZIP via calibre's custom loader (virtual
    # __file__), so filesystem lookups never work there; get_resources is
    # injected by that loader and reads the zip. Running from source uses the file.
    g = globals().get('get_resources')
    if callable(g):
        data = g('assets/default_compression_dict.bin')
        if data is not None:
            return data
    with open(DEFAULT_DICT_PATH, 'rb') as f:
        return f.read()


def build_codec_spec(zstandard_ok: bool) -> dict:
    """The codec recorded in meta['text_codec']: zstd+dict when possible, zlib fallback."""
    if zstandard_ok:
        return {'name': 'zstd', 'dictionary': base64.b64encode(_load_default_dict()).decode('ascii')}
    return {'name': 'zlib'}


def codec_spec_json(name: str) -> str:
    """The meta['text_codec'] JSON for a codec name ('zstd' | 'zlib')."""
    return json.dumps(build_codec_spec(name == 'zstd'))


def recompress_marker(target: str) -> str:
    """A fresh meta['recompress'] marker asking the sqlite backend to convert to `target`."""
    return json.dumps({'target': target, 'done': [], 'cur_table': None, 'last_id': 0})


def write_codec_choice(meta: 'MetaStore', codec: str):
    """Record a confirmed compression switch in `meta`.

    With sqlite chunk data present this is a pending in-place conversion (the
    recompress marker, executed by finalize_schema); without it the codec spec
    itself is updated so data that will arrive uses the chosen codec."""
    with meta._lock:
        rows = meta.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'chunks_%'").fetchall()
    if any(not n[0].endswith('__new') for n in rows):
        meta.set_meta(RECOMPRESS_KEY, recompress_marker(codec))
    else:
        meta.set_meta(TEXT_CODEC_KEY, codec_spec_json(codec))


class TextCodec:
    """Compress/decompress chunk text per the codec recorded in meta['text_codec'].

    Write and read both use exactly this saved codec (no sniffing, no fallback):
    every row of a v2 chunk table was written with the same one.
    """

    def __init__(self, spec: dict):
        self.name = spec.get('name')
        if self.name == 'zstd':
            import zstandard  # guaranteed present by ensure_codec_setup

            d = zstandard.ZstdCompressionDict(base64.b64decode(spec['dictionary']))
            self._cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, dict_data=d)
            self._dctx = zstandard.ZstdDecompressor(dict_data=d)
        elif self.name == 'zlib':
            self._cctx = None
        else:
            raise ValueError(f'unknown text codec: {self.name!r}')

    def compress(self, text: str) -> bytes:
        data = text.encode('utf-8')
        if self.name == 'zstd':
            return self._cctx.compress(data)
        return zlib.compress(data, 6)

    def decompress(self, blob: bytes) -> str:
        if self.name == 'zstd':
            return self._dctx.decompress(blob).decode('utf-8')
        return zlib.decompress(blob).decode('utf-8')


def ensure_codec_setup(meta: 'MetaStore'):
    """Make sure the recorded text codec is usable in THIS session (once per open).

    Runs at migration / new-DB init and on every open — never on the write path.
    The stored codec (meta[TEXT_CODEC_KEY]) is the source of truth; it is kept as
    long as its package is installed. A missing or unusable record is healed to a
    usable codec: zstd when zstandard is available, zlib otherwise. Healing is only
    reachable for dataless libraries — a library whose chunks are stored with zstd
    but lacks the package is blocked in VectorStore.__init__ before this runs.
    """
    spec = None
    try:
        spec = json.loads(meta.get_meta(TEXT_CODEC_KEY))
    except (TypeError, ValueError):
        spec = None
    if spec is not None and spec.get('name') == 'zlib':
        return
    if spec is not None and spec.get('name') == 'zstd' and _module_available('zstandard'):
        return
    meta.set_meta(TEXT_CODEC_KEY, codec_spec_json('zstd' if _module_available('zstandard') else 'zlib'))


# -- DDL ------------------------------------------------------------------------

# Current schema version: brand-new DBs are created at this version, existing DBs
# reach it through the migrations/ steps (one self-contained file per version).
# user_version >= 3 also marks the vector blob format: half floats (2 bytes per
# component) instead of single precision; v2 and earlier store f32. v4 moves
# failed-book records from meta JSON into the `failed` table.
SCHEMA_VERSION = 4

META_SCHEMA = '''
CREATE TABLE IF NOT EXISTS books(
    id INTEGER PRIMARY KEY,
    fmt_id INTEGER NOT NULL,
    indexed_at INTEGER,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    model_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dirty(
    book_id INTEGER PRIMARY KEY,
    reason TEXT NOT NULL DEFAULT 'added',
    added_at INTEGER
);
CREATE TABLE IF NOT EXISTS attrs_raw(
    book_id INTEGER PRIMARY KEY,
    json TEXT NOT NULL DEFAULT '{}',
    fields TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS failed(
    book_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    error TEXT NOT NULL,
    PRIMARY KEY (book_id, kind)
);
CREATE TABLE IF NOT EXISTS models(
    id INTEGER PRIMARY KEY,
    model TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS formats(
    id INTEGER PRIMARY KEY,
    fmt TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS file_info(
    book_id INTEGER PRIMARY KEY,
    fmt_id INTEGER NOT NULL,
    size INTEGER NOT NULL,
    mtime_s INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
'''


def chunks_table_sql(name: str) -> str:
    """DDL for one per-model chunk table (v2 slim shape; identifier from model_table_name)."""
    return f'''
CREATE TABLE IF NOT EXISTS {name}(
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    chunk_no INTEGER NOT NULL,
    text_z BLOB NOT NULL,
    chapter_path TEXT NOT NULL DEFAULT '',
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
            pre_existing = self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone() is not None
            self.conn.executescript(META_SCHEMA)
            if not pre_existing:
                # brand-new file, created at the current schema version. In WAL mode the
                # auto_vacuum setting only takes effect through a VACUUM (instant here,
                # the tables are empty), so run it to arm incremental vacuum from day one.
                self.conn.execute('PRAGMA auto_vacuum=INCREMENTAL')
                self.conn.execute('VACUUM')
                self.conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
            else:
                from .migrations.v1 import upgrade

                upgrade(self)
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    def wal_checkpoint_truncate(self):
        """Drain the WAL into the main DB file and truncate it. Caller holds self._lock."""
        self.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')

    # -- models/formats registries (shared by the write path and migrations/meta.py) --

    def _upsert_model(self, model: str) -> int:
        self.conn.execute('INSERT INTO models(model) VALUES(?) ON CONFLICT(model) DO NOTHING', (model,))
        return self.conn.execute('SELECT id FROM models WHERE model=?', (model,)).fetchone()[0]

    def _upsert_format(self, fmt: str) -> int:
        self.conn.execute('INSERT INTO formats(fmt) VALUES(?) ON CONFLICT(fmt) DO NOTHING', (fmt,))
        return self.conn.execute('SELECT id FROM formats WHERE fmt=?', (fmt,)).fetchone()[0]

    def _model_id(self, model: str) -> int:
        row = self.conn.execute('SELECT id FROM models WHERE model=?', (model,)).fetchone()
        return row[0] if row else self._upsert_model(model)

    def _fmt_id(self, fmt: str) -> int:
        row = self.conn.execute('SELECT id FROM formats WHERE fmt=?', (fmt,)).fetchone()
        return row[0] if row else self._upsert_format(fmt)

    def _meta_keys_locked(self, prefix: str) -> list[str]:
        if prefix:
            rows = self.conn.execute('SELECT key FROM meta WHERE key LIKE ?', (prefix + '%',)).fetchall()
        else:
            rows = self.conn.execute('SELECT key FROM meta').fetchall()
        return [r[0] for r in rows]

    def _meta_value_locked(self, key: str):
        row = self.conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

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
            return self._meta_keys_locked(prefix)

    # -- dirty queue -------------------------------------------------------------

    def add_dirty(self, book_id: int, reason: str = 'added'):
        with self._lock:
            self.conn.execute(
                'INSERT INTO dirty(book_id, reason, added_at) VALUES(?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET reason=excluded.reason, added_at=excluded.added_at',
                (book_id, reason, int(time.time())),
            )
            self.conn.commit()

    def add_dirty_many(self, book_ids, reason: str = 'added'):
        """Queue many books in a single commit (per-book add_dirty fsyncs do not scale)."""
        if not book_ids:
            return
        ts = int(time.time())
        with self._lock:
            self.conn.executemany(
                'INSERT INTO dirty(book_id, reason, added_at) VALUES(?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET reason=excluded.reason, added_at=excluded.added_at',
                [(b, reason, ts) for b in book_ids],
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
            rows = self.conn.execute(
                'SELECT b.id, f.fmt, b.fmt_id, b.n_chunks, m.model, b.model_id, b.indexed_at '
                'FROM books b LEFT JOIN formats f ON f.id=b.fmt_id LEFT JOIN models m ON m.id=b.model_id'
            ).fetchall()
        return [
            {
                'id': r[0],
                'fmt': r[1] or '',
                'fmt_id': r[2],
                'n_chunks': r[3],
                'model': r[4] or '',
                'model_id': r[5],
                'indexed_at': r[6],
            }
            for r in rows
        ]

    def upsert_book(self, book_id: int, fmt: str, n_chunks: int, model: str):
        with self._lock:
            norm = normalize_model(model)
            self.conn.execute(
                'INSERT INTO books(id, fmt_id, indexed_at, n_chunks, model_id) VALUES(?,?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET fmt_id=excluded.fmt_id, indexed_at=excluded.indexed_at, '
                'n_chunks=excluded.n_chunks, model_id=excluded.model_id',
                (book_id, self._fmt_id(fmt), int(time.time()), n_chunks, self._model_id(norm)),
            )
            self.conn.commit()

    # -- file change-detection registry ---------------------------------------------

    def get_file_info(self, book_id: int):
        with self._lock:
            row = self.conn.execute('SELECT fmt_id, size, mtime_s FROM file_info WHERE book_id=?', (book_id,)).fetchone()
            if row is None:
                return None
            frow = self.conn.execute('SELECT fmt FROM formats WHERE id=?', (row[0],)).fetchone()
        return {'fmt': frow[0] if frow else '', 'size': row[1], 'mtime_s': row[2]}

    def set_file_info(self, book_id: int, fmt: str, size: int, mtime_s: int):
        with self._lock:
            self.conn.execute(
                'INSERT INTO file_info(book_id, fmt_id, size, mtime_s) VALUES(?,?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET fmt_id=excluded.fmt_id, size=excluded.size, mtime_s=excluded.mtime_s',
                (book_id, self._fmt_id(fmt), int(size), int(mtime_s)),
            )
            self.conn.commit()

    def clear_file_info(self, book_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM file_info WHERE book_id=?', (book_id,))
            self.conn.commit()

    def file_info_book_ids(self):
        with self._lock:
            rows = self.conn.execute('SELECT book_id FROM file_info').fetchall()
        return [r[0] for r in rows]

    # -- attributes ---------------------------------------------------------------

    def set_attrs(self, book_id: int, values: dict):
        with self._lock:
            self.conn.execute(
                'INSERT INTO attrs_raw(book_id, json, fields) VALUES(?,?,?) '
                'ON CONFLICT(book_id) DO UPDATE SET json=excluded.json, fields=excluded.fields',
                (book_id, json.dumps(values), ','.join(values.keys())),
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

    def all_attrs(self) -> dict:
        """{book_id: values dict} for every book with stored attributes (one query)."""
        with self._lock:
            rows = self.conn.execute('SELECT book_id, json FROM attrs_raw').fetchall()
        out = {}
        for bid, raw in rows:
            try:
                data = json.loads(raw or '{}')
            except Exception:
                continue
            if isinstance(data, dict):
                out[bid] = data
        return out

    def clear_attrs(self, book_id: int):
        with self._lock:
            self.conn.execute('DELETE FROM attrs_raw WHERE book_id=?', (book_id,))
            self.conn.commit()

    def attr_book_ids(self):
        with self._lock:
            rows = self.conn.execute('SELECT book_id FROM attrs_raw').fetchall()
        return [r[0] for r in rows]

    def attrs_fields(self):
        """{book_id: frozenset(field names)} for every book with stored attributes.

        One query instead of a get_attrs round-trip per book; the fields column is
        maintained as ','.join(keys) by set_attrs, so it matches the JSON keys."""
        with self._lock:
            rows = self.conn.execute('SELECT book_id, fields FROM attrs_raw').fetchall()
        out = {}
        for bid, fields in rows:
            out[bid] = frozenset(fields.split(',')) if fields else frozenset()
        return out

    # -- failed books ---------------------------------------------------------------
    #
    # Books that failed indexing (kind 'index') or attribute extraction (kind
    # 'attr'), until retried. One row per book and kind: a book can be in both
    # states at once (indexed but attr-extraction failed, then re-index fails).

    def set_failed(self, book_id: int, kind: str, error: str):
        with self._lock:
            self.conn.execute(
                'INSERT INTO failed(book_id, kind, error) VALUES(?,?,?) '
                'ON CONFLICT(book_id, kind) DO UPDATE SET error=excluded.error',
                (book_id, kind, error),
            )
            self.conn.commit()

    def clear_failed(self, book_id: int, kind: str | None = None):
        """Drop one book's failure record(s); `kind=None` drops both kinds."""
        with self._lock:
            if kind is None:
                self.conn.execute('DELETE FROM failed WHERE book_id=?', (book_id,))
            else:
                self.conn.execute('DELETE FROM failed WHERE book_id=? AND kind=?', (book_id, kind))
            self.conn.commit()

    def wipe_failed(self, kind: str):
        """Drop every failure record of one kind (the 'retry all' actions)."""
        with self._lock:
            self.conn.execute('DELETE FROM failed WHERE kind=?', (kind,))
            self.conn.commit()

    def failed_book_ids(self, kind: str | None = None) -> list[int]:
        with self._lock:
            if kind is None:
                rows = self.conn.execute('SELECT book_id FROM failed ORDER BY book_id').fetchall()
            else:
                rows = self.conn.execute('SELECT book_id FROM failed WHERE kind=? ORDER BY book_id', (kind,)).fetchall()
        return [r[0] for r in rows]

    def failed_entries(self, kind: str | None = None) -> list[dict]:
        """{'book_id', 'kind', 'error'} rows, ordered by book_id then kind."""
        with self._lock:
            if kind is None:
                rows = self.conn.execute('SELECT book_id, kind, error FROM failed ORDER BY book_id, kind').fetchall()
            else:
                rows = self.conn.execute(
                    'SELECT book_id, kind, error FROM failed WHERE kind=? ORDER BY book_id', (kind,)
                ).fetchall()
        return [{'book_id': r[0], 'kind': r[1], 'error': r[2]} for r in rows]


class SqliteVectorBackend:
    """Vectors + chunk text in the same SQLite file, one table per embedding model."""

    name = 'sqlite'

    def __init__(self, meta: MetaStore):
        self.meta = meta
        self.conn = meta.conn  # share connection/lock
        ensure_codec_setup(meta)
        self._codec = TextCodec(json.loads(meta.get_meta(TEXT_CODEC_KEY)))

    # -- table management ------------------------------------------------------

    def _chunk_tables_locked(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return sorted(r[0] for r in rows if r[0].startswith('chunks_') and not r[0].endswith('__new'))

    def _chunk_tables(self) -> list[str]:
        with self.meta._lock:
            return self._chunk_tables_locked()

    def _table_exists(self, name: str) -> bool:
        with self.meta._lock:
            row = self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        return row is not None

    def _table_for(self, model: str) -> str:
        return model_table_name(normalize_model(model))

    def _ensure_table(self, model: str) -> str:
        name = self._table_for(model)
        with self.meta._lock:
            if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                self.conn.executescript(chunks_table_sql(name))
                self.conn.commit()
        return name

    # -- migration -----------------------------------------------------------------

    def pending_work(self) -> bool:
        from .migrations.v2 import pending as v2_pending
        from .migrations.v3 import pending as v3_pending

        return v2_pending(self) or v3_pending(self)

    def finalize(self, say=None) -> bool:
        """Run the pending chunk migrations (v2 slim shape, then v3 half-float)."""
        from .migrations.v2 import pending as v2_pending
        from .migrations.v2 import upgrade as v2_upgrade
        from .migrations.v3 import pending as v3_pending
        from .migrations.v3 import upgrade as v3_upgrade

        did = False
        if v2_pending(self):
            did = v2_upgrade(self) or did
        if v3_pending(self):
            if say is not None:
                say('schema', 'converting vectors to half precision')
            did = v3_upgrade(self, say) or did
        return did

    def drop_stale_models(self, current_model: str | None = None) -> int:
        """Drop chunk tables whose model no longer has any indexed book (e.g. after a model switch).

        Table names are canonical (chunks_<normalized model>), so ownership is derived
        from the name; anything else is stale."""
        in_use = {b['model'] for b in self.meta.indexed_books() if b.get('model')}
        if current_model:
            in_use.add(normalize_model(current_model))
        keep = {model_table_name(m) for m in in_use}
        dropped = 0
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                if t not in keep:
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
        """items: list of (chunk, vector, model); text is compressed per the saved codec."""
        name = self._ensure_table(items[0][2])
        # match the DB's current blob format so a table never mixes widths while a
        # v3 conversion is still pending (new rows are converted with the rest)
        f32 = self._vec_bytes() == 4
        rows = [
            (book_id, c.chunk_no, self._codec.compress(c.text), ' > '.join(c.chapter_path or []), vec_to_blob(v, f32=f32))
            for c, v, _model in items
        ]
        with self.meta._lock:
            self.conn.executemany(
                f'INSERT INTO {name}(book_id, chunk_no, text_z, chapter_path, vector) VALUES(?,?,?,?,?)',
                rows,
            )

    def commit(self):
        with self.meta._lock:
            self.conn.commit()

    def book_chunks_text(self, book_id: int):
        out = []
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                rows = self.conn.execute(f'SELECT chunk_no, text_z FROM {t} WHERE book_id=?', (book_id,)).fetchall()
                out.extend((n, self._codec.decompress(z)) for n, z in rows)
        out.sort(key=lambda r: r[0])
        return [r[1] for r in out]

    # -- search --------------------------------------------------------------------

    def _search_budget(self) -> int:
        free = available_ram_bytes()
        if free is None:
            free = SEARCH_MIN_BUDGET * 8  # unknown: assume a modest amount of headroom
        return max(SEARCH_MIN_BUDGET, free // 4)

    def _vec_bytes(self) -> int:
        """Bytes per vector component in this DB's chunk tables (2 for f16 at
        user_version >= 3, 4 for the legacy f32)."""
        with self.meta._lock:
            uv = self.conn.execute('PRAGMA user_version').fetchone()[0]
        return 2 if uv >= 3 else 4

    def _table_dim(self, name: str):
        with self.meta._lock:
            row = self.conn.execute(f'SELECT LENGTH(vector) FROM {name} LIMIT 1').fetchone()
        if not row or not row[0]:
            return None
        # A table converted mid-v3-migration holds f16 blobs while user_version is
        # still 2, so it reports half its dimension and is skipped by the search's
        # per-table dim check until the migration's final commit flips it.
        return row[0] // self._vec_bytes()

    def _score_rows(self, rows, qv, f16: bool = False) -> list[float]:
        """Dot products of each row's vector (last column) with the query vector.

        `f16` says whether the row blobs are half floats; scores are computed in
        single precision either way (half values upcast to f32 exactly)."""
        if np is not None:
            m = np.empty((len(rows), len(qv)), dtype='<f4')
            for i, r in enumerate(rows):
                m[i] = blob_to_vec(r[-1], f16)
            return list(m @ np.asarray(qv, dtype='<f4'))
        import array

        qa = array.array('f')
        qa.frombytes(vec_to_blob(qv, f32=True))
        # map/operator.mul runs the per-element multiply in C (same left-to-right
        # double accumulation as a Python loop, but without the bytecode overhead)
        mul = operator.mul
        out = []
        for r in rows:
            va = blob_to_vec(r[-1], f16)
            out.append(sum(map(mul, qa, va)))
        return out

    def search(self, query_vec, limit: int, min_score: float, model: str | None = None) -> list[SearchResult]:
        """Top-`limit` books by their best chunk's cosine similarity (one result per book).

        Phase 1 reads vectors in RAM-budgeted batches (a quarter of the available free
        RAM, with a SEARCH_MIN_BUDGET floor) and keeps only each book's best row so far,
        so memory stays bounded no matter how many chunks are stored; phase 2 fetches
        the winning rows by id for their text.
        """
        if limit <= 0:
            return []
        # While a v3 conversion is in flight the tables mix blob widths (a converted
        # f16 table can even masquerade as another model's dimension), so searching
        # them would misread rows; stay out of the way until it finishes.
        if self.meta.get_meta(VEC_MIGRATE_KEY) is not None:
            return []
        qv = l2_normalize(query_vec)
        dim = qv.shape[0] if np is not None else len(qv)
        tables = [self._table_for(model)] if model is not None else self._chunk_tables()
        budget = self._search_budget()
        f16 = self._vec_bytes() == 2
        best: dict[int, tuple[float, str, int]] = {}  # book_id -> (score, table, row id)
        for t in tables:
            if not self._table_exists(t) or self._table_dim(t) != dim:
                continue
            # per row: the vector blob plus one f32 scoring-matrix row, plus overhead
            row_bytes = dim * ((2 if f16 else 4) + 4) + 64
            batch = max(1, budget // max(1, row_bytes))
            last_id = 0
            while True:
                with self.meta._lock:
                    rows = self.conn.execute(
                        f'SELECT id, book_id, vector FROM {t} WHERE id>? ORDER BY id LIMIT ?',
                        (last_id, batch),
                    ).fetchall()
                if not rows:
                    break
                last_id = rows[-1][0]
                for r, s in zip(rows, self._score_rows(rows, qv, f16)):
                    if s < min_score:
                        continue
                    cur = best.get(r[1])
                    if cur is None or s > cur[0]:
                        best[r[1]] = (s, t, r[0])
        top = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:limit]
        by_table: dict[str, list[int]] = {}  # table -> winning row ids
        for _bid, (_s, t, rid) in top:
            by_table.setdefault(t, []).append(rid)
        fmt_map = {i['id']: i['fmt'] for i in self.meta.indexed_books()}
        out: list[SearchResult] = []
        for t, rids in by_table.items():
            for i in range(0, len(rids), 400):  # sqlite binds at most 999 variables per statement
                part = rids[i : i + 400]
                marks = ','.join('?' * len(part))
                with self.meta._lock:
                    rows = self.conn.execute(
                        f'SELECT book_id, chunk_no, text_z, chapter_path FROM {t} WHERE id IN ({marks})',
                        part,
                    ).fetchall()
                for bid, cno, z, path in rows:
                    out.append(
                        SearchResult(
                            book_id=bid,
                            fmt=fmt_map.get(bid, ''),
                            chunk_no=cno,
                            text=self._codec.decompress(z),
                            chapter_path=[p for p in path.split(' > ') if p],
                            score=best[bid][0],
                        )
                    )
        out.sort(key=lambda x: (-x.score, x.book_id))
        return out

    def search_book(self, query_vec, book_id: int, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        """All of one book's chunks scoring >= min_score, best first (no cap)."""
        if self.meta.get_meta(VEC_MIGRATE_KEY) is not None:
            return []
        qv = l2_normalize(query_vec)
        dim = qv.shape[0] if np is not None else len(qv)
        tables = [self._table_for(model)] if model is not None else self._chunk_tables()
        f16 = self._vec_bytes() == 2
        fmt_map = {i['id']: i['fmt'] for i in self.meta.indexed_books()}
        out: list[SearchResult] = []
        for t in tables:
            if not self._table_exists(t) or self._table_dim(t) != dim:
                continue
            with self.meta._lock:
                rows = self.conn.execute(
                    f'SELECT chunk_no, text_z, chapter_path, vector FROM {t} WHERE book_id=?',
                    (book_id,),
                ).fetchall()
            for r, s in zip(rows, self._score_rows(rows, qv, f16)):
                if s < min_score:
                    continue
                out.append(
                    SearchResult(
                        book_id=book_id,
                        fmt=fmt_map.get(book_id, ''),
                        chunk_no=r[0],
                        text=self._codec.decompress(r[1]),
                        chapter_path=[p for p in r[2].split(' > ') if p],
                        score=s,
                    )
                )
        out.sort(key=lambda x: (-x.score, x.chunk_no))
        return out


class LanceVectorBackend:
    """Vectors + chunk text in LanceDB (optional dependency).

    One table per embedding model so dimension changes never mix vectors;
    tables are named from the normalized model name, like the sqlite backend.
    """

    name = 'lancedb'

    def __init__(self, meta: MetaStore):
        import lancedb  # lazy: optional dependency

        self.meta = meta
        db_path = lancedb_dir_for(meta.db_path)
        self._db = lancedb.connect(db_path)
        self._tables: dict[str, object] = {}

    def pending_work(self) -> bool:
        return False  # existing tables are untouched; new tables are created slim on demand

    def finalize(self, say=None) -> bool:
        return False

    # -- ANN index maintenance -------------------------------------------------------
    #
    # Vectors are stored as half floats (the same lossy storage the sqlite backend
    # uses), so an IVF_HNSW_SQ index over them is worth building only once a table
    # has enough rows; below that, a flat scan wins. The index state lives in the
    # LanceDB dataset manifest, so no meta marker is needed: an interrupted build
    # simply leaves "no index" behind and the next open detects it again.

    INDEX_TYPE = 'IvfHnswSq'  # as reported by IndexConfig.index_type
    MERGE_TAIL_ROWS = 1000  # re-merge once this many rows wait unindexed (flat-scan tail cost)
    PARTITION_TARGET_ROWS = 1_048_576  # LanceDB's default IVF partition size

    def _our_index(self, t):
        """The IndexConfig of our vector index on table `t`, or None."""
        try:
            cfgs = list(t.list_indices())
        except Exception:
            return None
        for c in cfgs:
            if getattr(c, 'index_type', None) == self.INDEX_TYPE and getattr(c, 'columns', None) == ['vector']:
                return c
        return None

    def index_pending(self) -> bool:
        """True when any table with rows lacks our index or has a long unindexed tail."""
        for name in sorted(self._table_names()):
            t = self._open_named(name)
            if t is None:
                continue
            try:
                n = t.count_rows()
            except Exception:
                continue
            if not n:
                continue
            cfg = self._our_index(t)
            if cfg is None:
                return True
            if (getattr(cfg, 'num_unindexed_rows', 0) or 0) >= self.MERGE_TAIL_ROWS:
                return True
        return False

    def finalize_index(self, say=None):
        """Build / incrementally update the vector index of every table that needs it.

        A missing index is built (IVF_HNSW_SQ, cosine); a stale one is refreshed with
        optimize(), which also compacts and prunes fragments — followed by an
        aggressive prune so the space of replaced fragments is reclaimed immediately."""
        for name in sorted(self._table_names()):
            t = self._open_named(name)
            if t is None:
                continue
            try:
                n = t.count_rows()
            except Exception:
                continue
            if not n:
                continue
            cfg = self._our_index(t)
            if cfg is None:
                if say is not None:
                    say('index', f'building vector index for {name} ({n:,} rows)')
                from lancedb.index import IvfHnswSq

                t.create_index(
                    'vector',
                    config=IvfHnswSq(
                        distance_type='cosine',
                        num_partitions=max(1, n // self.PARTITION_TARGET_ROWS),
                        ef_construction=150,
                    ),
                )
            elif (getattr(cfg, 'num_unindexed_rows', 0) or 0) >= self.MERGE_TAIL_ROWS:
                if say is not None:
                    say('index', f'updating vector index for {name} ({cfg.num_unindexed_rows:,} new rows)')
                t.optimize()
                t.optimize(cleanup_older_than=timedelta(0))

    def _table_name(self, model: str) -> str:
        return model_table_name(normalize_model(model or ''))

    def _candidate_models(self, current_model: str | None = None):
        """Normalized model names whose tables may hold live data."""
        out = {i['model'] for i in self.meta.indexed_books() if i.get('model')}
        if current_model:
            out.add(normalize_model(current_model))
        return out

    def _open_table(self, model: str):
        return self._open_named(self._table_name(model))

    def _open_named(self, name):
        if name in self._table_names():
            t = self._db.open_table(name)
            self._tables[name] = t
            return t
        return None

    def drop_stale_models(self, current_model: str | None = None) -> int:
        keep = {self._table_name(m) for m in self._candidate_models(current_model)}
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
        if name in self._table_names():
            t = self._db.open_table(name)
            self._tables[name] = t
            return t
        import pyarrow as pa

        schema = pa.schema(
            [
                ('book_id', pa.int64()),
                ('chunk_no', pa.int32()),
                ('text', pa.string()),
                ('chapter_path', pa.string()),
                # half floats: same lossy storage as the sqlite backend's v3 blobs;
                # searches re-rank candidates against the stored values (refine)
                ('vector', pa.list_(pa.float16(), dim)),
            ]
        )
        t = self._db.create_table(name, schema=schema)
        self._tables[name] = t
        return t

    def _all_tables(self):
        names = {self._table_name(m) for m in self._candidate_models()}
        out = []
        for n in sorted(names):
            if n in self._table_names():
                out.append(self._db.open_table(n))
        return out

    def delete_book(self, book_id: int):
        for t in self._all_tables():
            t.delete(f'book_id = {int(book_id)}')

    def insert_chunks(self, book_id: int, items):
        first_model = items[0][2]
        dim = len(items[0][1])
        t = self._get_table(first_model, dim)
        try:
            cols = set(t.schema.names)
        except Exception:
            cols = set()
        legacy = 'para_start' in cols  # pre-v2 table shape: keep its extra columns populated
        data = []
        for c, v, _model in items:
            vec = v.tolist() if np is not None else list(v)
            row = {
                'book_id': book_id,
                'chunk_no': c.chunk_no,
                'text': c.text,
                'chapter_path': ' > '.join(c.chapter_path or []),
                'vector': vec,
            }
            if legacy:
                row.update(para_start=c.para_start or 0, para_end=c.para_end or 0, char_offset=c.char_offset or 0, dim=dim)
            data.append(row)
        t.add(data)

    def commit(self):
        pass  # lancedb commits per add()

    def book_chunks_text(self, book_id: int):
        out = []
        for t in self._all_tables():
            try:
                rows = t.search().where(f'book_id = {int(book_id)}').limit(10_000_000).to_list()
            except Exception:
                continue
            if not rows:
                continue
            rows.sort(key=lambda r: r['chunk_no'])
            out.extend(str(r['text']) for r in rows)
        return out

    SEARCH_PAGE = 500  # rows per ANN page in search()

    def _ann_page(self, t, vec, offset):
        """One page of table `t`'s ANN results (distance order). refine re-scores the
        candidates against the stored vectors, correcting the lossy half-float / SQ index."""
        return t.search(vec).metric('cosine').limit(self.SEARCH_PAGE).offset(offset).refine_factor(1).to_list()

    def search(self, query_vec, limit: int, min_score: float, model: str | None = None) -> list[SearchResult]:
        """Top-`limit` books by their best chunk's cosine similarity (one result per book).

        Pages each table's ANN results and merges them k-way, keeping the best chunk seen
        so far for every book; stops once `limit` books are in hand and no pending page can
        still beat the weakest of them.
        """
        if limit <= 0:
            return []
        qv = l2_normalize(query_vec)
        vec = qv.tolist() if np is not None else list(qv)
        tables = [self._open_table(model)] if model is not None else self._all_tables()
        cursors = []  # each: {'t': table, 'rows': pending page rows, 'off': next offset}
        for t in tables:
            if t is None:
                continue
            try:
                rows = self._ann_page(t, vec, 0)
            except Exception:
                continue
            cursors.append({'t': t, 'rows': list(rows), 'off': len(rows)})
        best: dict[int, SearchResult] = {}
        fmt_map = {i['id']: i['fmt'] for i in self.meta.indexed_books()}
        T = float('-inf')  # lower bound on the weakest book still in the top-`limit`
        while True:
            heads = [c['rows'][0] for c in cursors if c['rows']]
            if not heads:
                break
            # cosine distance = 1 - similarity; stored vectors are normalized, so this is
            # the same cosine-similarity scale as the sqlite backend (dot of unit vectors)
            if len(best) >= limit and max(1.0 - float(h['_distance']) for h in heads) < T:
                break
            ci = max((i for i, c in enumerate(cursors) if c['rows']), key=lambda i: -float(cursors[i]['rows'][0]['_distance']))
            r = cursors[ci]['rows'].pop(0)
            s = 1.0 - float(r['_distance'])
            bid = int(r['book_id'])
            cur = best.get(bid)
            if s >= min_score and (cur is None or s > cur.score):
                best[bid] = SearchResult(
                    book_id=bid,
                    fmt=fmt_map.get(bid, ''),
                    chunk_no=int(r['chunk_no']),
                    text=str(r['text']),
                    chapter_path=[p for p in str(r['chapter_path']).split(' > ') if p],
                    score=s,
                )
                # a new book may have pushed the top-`limit` threshold up; refresh it
                # periodically (a stale-low T only delays the stop below, never wrongs it)
                if cur is None and len(best) >= limit and len(best) % 256 == 0:
                    T = heapq.nlargest(limit, (x.score for x in best.values()))[-1]
            if not cursors[ci]['rows']:
                try:
                    page = self._ann_page(cursors[ci]['t'], vec, cursors[ci]['off'])
                except Exception:
                    page = []
                cursors[ci]['rows'] = list(page)
                cursors[ci]['off'] += len(page)
        return sorted(best.values(), key=lambda x: (-x.score, x.book_id))[:limit]

    def search_book(self, query_vec, book_id: int, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        """All of one book's chunks scoring >= min_score, best first (no cap)."""
        qv = l2_normalize(query_vec)
        tables = [self._open_table(model)] if model is not None else self._all_tables()
        fmt_map = {i['id']: i['fmt'] for i in self.meta.indexed_books()}
        out: list[SearchResult] = []
        for t in tables:
            if t is None:
                continue
            try:
                rows = t.search().where(f'book_id = {int(book_id)}').limit(10_000_000).to_list()
            except Exception:
                continue
            if not rows:
                continue
            # a filter-only scan has no _distance; score the stored (unit) vectors
            # directly, normalizing to guard against half-float norm drift
            if np is not None:
                m = np.empty((len(rows), len(qv)), dtype='<f4')
                for i, r in enumerate(rows):
                    m[i] = np.asarray(r['vector'], dtype='<f4')
                norms = np.linalg.norm(m, axis=1)
                scores = (m @ np.asarray(qv, dtype='<f4')) / np.where(norms == 0, 1.0, norms)
            else:
                qa = [float(x) for x in qv]
                scores = []
                for r in rows:
                    v = [float(x) for x in r['vector']]
                    n = (sum(x * x for x in v)) ** 0.5
                    scores.append(sum(a * b for a, b in zip(qa, v)) / n if n else 0.0)
            for r, s in zip(rows, scores):
                if s < min_score:
                    continue
                out.append(
                    SearchResult(
                        book_id=book_id,
                        fmt=fmt_map.get(book_id, ''),
                        chunk_no=int(r['chunk_no']),
                        text=str(r['text']),
                        chapter_path=[p for p in str(r['chapter_path']).split(' > ') if p],
                        score=float(s),
                    )
                )
        out.sort(key=lambda x: (-x.score, x.chunk_no))
        return out


class _MigrateRow:
    """Minimal chunk shape so backend.insert_chunks() can be reused for migrations."""

    __slots__ = ('chunk_no', 'chapter_path', 'text', 'para_start', 'para_end', 'char_offset')

    def __init__(self, chunk_no, chapter_path, text):
        self.chunk_no = chunk_no
        self.chapter_path = chapter_path
        self.text = text
        self.para_start = 0
        self.para_end = 0
        self.char_offset = 0


class VectorStore:
    """Facade combining MetaStore + a vector backend."""

    def __init__(self, db_path: str, backend: str | None = None):
        self.db_path = db_path
        self.meta = MetaStore(db_path)
        # Per-library resolution: meta[BACKEND_KEY] is the source of truth. The
        # argument (the global default from settings) only applies to libraries that
        # hold no chunk data yet; a library with data keeps it where the data lives
        # until the user picks another backend, which finalize_schema() then performs.
        stored = self.meta.get_meta(BACKEND_KEY)
        if stored not in ('sqlite', 'lancedb'):
            stored = None
        loc = self._data_location()
        if loc is None:
            want = backend if backend in ('sqlite', 'lancedb') else 'sqlite'
        else:
            want = stored if stored is not None else loc
        self.meta.set_meta(BACKEND_KEY, want)
        self.want_backend = want
        self.backend_name = loc if loc is not None else want
        # Pre-flight: what is stored must be readable with the currently installed
        # packages. A pending switch to an unavailable backend does NOT block — the
        # data stays where it is until the migration can run (installing the package,
        # or switching back in settings, unblocks it). For a fresh library
        # backend_name == want_backend, so this covers that case too.
        dep = self._missing_dependency(self.backend_name)
        if dep is not None:
            self._raise_missing(dep)
        rp = self._recompress_progress()
        if rp is not None and rp.get('target') == 'zstd' and not _module_available('zstandard'):
            self._raise_missing('zstandard')
        # No transfer in flight, yet both locations hold data: the extra one is a
        # leftover of an interrupted/abandoned transfer; drop it so its space is
        # reclaimed. (While MIGRATE_KEY is set the other side is the source of a
        # resumable transfer, not garbage.)
        if self.backend_name == self.want_backend and self.meta.get_meta(MIGRATE_KEY) is None:
            other = 'lancedb' if self.backend_name == 'sqlite' else 'sqlite'
            if (other == 'lancedb' and self._lancedb_has_data()) or (other == 'sqlite' and self._sqlite_has_chunks()):
                self._drop_backend_storage(other)
        self._swap_backend(self.backend_name)
        self._pending: list[tuple[int, object]] = []  # (book_id, [(chunk, vec, model)])

    def _wanted_backend(self):
        """The backend meta says this library should use, read live (a settings change
        can land while the store is open; __init__'s copy may be stale)."""
        w = self.meta.get_meta(BACKEND_KEY)
        return w if w in ('sqlite', 'lancedb') else self.want_backend

    def stored_codec(self):
        """The codec this library's chunks are actually stored with ('zstd'/'zlib'),
        or None when there is no sqlite chunk data to speak of."""
        if self.backend_name != 'sqlite' or not self._sqlite_has_chunks():
            return None
        return self._recorded_codec_name()

    def _meta_version(self) -> int:
        with self.meta._lock:
            return self.meta.conn.execute('PRAGMA user_version').fetchone()[0]

    def pending_stages(self) -> list[str]:
        """Ordered migration stages still to run: 'schema', 'backend', 'codec', 'index'.

        The 'codec' stage is pending while a meta[RECOMPRESS_KEY] marker asks for a
        conversion (written when the user confirms a compression switch); it is
        deleted when the conversion finishes. A marker whose target package is
        missing blocks the open in __init__, so here it always means the work can
        actually run. The lancedb 'index' stage is pending while any table with rows
        lacks its vector index or has a long unindexed tail; it needs no marker,
        because the index state lives in the LanceDB dataset manifest. The meta
        schema (v4: failed records out of meta JSON) is pending until user_version
        reaches SCHEMA_VERSION; it runs at the end of the 'schema' stage, after the
        chunk migrations have settled the blob format."""
        want = self._wanted_backend()
        stages = []
        if self.backend.pending_work() or self._meta_version() < SCHEMA_VERSION:
            stages.append('schema')
        # a transfer is pending when the backends differ, or when one was interrupted
        # (MIGRATE_KEY is set from the first inserted row until completion)
        if self.backend_name != want or self.meta.get_meta(MIGRATE_KEY) is not None:
            stages.append('backend')
        if self.backend_name == 'sqlite' and self.meta.get_meta(RECOMPRESS_KEY) is not None:
            stages.append('codec')
        if self.backend_name == 'lancedb' and self.backend.index_pending():
            stages.append('index')
        return stages

    def needs_finalize(self) -> bool:
        """True when a migration ran or is pending and the final VACUUM has not run yet."""
        if self.pending_stages():
            return True
        # fully migrated but the final VACUUM never completed (e.g. interrupted last time):
        # in WAL mode the auto_vacuum setting only persists through a finished VACUUM,
        # so av != 2 on an otherwise-v2 DB means the space was never reclaimed
        with self.meta._lock:
            return self.meta.conn.execute('PRAGMA auto_vacuum').fetchone()[0] != 2

    def finalize_schema(self, progress=None, cancel=None):
        """Run pending migrations (schema -> backend -> codec -> index), then one final VACUUM.

        Idempotent and resumable: every stage records its progress in meta (or, for
        the lancedb index stage, in the dataset manifest itself), so an interrupted
        run continues where it stopped on the next open. `progress` is called as
        progress(stage, detail) with human-readable strings. `cancel` is a threading
        Event: when set, the run stops at the next checkpoint by raising
        FinalizeCancelled (checkpoints are every stage boundary and every say() call,
        i.e. per batch / per book inside the long stages). Meant to be called from a
        worker thread before indexing starts (see gui._start_for_library)."""

        def cancelled():
            if cancel is not None and cancel.is_set():
                raise FinalizeCancelled()

        def say(stage, detail):
            # say doubles as the fine-grained cancel checkpoint; progress errors
            # still never break the migration
            cancelled()
            if progress is not None:
                try:
                    progress(stage, detail)
                except Exception:
                    pass

        cancelled()
        did = False
        if self.backend.pending_work():
            say('schema', 'migrating chunk tables')
            self.backend.finalize(say)
            did = True
        # evaluated live: a legacy DB finishes its chunk migrations (v2/v3) above and
        # lands here in the same pass. The version bump must come after them — v3's
        # pending check and the blob-width derivation both gate on user_version >= 3,
        # so flipping to SCHEMA_VERSION earlier would skip the f16 conversion while
        # search already reads half-float widths.
        cancelled()
        if self._meta_version() < SCHEMA_VERSION:
            from .migrations.v4 import upgrade as v4_upgrade

            say('schema', 'moving failed-book records to their own table')
            v4_upgrade(self.meta)
            did = True
        src, dst = self._pending_transfer()
        if dst is not None and dst != src:
            say('backend', f'moving data to the {dst} backend')
            self._migrate_backend(src, dst, say)
            did = True
        cancelled()
        if self.backend_name == 'sqlite' and self.meta.get_meta(RECOMPRESS_KEY) is not None:
            say('codec', 'recompressing chunk text')
            self._recompress_chunks(say)
            did = True
        # evaluated live: a transfer that just finished lands here in the same pass
        cancelled()
        if self.backend_name == 'lancedb' and self.backend.index_pending():
            say('index', 'building vector index')
            self.backend.finalize_index(say)
            did = True
        cancelled()
        with self.meta._lock:
            av = self.meta.conn.execute('PRAGMA auto_vacuum').fetchone()[0]
        if not did and av == 2:
            return
        with self.meta._lock:
            prev_ac = self.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            self.meta.conn.execute('PRAGMA wal_autocheckpoint=0')
            try:
                # A full VACUUM whenever work ran (or the setting was lost): freed pages
                # (e.g. dropped chunk tables after a backend switch) sit in the freelist
                # until then, and the VACUUM persists auto_vacuum=INCREMENTAL.
                self.meta.conn.execute('PRAGMA auto_vacuum=INCREMENTAL')
                self.meta.conn.execute('VACUUM')
                # VACUUM in WAL mode leaves the rebuilt DB in the WAL; drain it into
                # the main file now (the store stays open, so no close-time checkpoint).
                self.meta.wal_checkpoint_truncate()
            finally:
                self.meta.conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')

    # -- backend/codec state -------------------------------------------------------

    def _lancedb_dir(self) -> str:
        return lancedb_dir_for(self.db_path)

    def _lancedb_has_data(self) -> bool:
        try:
            d = self._lancedb_dir()
            return any(n.endswith('.lance') and os.path.isdir(os.path.join(d, n)) for n in os.listdir(d))
        except OSError:
            return False

    def _sqlite_has_chunks(self) -> bool:
        with self.meta._lock:
            rows = self.meta.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'chunks_%'").fetchall()
        return any(not n[0].endswith('__new') for n in rows)

    def _data_location(self):
        """Where chunk data currently lives ('sqlite'/'lancedb'), or None when neither has any."""
        stored = self.meta.get_meta(BACKEND_KEY)
        l = self._lancedb_has_data()
        s = self._sqlite_has_chunks()
        if stored in ('sqlite', 'lancedb') and ((stored == 'lancedb' and l) or (stored == 'sqlite' and s)):
            return stored
        if l:
            return 'lancedb'
        if s:
            return 'sqlite'
        return None

    def _recorded_codec_name(self):
        raw = self.meta.get_meta(TEXT_CODEC_KEY)
        if raw is None:
            return None  # created on demand with whatever is available; never blocks
        try:
            name = json.loads(raw).get('name')
        except Exception:
            return None
        return name

    def _missing_dependency(self, backend_name):
        """Package missing for `backend_name` to be usable (its data readable), or None.

        Only stored data blocks: a dataless library never does, because its codec
        record is healed to a usable one in ensure_codec_setup() instead."""
        if backend_name == 'lancedb' and not _module_available('lancedb'):
            return 'lancedb'
        if (
            backend_name == 'sqlite'
            and self._sqlite_has_chunks()
            and self._recorded_codec_name() == 'zstd'
            and not _module_available('zstandard')
        ):
            return 'zstandard'
        return None

    def _raise_missing(self, dep):
        has_data = self._data_location() is not None
        try:
            self.meta.close()
        except Exception:
            pass
        raise MissingDependencyError(dep, has_data, getattr(self, 'want_backend', None)) from None

    def _swap_backend(self, name):
        """(Re)build the backend object; picks up any meta changes (codec spec, tables)."""
        self.backend_name = name
        if name == 'lancedb':
            self.backend = LanceVectorBackend(self.meta)
        else:
            self.backend = SqliteVectorBackend(self.meta)

    def _drop_backend_storage(self, name):
        """Delete the chunk storage of backend `name` (its tables / directory)."""
        if name == 'sqlite':
            with self.meta._lock:
                rows = self.meta.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'chunks_%'").fetchall()
                for (t,) in rows:
                    self.meta.conn.execute(f'DROP TABLE {t}')
                if rows:
                    self.meta.conn.commit()
        else:
            shutil.rmtree(self._lancedb_dir(), ignore_errors=True)

    # -- backend migration -----------------------------------------------------------

    def _pending_transfer(self):
        """(src, dst) of the cross-backend transfer to run (dst may equal src).

        Prefers the in-flight marker (a resume keeps its original src even after a
        crash flipped where the data is seen to live); otherwise the data moves from
        where it currently is to the wanted backend. A corrupt marker is dropped:
        it cannot be resumed, and the store must not stay wedged on it."""
        raw = self.meta.get_meta(MIGRATE_KEY)
        if raw is not None:
            try:
                p = json.loads(raw)
                src, dst = p.get('src'), p.get('dst')
                if src in ('sqlite', 'lancedb') and dst in ('sqlite', 'lancedb') and src != dst:
                    return src, dst
            except Exception:
                pass
            self.meta.delete_meta(MIGRATE_KEY)
        return self.backend_name, self._wanted_backend()

    def _new_migrate_progress(self, src, dst):
        prog = {'src': src, 'dst': dst, 'models_done': [], 'books_done': []}
        # mark in-flight BEFORE the first row moves: a crash at any point after this
        # must resume the transfer instead of sweeping the source as an orphan
        self.meta.set_meta(MIGRATE_KEY, json.dumps(prog))
        return prog

    def _load_migrate_progress(self, src, dst):
        raw = self.meta.get_meta(MIGRATE_KEY)
        if raw is not None:
            try:
                p = json.loads(raw)
                if p.get('src') == src and p.get('dst') == dst and isinstance(p.get('models_done'), list) and isinstance(p.get('books_done'), list):
                    return p
            except Exception:
                pass
        return self._new_migrate_progress(src, dst)

    def _save_migrate_progress(self, prog):
        self.meta.set_meta(MIGRATE_KEY, json.dumps({'src': prog['src'], 'dst': prog['dst'], 'models_done': sorted(prog['models_done']), 'books_done': sorted(prog['books_done'])}))

    def _backend_book_items(self, src: str, reader, model: str, table: str, book_id: int):
        """(shim_chunk, vector) pairs of one book's chunks in backend `src` (table `table`)."""
        if src == 'sqlite':
            with self.meta._lock:
                rows = self.meta.conn.execute(
                    f'SELECT chunk_no, text_z, chapter_path, vector FROM {table} WHERE book_id=? ORDER BY chunk_no', (book_id,)
                ).fetchall()
                uv = self.meta.conn.execute('PRAGMA user_version').fetchone()[0]
            codec = reader._codec
            # the schema stage ran first in finalize_schema, so the format is settled
            return [(_MigrateRow(n, [p for p in cp.split(' > ') if p], codec.decompress(z)), blob_to_vec(v, uv >= 3)) for n, z, cp, v in rows]
        t = reader._open_named(table)
        if t is None:
            return []
        rows = t.search().where(f'book_id = {int(book_id)}').limit(10_000_000).to_list()
        rows.sort(key=lambda r: int(r['chunk_no']))
        return [(_MigrateRow(int(r['chunk_no']), [p for p in str(r['chapter_path']).split(' > ') if p], str(r['text'])), r['vector']) for r in rows]

    def _migrate_backend(self, src: str, dst: str, say=None):
        """Move all chunk data from the `src` backend to `dst`, then drop the src storage.

        Resumable: meta[MIGRATE_KEY] marks the transfer in-flight from the first row
        moved until completion and records per-book progress; each book is re-created
        in the target (delete + insert), so a restart never duplicates. The
        meta[BACKEND_KEY] flip happens only after every book has moved, and the old
        storage is dropped last — an interruption always leaves a consistent state
        that the next open resumes. `say(stage, detail)` reports per-book progress
        (and doubles as the cancel checkpoint between books)."""
        dep = self._missing_dependency(dst)
        if dep is not None:  # e.g. switching to lancedb before its package is installed
            raise MissingDependencyError(dep, True, dst)
        target = LanceVectorBackend(self.meta) if dst == 'lancedb' else SqliteVectorBackend(self.meta)
        # the current backend object can read its own storage directly; only a resume
        # (after a crash flipped where the data is seen to live) needs a second one
        reader = self.backend if src == self.backend_name else (SqliteVectorBackend(self.meta) if src == 'sqlite' else LanceVectorBackend(self.meta))
        books_by_model: dict[str, list[int]] = {}
        for b in self.meta.indexed_books():
            if b['n_chunks'] and b.get('model'):
                books_by_model.setdefault(b['model'], []).append(b['id'])
        prog = self._load_migrate_progress(src, dst)
        models_done = set(prog['models_done'])
        books_done = set(prog['books_done'])

        def save():
            prog['models_done'] = sorted(models_done)
            prog['books_done'] = sorted(books_done)
            self._save_migrate_progress(prog)

        total_books = sum(len(v) for v in books_by_model.values())
        for model in sorted(books_by_model):
            if model in models_done:
                continue
            table = model_table_name(model)
            for bid in books_by_model[model]:
                if bid in books_done:
                    continue
                items = self._backend_book_items(src, reader, model, table, bid)
                target.delete_book(bid)
                if items:
                    target.insert_chunks(bid, [(c, v, model) for c, v in items])
                books_done.add(bid)
                save()
                if say is not None:
                    say('backend', f'{dst}: book {len(books_done)}/{total_books}')
            models_done.add(model)
            with self.meta._lock:
                self.meta.wal_checkpoint_truncate()
            save()
        self.meta.set_meta(BACKEND_KEY, dst)
        self.meta.delete_meta(MIGRATE_KEY)
        self.meta.delete_meta(RECOMPRESS_KEY)  # the data left sqlite; a codec conversion is moot
        self._drop_backend_storage(src)
        self._swap_backend(dst)

    # -- codec conversion --------------------------------------------------------------

    def _recompress_progress(self):
        raw = self.meta.get_meta(RECOMPRESS_KEY)
        if raw is None:
            return None
        try:
            p = json.loads(raw)
        except Exception:
            return None
        return p if isinstance(p, dict) else None

    def _recompress_chunks(self, say):
        """Convert every sqlite chunk row to the codec named in meta[RECOMPRESS_KEY].

        In place and resumable: rows are updated in keyset batches, each batch
        committed together with its progress marker (meta[RECOMPRESS_KEY]); a row
        already in the target format is recognized by magic bytes and skipped, so
        a restart never double-converts. The codec meta flips only after every
        table has been fully converted, then the marker is deleted."""
        prog = self._recompress_progress()
        if prog is None or prog.get('target') not in ('zstd', 'zlib'):
            return  # no valid marker: nothing to do (pending_stages gates on it)
        want = prog['target']
        if want == 'zstd' and not _module_available('zstandard'):
            # defensive: __init__ blocks the open in this state — name the package anyway
            raise MissingDependencyError('zstandard', self._sqlite_has_chunks(), self.want_backend)
        cur_spec = json.loads(self.meta.get_meta(TEXT_CODEC_KEY))
        target_spec = build_codec_spec(want == 'zstd')
        src = TextCodec(cur_spec)
        dstc = TextCodec(target_spec)
        tables = self.backend._chunk_tables()
        done = set(prog.get('done') or [])
        zstd_magic = b'\x28\xb5\x2f\xfd'

        def convert(blob):
            if prog['target'] == 'zstd':
                if blob[:4] == zstd_magic:
                    return None  # already converted
                text = src.decompress(blob)
            else:
                if blob[:1] == b'\x78':
                    return None
                text = src.decompress(blob)
            return dstc.compress(text)

        with self.meta._lock:
            prev_ac = self.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            self.meta.conn.execute('PRAGMA wal_autocheckpoint=0')
        try:
            for t in tables:
                if t in done:
                    continue
                last_id = prog['last_id'] if prog.get('cur_table') == t else 0
                while True:
                    with self.meta._lock:
                        rows = self.meta.conn.execute(f'SELECT id, text_z FROM {t} WHERE id>? ORDER BY id LIMIT 500', (last_id,)).fetchall()
                    if not rows:
                        break
                    updates = []
                    for rid, blob in rows:
                        new = convert(blob)
                        if new is not None and new != blob:
                            updates.append((new, rid))
                    last_id = rows[-1][0]
                    with self.meta._lock:
                        if updates:
                            self.meta.conn.executemany(f'UPDATE {t} SET text_z=? WHERE id=?', updates)
                    prog['cur_table'] = t
                    prog['last_id'] = last_id
                    self.meta.set_meta(RECOMPRESS_KEY, json.dumps(prog))  # commits the batch + progress atomically
                    say('codec', f'{t}: row {last_id}')
                done.add(t)
                prog['done'] = sorted(done)
                prog['cur_table'] = None
                prog['last_id'] = 0
                self.meta.set_meta(RECOMPRESS_KEY, json.dumps(prog))
                with self.meta._lock:
                    self.meta.wal_checkpoint_truncate()
        finally:
            with self.meta._lock:
                self.meta.conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')
        self.meta.delete_meta(RECOMPRESS_KEY)
        self.meta.set_meta(TEXT_CODEC_KEY, json.dumps(target_spec))
        with self.meta._lock:
            self.meta.wal_checkpoint_truncate()
        self._swap_backend('sqlite')  # rebuild the TextCodec on the new spec

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

    def add_dirty(self, book_id, reason='added'):
        self.meta.add_dirty(book_id, reason)

    def add_dirty_many(self, book_ids, reason='added'):
        self.meta.add_dirty_many(book_ids, reason)

    def dirty_book_ids(self):
        return self.meta.dirty_book_ids()

    def remove_dirty(self, book_id):
        self.meta.remove_dirty(book_id)

    def book_is_indexed(self, book_id):
        return self.meta.book_is_indexed(book_id)

    def indexed_books(self):
        return self.meta.indexed_books()

    def get_file_info(self, book_id):
        return self.meta.get_file_info(book_id)

    def set_file_info(self, book_id, fmt, size, mtime_s):
        self.meta.set_file_info(book_id, fmt, size, mtime_s)

    def clear_file_info(self, book_id):
        self.meta.clear_file_info(book_id)

    def file_info_book_ids(self):
        return self.meta.file_info_book_ids()

    # -- indexing --------------------------------------------------------------------

    def clear_book(self, book_id: int):
        self.backend.delete_book(book_id)
        self.meta.clear_book(book_id)

    def insert_chunk(self, book_id, chunk, model, vector):
        self._pending.append((book_id, (chunk, vector, model)))

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

    def upsert_book(self, book_id, fmt, n_chunks, model):
        self.meta.upsert_book(book_id, fmt, n_chunks, model)

    def set_attrs(self, book_id, values):
        self.meta.set_attrs(book_id, values)

    def get_attrs(self, book_id):
        return self.meta.get_attrs(book_id)

    def all_attrs(self):
        return self.meta.all_attrs()

    def clear_attrs(self, book_id):
        self.meta.clear_attrs(book_id)

    def attr_book_ids(self):
        return self.meta.attr_book_ids()

    def attrs_fields(self):
        return self.meta.attrs_fields()

    def set_failed(self, book_id, kind, error):
        self.meta.set_failed(book_id, kind, error)

    def clear_failed(self, book_id, kind=None):
        self.meta.clear_failed(book_id, kind)

    def wipe_failed(self, kind):
        self.meta.wipe_failed(kind)

    def failed_book_ids(self, kind=None):
        return self.meta.failed_book_ids(kind)

    def failed_entries(self, kind=None):
        return self.meta.failed_entries(kind)

    # -- search ------------------------------------------------------------------------

    def search(self, query_vec, limit: int = 20, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        """Top-`limit` books by their best chunk's cosine similarity (one result per book)."""
        return self.backend.search(query_vec, limit, min_score, model)

    def search_book(self, query_vec, book_id: int, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        """All of one book's chunks scoring >= min_score, best first (no cap)."""
        return self.backend.search_book(query_vec, book_id, min_score, model)

    def book_chunks_text(self, book_id: int):
        return self.backend.book_chunks_text(book_id)

    def cleanup_stale_models(self, current_model=None):
        """Drop chunk tables left over from embedding models no book is indexed with."""
        return self.backend.drop_stale_models(current_model)
