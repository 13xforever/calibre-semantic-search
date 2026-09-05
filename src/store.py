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
    cleanup_stale_models(current_model=None)
    close

Backends (both keep one chunk table per embedding model):
  'sqlite'  - vectors in a local SQLite file (always available; numpy speeds it up)
  'lancedb' - vectors in LanceDB (optional dependency, lazy import)
  'auto'    - lancedb if importable, else sqlite

Schema versioning: the meta tables (books/dirty/attrs_raw plus the models/formats/
file_info registries) are versioned with PRAGMA user_version and migrated
structurally on open. The sqlite chunk tables migrate from the legacy single
`chunks` table to slim per-model tables whose text lives in a compressed `text_z`
BLOB; the codec is recorded once in meta['text_codec'] and used verbatim by both
reads and writes.
'''

from __future__ import annotations

import base64
import heapq
import json
import os
import re
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

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


# -- text compression -----------------------------------------------------------

TEXT_CODEC_KEY = 'text_codec'
MODEL_ALIASES_KEY = 'model_aliases'
ZSTD_LEVEL = 3
DEFAULT_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'default_compression_dict.bin')


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


def _install_zstandard(progress=None) -> bool:
    """Best-effort pip install of zstandard into calibre's Python.

    Self-contained on purpose: this module is loaded standalone in tests and
    cannot import from the plugin package (see utils.install_zstandard for the
    shared-UI twin). Returns True when the package imports afterwards.
    """

    def log(line):
        if progress is not None:
            try:
                progress(line)
            except Exception:
                pass
        print(f'[semantic-search] {line}', file=sys.stderr, flush=True)

    for extra in ((), ('--user',)):
        cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *extra, 'zstandard']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            log(f'zstandard install did not start: {e!r}')
            continue
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(line)
            rc = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            log('zstandard install timed out')
            continue
        if rc == 0:
            return True
        log(f'pip exited with code {rc}')
    return False


def _import_zstandard(progress=None):
    try:
        import zstandard

        return zstandard
    except ImportError:
        pass
    if _install_zstandard(progress):
        import importlib

        importlib.invalidate_caches()
        try:
            import zstandard

            return zstandard
        except ImportError:
            pass
    return None


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


def ensure_codec_setup(meta: 'MetaStore', progress=None):
    """Record the text codec in meta (once per DB).

    Runs at migration / new-DB init only — never on the write path. Installs
    zstandard when it is missing; falls back to zlib when that is impossible.
    """
    if meta.get_meta(TEXT_CODEC_KEY) is not None:
        return
    spec = build_codec_spec(_import_zstandard(progress) is not None)
    meta.set_meta(TEXT_CODEC_KEY, json.dumps(spec))


# -- DDL ------------------------------------------------------------------------

SCHEMA_VERSION = 2

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


def chunks_table_sql_v1(name: str) -> str:
    """Legacy (v1) chunk-table shape, used only as the v0->v1 split target."""
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


def _parse_fileinfo_value(value: str):
    """Legacy meta value 'FMT|size|mtime' -> (fmt, size, mtime_s); None when incomplete."""
    parts = (value or '').split('|')
    if len(parts) != 3:
        return None
    fmt, size_s, mtime_s = parts
    if not fmt:
        return None
    try:
        size = int(size_s)
        mtime = int(Decimal(mtime_s).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    except (ValueError, ArithmeticError):
        return None
    return fmt, size, mtime


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
        self.migrated = False
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
                self.migrated = self._migrate_meta()
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    def wal_checkpoint_truncate(self):
        """Drain the WAL into the main DB file and truncate it. Caller holds self._lock."""
        self.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')

    # -- schema migration --------------------------------------------------------

    def _user_version(self) -> int:
        return int(self.conn.execute('PRAGMA user_version').fetchone()[0])

    def _has_table(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def _table_cols(self, name: str) -> set[str]:
        return {r[1] for r in self.conn.execute(f'PRAGMA table_info({name})').fetchall()}

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

    def _record_model_alias(self, raw_model: str, norm: str):
        try:
            row = self.conn.execute('SELECT value FROM meta WHERE key=?', (MODEL_ALIASES_KEY,)).fetchone()
            aliases = json.loads(row[0]) if row else {}
        except Exception:
            aliases = {}
        if not isinstance(aliases, dict):
            aliases = {}
        if aliases.get(raw_model) != norm:
            aliases[raw_model] = norm
            self.conn.execute(
                'INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (MODEL_ALIASES_KEY, json.dumps(aliases)),
            )

    def _migrate_meta(self) -> bool:
        """Bring the meta tables from any legacy shape to v2.

        Structural and resumable: each sub-step detects whether its own work is
        still pending and runs in its own committed transaction. Returns True when
        anything was migrated (the caller then runs the final VACUUM).
        """
        if self._user_version() >= SCHEMA_VERSION:
            return False
        did = False
        books_cols = self._table_cols('books')
        dirty_cols = self._table_cols('dirty') if self._has_table('dirty') else set()
        attrs_cols = self._table_cols('attrs_raw') if self._has_table('attrs_raw') else set()

        # 1) models/formats registries, populated while the legacy columns still exist
        model_names: set[str] = set()
        fmt_names: set[str] = set()
        raw_models: list[str] = []
        if 'model' in books_cols:
            raw_models = [r[0] for r in self.conn.execute('SELECT DISTINCT model FROM books')]
            model_names = {normalize_model(m) for m in raw_models}
        if 'fmt' in books_cols:
            fmt_names |= {r[0] for r in self.conn.execute('SELECT DISTINCT fmt FROM books')}
        if 'fmt' in dirty_cols:
            fmt_names |= {r[0] for r in self.conn.execute('SELECT DISTINCT fmt FROM dirty')}
        fileinfo_keys = [k for k in self._meta_keys_locked('fileinfo:') if k.split(':', 1)[1].isdigit()]
        for key in fileinfo_keys:
            parsed = _parse_fileinfo_value(self._meta_value_locked(key))
            if parsed is not None:
                fmt_names.add(parsed[0])
        if model_names or fmt_names:
            for m in sorted(model_names):
                self._upsert_model(m)
            for f in sorted(fmt_names):
                self._upsert_format(f)
            for raw in raw_models:
                if raw is not None:
                    self._record_model_alias(raw, normalize_model(raw))
            self.conn.commit()
            did = True

        # 2) books: fmt/model TEXT -> fmt_id/model_id, indexed_at REAL -> INTEGER seconds
        if 'model' in books_cols:
            rows = self.conn.execute('SELECT id, fmt, indexed_at, n_chunks, model FROM books').fetchall()
            self.conn.execute('DROP TABLE IF EXISTS books__new')
            self.conn.execute(
                'CREATE TABLE books__new('
                'id INTEGER PRIMARY KEY, fmt_id INTEGER NOT NULL, indexed_at INTEGER, '
                'n_chunks INTEGER NOT NULL DEFAULT 0, model_id INTEGER NOT NULL)'
            )
            self.conn.executemany(
                'INSERT INTO books__new(id, fmt_id, indexed_at, n_chunks, model_id) VALUES(?,?,?,?,?)',
                [
                    (bid, self._fmt_id(fmt), int(round(iat)) if iat is not None else None, nch, self._model_id(normalize_model(model)))
                    for bid, fmt, iat, nch, model in rows
                ],
            )
            self.conn.execute('DROP TABLE books')
            self.conn.execute('ALTER TABLE books__new RENAME TO books')
            self.conn.commit()
            did = True

        # 3) dirty: drop write-only fmt column, added_at REAL -> INTEGER seconds
        if 'fmt' in dirty_cols:
            rows = self.conn.execute('SELECT book_id, reason, added_at FROM dirty').fetchall()
            self.conn.execute('DROP TABLE IF EXISTS dirty__new')
            self.conn.execute(
                'CREATE TABLE dirty__new('
                'book_id INTEGER PRIMARY KEY, reason TEXT NOT NULL DEFAULT \'added\', added_at INTEGER)'
            )
            self.conn.executemany(
                'INSERT INTO dirty__new(book_id, reason, added_at) VALUES(?,?,?)',
                [(bid, reason, int(round(at)) if at is not None else None) for bid, reason, at in rows],
            )
            self.conn.execute('DROP TABLE dirty')
            self.conn.execute('ALTER TABLE dirty__new RENAME TO dirty')
            self.conn.commit()
            did = True

        # 4) attrs_raw: drop never-read updated_at column
        if 'updated_at' in attrs_cols:
            rows = self.conn.execute('SELECT book_id, json, fields FROM attrs_raw').fetchall()
            self.conn.execute('DROP TABLE IF EXISTS attrs_raw__new')
            self.conn.execute(
                "CREATE TABLE attrs_raw__new("
                "book_id INTEGER PRIMARY KEY, json TEXT NOT NULL DEFAULT '{}', fields TEXT NOT NULL DEFAULT '')"
            )
            self.conn.executemany('INSERT INTO attrs_raw__new(book_id, json, fields) VALUES(?,?,?)', rows)
            self.conn.execute('DROP TABLE attrs_raw')
            self.conn.execute('ALTER TABLE attrs_raw__new RENAME TO attrs_raw')
            self.conn.commit()
            did = True

        # 5) file_info: move meta['fileinfo:<id>'] values into a table (all NOT NULL;
        #    incomplete legacy values are dropped — a missing row means "unknown file
        #    info", which already triggers a reindex)
        if fileinfo_keys:
            for key in fileinfo_keys:
                bid = int(key.split(':', 1)[1])
                parsed = _parse_fileinfo_value(self._meta_value_locked(key))
                if parsed is not None:
                    fmt, size, mtime_s = parsed
                    self.conn.execute(
                        'INSERT INTO file_info(book_id, fmt_id, size, mtime_s) VALUES(?,?,?,?) '
                        'ON CONFLICT(book_id) DO UPDATE SET fmt_id=excluded.fmt_id, size=excluded.size, mtime_s=excluded.mtime_s',
                        (bid, self._fmt_id(fmt), size, mtime_s),
                    )
                self.conn.execute('DELETE FROM meta WHERE key=?', (key,))
            self.conn.commit()
            did = True

        self.conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        self.conn.commit()
        return did

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
            self._record_model_alias(model, norm)
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
        ensure_codec_setup(meta)
        self._codec = TextCodec(json.loads(meta.get_meta(TEXT_CODEC_KEY)))
        self._text_len_cache: dict[str, int] = {}

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

    def _set_chunk_model(self, name: str, norm: str):
        self.conn.execute(
            'INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
            (f'chunk_model:{name}', norm),
        )

    def _ensure_table(self, model: str) -> str:
        name = self._table_for(model)
        with self.meta._lock:
            if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                self.conn.executescript(chunks_table_sql(name))
                self._set_chunk_model(name, normalize_model(model))
                self.conn.commit()
        return name

    # -- migration -----------------------------------------------------------------

    def _legacy_table_locked(self):
        """First per-model table still in the legacy (v1) shape, or None."""
        for t in self._chunk_tables_locked():
            cols = {r[1] for r in self.conn.execute(f'PRAGMA table_info({t})').fetchall()}
            if 'text' in cols:
                return t
        return None

    def pending_work(self) -> bool:
        with self.meta._lock:
            if self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone():
                return True
            return self._legacy_table_locked() is not None

    def finalize(self) -> bool:
        """Run the pending chunk migrations (v0 split, then v2 slim per table).

        Each model group / table is committed independently so progress survives an
        interrupt. While running, WAL auto-checkpointing is disabled and the WAL is
        checkpointed explicitly (TRUNCATE) at phase boundaries: with a multi-GB WAL
        the default in-commit passive checkpoint can stall for minutes on slow
        storage. Returns True when anything was migrated."""
        did = False
        with self.meta._lock:
            prev_ac = self.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            self.conn.execute('PRAGMA wal_autocheckpoint=0')
        try:
            with self.meta._lock:
                if self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone():
                    self._split_legacy_chunks()
                    did = True
                self.meta.wal_checkpoint_truncate()
            while True:
                with self.meta._lock:
                    t = self._legacy_table_locked()
                if t is None:
                    break
                self._slim_table(t)
                with self.meta._lock:
                    self.meta.wal_checkpoint_truncate()
                did = True
        finally:
            with self.meta._lock:
                self.conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')
        return did

    def _legacy_where(self, raws):
        parts, params = [], []
        for m in raws:
            if m is None:
                parts.append('model IS NULL')
            else:
                parts.append('model=?')
                params.append(m)
        return '(' + ' OR '.join(parts) + ')', params

    def _split_legacy_chunks(self):
        """v0 -> v1: copy rows from the single `chunks` table into per-model tables
        (grouped by normalized model name), then drop it. One commit per model group
        bounds WAL growth; a group whose target already holds all source rows is
        skipped, so an interrupted split resumes where it left off."""
        raw_models = [r[0] for r in self.conn.execute('SELECT DISTINCT model FROM chunks')]
        groups: dict[str, list] = {}
        for m in raw_models:
            groups.setdefault(normalize_model(m), []).append(m)
        cols = 'book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector'
        for norm, raws in sorted(groups.items()):
            name = model_table_name(norm)
            if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                self.conn.executescript(chunks_table_sql_v1(name))
            where, params = self._legacy_where(raws)
            src = self.conn.execute(f'SELECT COUNT(*) FROM chunks WHERE {where}', params).fetchone()[0]
            dst = self.conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
            if dst < src:
                # OR REPLACE: rows can pre-exist after an interrupted split; re-copying
                # the same ids is safe (identical values).
                self.conn.execute(f'INSERT OR REPLACE INTO {name}({cols}) SELECT {cols} FROM chunks WHERE {where}', params)
                self._set_chunk_model(name, norm)
            self.conn.commit()
        self.conn.execute('DROP TABLE chunks')
        self.conn.commit()

    def _slim_table(self, t):
        """Rebuild one legacy-shape chunk table into the slim v2 shape. Single forward
        pass into a staging table (keyset resume point = staging MAX(id)); the swap is
        one transaction so the old table stays fully readable until it commits. The
        lock is only held for SQL, so searches can interleave between batches."""
        staging = f'{t}__new'
        with self.meta._lock:
            row = self.conn.execute(f'SELECT model FROM {t} LIMIT 1').fetchone()
            norm = normalize_model(row[0]) if row and row[0] else 'model'
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS {staging}('
                'id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_no INTEGER NOT NULL, '
                "text_z BLOB NOT NULL, chapter_path TEXT NOT NULL DEFAULT '', vector BLOB NOT NULL)"
            )
        codec = self._codec
        while True:
            with self.meta._lock:
                last_id = self.conn.execute(f'SELECT COALESCE(MAX(id), 0) FROM {staging}').fetchone()[0]
                rows = self.conn.execute(
                    f'SELECT id, book_id, chunk_no, text, chapter_path, vector FROM {t} WHERE id>? ORDER BY id LIMIT 2000',
                    (last_id,),
                ).fetchall()
            if not rows:
                break
            payload = [(i, b, n, codec.compress(tx), cp or '', v) for (i, b, n, tx, cp, v) in rows]
            with self.meta._lock:
                self.conn.executemany(
                    'INSERT INTO {}(id, book_id, chunk_no, text_z, chapter_path, vector) VALUES(?,?,?,?,?,?)'.format(staging),
                    payload,
                )
                self.conn.commit()
        with self.meta._lock:
            total = self.conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
            done = self.conn.execute(f'SELECT COUNT(*) FROM {staging}').fetchone()[0]
            if done != total:
                raise RuntimeError(f'chunk migration row count mismatch for {t}: {done}/{total}')
            for _sid, stz, old in self.conn.execute(f'SELECT s.id, s.text_z, o.text FROM {staging} s JOIN {t} o ON o.id=s.id LIMIT 5'):
                if codec.decompress(stz) != old:
                    raise RuntimeError(f'chunk migration round-trip mismatch for {t}')
            canonical = model_table_name(norm)
            if canonical != t and self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (canonical,)).fetchone():
                canonical = t  # name already taken by another table; keep this one
            self.conn.execute(f'DROP TABLE {t}')
            self.conn.execute(f'ALTER TABLE {staging} RENAME TO {canonical}')
            self.conn.execute(f'CREATE INDEX idx_{canonical}_book ON {canonical}(book_id)')
            if canonical != t:
                self.conn.execute('DELETE FROM meta WHERE key=?', (f'chunk_model:{t}',))
                self._set_chunk_model(canonical, norm)
            self.conn.commit()

    def drop_stale_models(self, current_model: str | None = None) -> int:
        """Drop chunk tables whose model no longer has any indexed book (e.g. after a model switch)."""
        in_use = {b['model'] for b in self.meta.indexed_books() if b.get('model')}
        if current_model:
            in_use.add(normalize_model(current_model))
        dropped = 0
        with self.meta._lock:
            for t in self._chunk_tables_locked():
                row = self.conn.execute('SELECT value FROM meta WHERE key=?', (f'chunk_model:{t}',)).fetchone()
                norm = row[0] if row else None
                if norm is None:
                    continue  # unknown table; leave it alone
                if norm not in in_use:
                    self.conn.execute(f'DROP TABLE {t}')
                    self.conn.execute('DELETE FROM meta WHERE key=?', (f'chunk_model:{t}',))
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
        rows = [
            (book_id, c.chunk_no, self._codec.compress(c.text), ' > '.join(c.chapter_path or []), vec_to_blob(v))
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
            free = SEARCH_MIN_BUDGET * 4  # unknown: assume a modest amount of headroom
        return max(SEARCH_MIN_BUDGET, free // 2)

    def _table_dim(self, name: str):
        with self.meta._lock:
            row = self.conn.execute(f'SELECT LENGTH(vector) FROM {name} LIMIT 1').fetchone()
        return row[0] // 4 if row and row[0] else None

    def _avg_text_chars(self, name: str) -> int:
        cached = self._text_len_cache.get(name)
        if cached is not None:
            return cached
        with self.meta._lock:
            rows = self.conn.execute(f'SELECT text_z FROM {name} LIMIT 200').fetchall()
        total = sum(len(self._codec.decompress(z)) for z, in rows)
        avg = int(total / len(rows)) if rows else 1024
        self._text_len_cache[name] = avg
        return avg

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
        tables = [self._table_for(model)] if model is not None else self._chunk_tables()
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
                        f'SELECT c.id, c.book_id, c.chunk_no, c.text_z, c.chapter_path, f.fmt, c.vector '
                        f'FROM {t} c LEFT JOIN books b ON b.id=c.book_id LEFT JOIN formats f ON f.id=b.fmt_id '
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
                                fmt=r[5] or '',
                                chunk_no=r[2],
                                text=self._codec.decompress(r[3]),
                                chapter_path=[p for p in r[4].split(' > ') if p],
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
    Table names keep the historical raw-model slug so existing tables are
    untouched; new tables use the slim schema.
    """

    name = 'lancedb'

    def __init__(self, meta: MetaStore):
        import lancedb  # lazy: optional dependency

        self.meta = meta
        db_path = os.path.splitext(meta.db_path)[0] + '-lancedb'
        self._db = lancedb.connect(db_path)
        self._tables: dict[str, object] = {}

    def pending_work(self) -> bool:
        return False  # existing tables are untouched; new tables are created slim on demand

    def finalize(self) -> bool:
        return False

    def _table_name(self, model: str) -> str:
        # historical naming: slug of the raw model string as passed at insert time
        return model_table_name(model or '')

    def _candidate_models(self, current_model: str | None = None):
        """Model spellings whose tables may hold live data.

        indexed_books() returns normalized names, but pre-existing tables are named
        after the raw setting string — so keep both spellings in play.
        """
        in_use = {i['model'] for i in self.meta.indexed_books() if i.get('model')}
        if current_model:
            in_use.add(normalize_model(current_model))
        try:
            aliases = json.loads(self.meta.get_meta(MODEL_ALIASES_KEY, '{}') or '{}')
        except Exception:
            aliases = {}
        if not isinstance(aliases, dict):
            aliases = {}
        out = set(in_use)
        for raw, norm in aliases.items():
            if norm in in_use:
                out.add(raw)
        return out

    def _open_table(self, model: str):
        for name in {self._table_name(model), self._table_name(normalize_model(model))}:
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
        for name in {self._table_name(model), self._table_name(normalize_model(model))}:
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
                ('vector', pa.list_(pa.float32(), dim)),
            ]
        )
        t = self._db.create_table(self._table_name(model), schema=schema)
        self._tables[self._table_name(model)] = t
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
        self._pending: list[tuple[int, object]] = []  # (book_id, [(chunk, vec, model)])
        self._finalized = False

    def needs_finalize(self) -> bool:
        """True when a migration ran or is pending and the final VACUUM has not run yet."""
        if self._finalized:
            return False
        if self.meta.migrated or self.backend.pending_work():
            return True
        # fully migrated but the final VACUUM never completed (e.g. interrupted last time):
        # in WAL mode the auto_vacuum setting only persists through a finished VACUUM,
        # so av != 2 on an otherwise-v2 DB means the space was never reclaimed
        with self.meta._lock:
            return self.meta.conn.execute('PRAGMA auto_vacuum').fetchone()[0] != 2

    def finalize_schema(self):
        """Run pending chunk migrations, then one final VACUUM if anything was migrated
        or the previous VACUUM never completed.

        Idempotent; returns immediately when nothing is pending. Meant to be called
        from a worker thread before indexing starts (see gui._start_for_library)."""
        did = self.backend.finalize()
        with self.meta._lock:
            av = self.meta.conn.execute('PRAGMA auto_vacuum').fetchone()[0]
        if not (did or self.meta.migrated) and av == 2:
            self._finalized = True
            return
        with self.meta._lock:
            prev_ac = self.meta.conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
            self.meta.conn.execute('PRAGMA wal_autocheckpoint=0')
            try:
                if av != 2:  # 2 == INCREMENTAL
                    self.meta.conn.execute('PRAGMA auto_vacuum=INCREMENTAL')
                    self.meta.conn.execute('VACUUM')
                # VACUUM in WAL mode leaves the rebuilt DB in the WAL; drain it into
                # the main file now (the store stays open, so no close-time checkpoint).
                self.meta.wal_checkpoint_truncate()
            finally:
                self.meta.conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')
        self._finalized = True

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

    def add_dirty(self, book_id, reason='added'):
        self.meta.add_dirty(book_id, reason)

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

    def clear_attrs(self, book_id):
        self.meta.clear_attrs(book_id)

    def attr_book_ids(self):
        return self.meta.attr_book_ids()

    # -- search ------------------------------------------------------------------------

    def search(self, query_vec, limit: int = 20, min_score: float = 0.0, model: str | None = None) -> list[SearchResult]:
        return self.backend.search(query_vec, limit, min_score, model)

    def book_chunks_text(self, book_id: int):
        return self.backend.book_chunks_text(book_id)

    def cleanup_stale_models(self, current_model=None):
        """Drop chunk tables left over from embedding models no book is indexed with."""
        return self.backend.drop_stale_models(current_model)
