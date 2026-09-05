'''Schema migration v1: legacy meta tables -> current shape.

Brings the whole database from version 0 to version 1: books/dirty/attrs_raw are
rebuilt and the models/formats/file_info registries are created; the chunk tables
are untouched in this step (they migrate in v2). Runs synchronously on open,
because the write path needs the current meta shape.

Structural and resumable: each sub-step detects whether its own work is still
pending and runs in its own committed transaction. Called from MetaStore.__init__
with the store lock already held.
'''

from __future__ import annotations

from decimal import Decimal

from ..store import normalize_model


def _user_version(conn) -> int:
    return int(conn.execute('PRAGMA user_version').fetchone()[0])


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _table_cols(conn, name: str) -> set[str]:
    return {r[1] for r in conn.execute(f'PRAGMA table_info({name})').fetchall()}


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
        mtime = int(Decimal(mtime_s))  # truncate: a stored mtime must never be later than the file's real one
    except (ValueError, ArithmeticError):
        return None
    return fmt, size, mtime


def upgrade(meta) -> bool:
    """Bring the meta tables from any legacy shape to v1 (the current meta shape).

    Structural and resumable: each sub-step detects whether its own work is still
    pending and runs in its own committed transaction. Returns True when anything
    was migrated (the caller then runs the final VACUUM). Caller holds meta._lock.
    """
    conn = meta.conn
    if _user_version(conn) >= 1:
        return False
    did = False
    books_cols = _table_cols(conn, 'books')
    dirty_cols = _table_cols(conn, 'dirty') if _has_table(conn, 'dirty') else set()
    attrs_cols = _table_cols(conn, 'attrs_raw') if _has_table(conn, 'attrs_raw') else set()

    # 1) models/formats registries, populated while the legacy columns still exist
    model_names: set[str] = set()
    fmt_names: set[str] = set()
    if 'model' in books_cols:
        model_names = {normalize_model(m) for m, in conn.execute('SELECT DISTINCT model FROM books') if m is not None}
    if 'fmt' in books_cols:
        fmt_names |= {r[0] for r in conn.execute('SELECT DISTINCT fmt FROM books')}
    if 'fmt' in dirty_cols:
        fmt_names |= {r[0] for r in conn.execute('SELECT DISTINCT fmt FROM dirty')}
    fileinfo_keys = [k for k in meta._meta_keys_locked('fileinfo:') if k.split(':', 1)[1].isdigit()]
    for key in fileinfo_keys:
        parsed = _parse_fileinfo_value(meta._meta_value_locked(key))
        if parsed is not None:
            fmt_names.add(parsed[0])
    if model_names or fmt_names:
        for m in sorted(model_names):
            meta._upsert_model(m)
        for f in sorted(fmt_names):
            meta._upsert_format(f)
        conn.commit()
        did = True

    # 2) books: fmt/model TEXT -> fmt_id/model_id, indexed_at REAL -> INTEGER seconds
    if 'model' in books_cols:
        rows = conn.execute('SELECT id, fmt, indexed_at, n_chunks, model FROM books').fetchall()
        conn.execute('DROP TABLE IF EXISTS books__new')
        conn.execute(
            'CREATE TABLE books__new('
            'id INTEGER PRIMARY KEY, fmt_id INTEGER NOT NULL, indexed_at INTEGER, '
            'n_chunks INTEGER NOT NULL DEFAULT 0, model_id INTEGER NOT NULL)'
        )
        conn.executemany(
            'INSERT INTO books__new(id, fmt_id, indexed_at, n_chunks, model_id) VALUES(?,?,?,?,?)',
            [
                (bid, meta._fmt_id(fmt), int(round(iat)) if iat is not None else None, nch, meta._model_id(normalize_model(model)))
                for bid, fmt, iat, nch, model in rows
            ],
        )
        conn.execute('DROP TABLE books')
        conn.execute('ALTER TABLE books__new RENAME TO books')
        conn.commit()
        did = True

    # 3) dirty: drop write-only fmt column, added_at REAL -> INTEGER seconds
    if 'fmt' in dirty_cols:
        rows = conn.execute('SELECT book_id, reason, added_at FROM dirty').fetchall()
        conn.execute('DROP TABLE IF EXISTS dirty__new')
        conn.execute(
            'CREATE TABLE dirty__new('
            'book_id INTEGER PRIMARY KEY, reason TEXT NOT NULL DEFAULT \'added\', added_at INTEGER)'
        )
        conn.executemany(
            'INSERT INTO dirty__new(book_id, reason, added_at) VALUES(?,?,?)',
            [(bid, reason, int(round(at)) if at is not None else None) for bid, reason, at in rows],
        )
        conn.execute('DROP TABLE dirty')
        conn.execute('ALTER TABLE dirty__new RENAME TO dirty')
        conn.commit()
        did = True

    # 4) attrs_raw: drop never-read updated_at column
    if 'updated_at' in attrs_cols:
        rows = conn.execute('SELECT book_id, json, fields FROM attrs_raw').fetchall()
        conn.execute('DROP TABLE IF EXISTS attrs_raw__new')
        conn.execute(
            "CREATE TABLE attrs_raw__new("
            "book_id INTEGER PRIMARY KEY, json TEXT NOT NULL DEFAULT '{}', fields TEXT NOT NULL DEFAULT '')"
        )
        conn.executemany('INSERT INTO attrs_raw__new(book_id, json, fields) VALUES(?,?,?)', rows)
        conn.execute('DROP TABLE attrs_raw')
        conn.execute('ALTER TABLE attrs_raw__new RENAME TO attrs_raw')
        conn.commit()
        did = True

    # 5) file_info: move meta['fileinfo:<id>'] values into a table (all NOT NULL;
    #    incomplete legacy values are dropped — a missing row means "unknown file
    #    info", which already triggers a reindex)
    if fileinfo_keys:
        for key in fileinfo_keys:
            bid = int(key.split(':', 1)[1])
            parsed = _parse_fileinfo_value(meta._meta_value_locked(key))
            if parsed is not None:
                fmt, size, mtime_s = parsed
                conn.execute(
                    'INSERT INTO file_info(book_id, fmt_id, size, mtime_s) VALUES(?,?,?,?) '
                    'ON CONFLICT(book_id) DO UPDATE SET fmt_id=excluded.fmt_id, size=excluded.size, mtime_s=excluded.mtime_s',
                    (bid, meta._fmt_id(fmt), size, mtime_s),
                )
            conn.execute('DELETE FROM meta WHERE key=?', (key,))
        conn.commit()
        did = True

    conn.execute('PRAGMA user_version=1')
    conn.commit()
    return did
