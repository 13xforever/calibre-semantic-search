'''Schema migration v2: legacy chunk layout -> slim per-model tables.

Brings the whole database from version 1 to version 2: the meta tables are already
current from v1, so this step migrates the chunk tables — phase 1 splits the
single legacy `chunks` table into per-model tables, phase 2 slims each fat table
(compressed text_z, no para/offset/model/dim columns). Runs deferred in
VectorStore.finalize_schema (background thread) before indexing starts.

Resumable: one commit per model group / table bounds WAL growth and survives an
interrupt; a group whose target already holds all source rows is skipped. While
running, WAL auto-checkpointing is disabled and the WAL is checkpointed explicitly
(TRUNCATE) at phase boundaries: with a multi-GB WAL the default in-commit passive
checkpoint can stall for minutes on slow storage.
'''

from __future__ import annotations

from ..store import model_table_name, normalize_model


def _legacy_table_sql(name: str) -> str:
    """Legacy fat chunk-table shape, used only as the split target."""
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


def has_legacy_table(conn) -> bool:
    """True when the single pre-v2 `chunks` table still exists."""
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is not None


def first_legacy_table(conn):
    """First per-model table still in the legacy (fat) shape, or None."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in sorted(r[0] for r in rows if r[0].startswith('chunks_') and not r[0].endswith('__new')):
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info({t})').fetchall()}
        if 'text' in cols:
            return t
    return None


def _legacy_where(raws):
    parts, params = [], []
    for m in raws:
        if m is None:
            parts.append('model IS NULL')
        else:
            parts.append('model=?')
            params.append(m)
    return '(' + ' OR '.join(parts) + ')', params


def pending(backend) -> bool:
    """True when the chunk tables are not yet in the current (v2) shape."""
    with backend.meta._lock:
        conn = backend.conn
        if has_legacy_table(conn):
            return True
        return first_legacy_table(conn) is not None


def upgrade(backend) -> bool:
    """Bring the chunk tables from the legacy shapes to the current slim shape.

    Returns True when anything was migrated."""
    conn = backend.conn
    lock = backend.meta._lock
    did = False
    with lock:
        prev_ac = conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
        conn.execute('PRAGMA wal_autocheckpoint=0')
    try:
        with lock:
            if has_legacy_table(conn):
                _split(backend)
                did = True
            backend.meta.wal_checkpoint_truncate()
        while True:
            with lock:
                t = first_legacy_table(conn)
            if t is None:
                break
            slim_table(backend, t)
            with lock:
                backend.meta.wal_checkpoint_truncate()
            did = True
    finally:
        with lock:
            conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')
    with lock:
        conn.execute('PRAGMA user_version=2')
        conn.commit()
    return did


def _split(backend) -> None:
    """Phase 1: copy rows from the single `chunks` table into per-model tables
    (grouped by normalized model name), then drop it. Caller holds backend.meta._lock."""
    conn = backend.conn
    raw_models = [r[0] for r in conn.execute('SELECT DISTINCT model FROM chunks')]
    groups: dict[str, list] = {}
    for m in raw_models:
        groups.setdefault(normalize_model(m), []).append(m)
    cols = 'book_id, chunk_no, text, chapter_path, para_start, para_end, char_offset, model, dim, vector'
    for norm, raws in sorted(groups.items()):
        name = model_table_name(norm)
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
            conn.executescript(_legacy_table_sql(name))
        where, params = _legacy_where(raws)
        src = conn.execute(f'SELECT COUNT(*) FROM chunks WHERE {where}', params).fetchone()[0]
        dst = conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
        if dst < src:
            # OR REPLACE: rows can pre-exist after an interrupted split; re-copying
            # the same ids is safe (identical values).
            conn.execute(f'INSERT OR REPLACE INTO {name}({cols}) SELECT {cols} FROM chunks WHERE {where}', params)
        conn.commit()
    conn.execute('DROP TABLE chunks')
    conn.commit()


def slim_table(backend, t) -> None:
    """Phase 2 (per table): rebuild one legacy-shape chunk table into the slim v2
    shape. Single forward pass into a staging table (keyset resume point = staging
    MAX(id)); the swap is one transaction so the old table stays fully readable
    until it commits. The lock is only held for SQL, so searches can interleave
    between batches."""
    conn = backend.conn
    lock = backend.meta._lock
    codec = backend._codec
    staging = f'{t}__new'
    with lock:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {staging}('
            'id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_no INTEGER NOT NULL, '
            "text_z BLOB NOT NULL, chapter_path TEXT NOT NULL DEFAULT '', vector BLOB NOT NULL)"
        )
    while True:
        with lock:
            last_id = conn.execute(f'SELECT COALESCE(MAX(id), 0) FROM {staging}').fetchone()[0]
            rows = conn.execute(
                f'SELECT id, book_id, chunk_no, text, chapter_path, vector FROM {t} WHERE id>? ORDER BY id LIMIT 2000',
                (last_id,),
            ).fetchall()
        if not rows:
            break
        payload = [(i, b, n, codec.compress(tx), cp or '', v) for (i, b, n, tx, cp, v) in rows]
        with lock:
            conn.executemany(
                'INSERT INTO {}(id, book_id, chunk_no, text_z, chapter_path, vector) VALUES(?,?,?,?,?,?)'.format(staging),
                payload,
            )
            conn.commit()
    with lock:
        total = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        done = conn.execute(f'SELECT COUNT(*) FROM {staging}').fetchone()[0]
        if done != total:
            raise RuntimeError(f'chunk migration row count mismatch for {t}: {done}/{total}')
        for _sid, stz, old in conn.execute(f'SELECT s.id, s.text_z, o.text FROM {staging} s JOIN {t} o ON o.id=s.id LIMIT 5'):
            if codec.decompress(stz) != old:
                raise RuntimeError(f'chunk migration round-trip mismatch for {t}')
        conn.execute(f'DROP TABLE {t}')
        conn.execute(f'ALTER TABLE {staging} RENAME TO {t}')
        conn.execute(f'CREATE INDEX idx_{t}_book ON {t}(book_id)')
        conn.commit()
