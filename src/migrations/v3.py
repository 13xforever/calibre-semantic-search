'''Schema migration v3: float32 -> float16 vector storage (sqlite only).

Brings the chunk tables from version 2 to version 3: each per-model table is
rebuilt into a staging copy with its vectors quantized from IEEE binary32 to
binary16 (round-to-nearest-even, byte-identical to numpy's '<f2' cast), then the
staging table atomically replaces the original. The table shape is unchanged -
the format lives in the blob width and user_version (>= 3 means f16) - so a table
converted while its siblings are not simply reports half its dimension and is
skipped by search's per-table dim check until the final commit flips it.

Vectors that fail a finiteness check (inf/nan, e.g. from truncated or corrupt
blobs) are dropped: every chunk of an affected book is skipped during the copy,
the book is queued for re-indexing (dirty, reason 'reindex'), and any of its rows
already in staging are deleted. The books registry row is left in place; the
re-index clears and rebuilds it.

Resumable: meta['vec_migrate'] records converted tables and known-bad books,
written before work starts and updated per batch in the same transaction as the
rows; a crash resumes where it stopped.
While running, WAL auto-checkpointing is disabled and the WAL is checkpointed
explicitly between tables. The final commit flips user_version to 3 and deletes
the marker in one transaction, so the f16 format only ever becomes visible
atomically.

Runs deferred in VectorStore.finalize_schema (background thread) before indexing
starts, chained after migrations.v2 by SqliteVectorBackend.finalize.
'''

from __future__ import annotations

import json
import time

from ..store import VEC_MIGRATE_KEY, blob_to_vec, np, vec_to_blob


def pending(backend) -> bool:
    """True when chunk vectors are still stored as float32 (user_version < 3).

    A dataless DB is left alone: it will write f32 until its first rows exist, at
    which point this becomes true and the conversion runs with them."""
    with backend.meta._lock:
        conn = backend.conn
        if conn.execute('PRAGMA user_version').fetchone()[0] >= 3:
            return False
        return any(
            r[0].startswith('chunks_') for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        )


def upgrade(backend, say=None) -> bool:
    """Convert every chunk table to half-float vectors and flip user_version to 3."""
    conn = backend.conn
    lock = backend.meta._lock
    with lock:
        uv = conn.execute('PRAGMA user_version').fetchone()[0]
    if uv >= 3:
        return False
    prog = _load_marker(backend)
    if prog is None:
        prog = {'done': [], 'bad': [], 'last': 0}
        # mark in-flight BEFORE the first row moves: a crash from here on must
        # resume the conversion, never re-decide it from scratch
        backend.meta.set_meta(VEC_MIGRATE_KEY, json.dumps(prog))
    with lock:
        prev_ac = conn.execute('PRAGMA wal_autocheckpoint').fetchone()[0]
        conn.execute('PRAGMA wal_autocheckpoint=0')
    try:
        while True:
            with lock:
                tables = sorted(
                    r[0]
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if r[0].startswith('chunks_') and not r[0].endswith('__new')
                )
            t = next((x for x in tables if x not in prog['done']), None)
            if t is None:
                break
            _convert_table(backend, t, prog, say)
            with lock:
                backend.meta.wal_checkpoint_truncate()
    finally:
        with lock:
            conn.execute(f'PRAGMA wal_autocheckpoint={prev_ac}')
    # one transaction: the f16 format and the marker's disappearance become visible
    # together, so readers never see a mixed-width DB without an in-flight marker
    with lock:
        conn.execute('PRAGMA user_version=3')
        conn.execute('DELETE FROM meta WHERE key=?', (VEC_MIGRATE_KEY,))
        conn.commit()
    return True


def _convert_table(backend, t, prog, say=None) -> None:
    """Rebuild one chunk table into staging with f16 vectors, then swap atomically.

    Rows of books whose vectors fail the finiteness check are not copied; the book
    is queued for re-indexing and any of its rows already in staging are deleted."""
    conn = backend.conn
    lock = backend.meta._lock
    staging = f'{t}__new'
    bad = set(prog['bad'])
    with lock:
        # a blob whose length is not a multiple of 4 cannot be f32: the table was
        # already swapped in an earlier run (protects against a lost marker)
        row = conn.execute(f'SELECT LENGTH(vector) FROM {t} LIMIT 1').fetchone()
        if row and row[0] and row[0] % 4 != 0:
            prog['done'].append(t)
            _save_marker(conn, prog, bad)
            conn.commit()
            return
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {staging}('
            'id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, chunk_no INTEGER NOT NULL, '
            "text_z BLOB NOT NULL, chapter_path TEXT NOT NULL DEFAULT '', vector BLOB NOT NULL)"
        )
        if bad:  # resume: drop rows of books found bad after a partial conversion
            conn.executemany(f'DELETE FROM {staging} WHERE book_id=?', [(b,) for b in sorted(bad)])
        conn.commit()
    while True:
        with lock:
            # the marker's anchor is authoritative; MAX(staging) only ever raises it
            # (a lost marker after a partial run must re-scan, and OR REPLACE below
            # makes that safe), because bad books' rows are deleted from staging
            last_id = max(
                prog.get('last', 0),
                conn.execute(f'SELECT COALESCE(MAX(id), 0) FROM {staging}').fetchone()[0],
            )
            rows = conn.execute(
                f'SELECT id, book_id, chunk_no, text_z, chapter_path, vector FROM {t} WHERE id>? ORDER BY id LIMIT 2000',
                (last_id,),
            ).fetchall()
        if not rows:
            break
        payload = []
        new_bad = set()
        for i, b, n, tz, cp, v in rows:
            if b in bad:
                continue
            vec = blob_to_vec(v, False)  # f32: user_version flips only in the final commit
            if not _finite(vec):
                new_bad.add(b)
                continue
            payload.append((i, b, n, tz, cp or '', vec_to_blob(vec)))
        last_id = rows[-1][0]
        with lock:
            if payload:
                conn.executemany(
                    f'INSERT OR REPLACE INTO {staging}(id, book_id, chunk_no, text_z, chapter_path, vector) VALUES(?,?,?,?,?,?)',
                    payload,
                )
            if new_bad:
                # after the insert: a batch can hold both good and bad rows of the
                # same book, so the bad books' just-copied rows are removed again
                for b in sorted(new_bad):
                    conn.execute(
                        'INSERT INTO dirty(book_id, reason, added_at) VALUES(?,?,?) '
                        'ON CONFLICT(book_id) DO UPDATE SET reason=excluded.reason, added_at=excluded.added_at',
                        (b, 'reindex', int(time.time())),
                    )
                bad |= new_bad
                conn.executemany(f'DELETE FROM {staging} WHERE book_id=?', [(b,) for b in sorted(new_bad)])
            prog['last'] = last_id
            _save_marker(conn, prog, bad)
            conn.commit()
        if say is not None:
            say('schema', f'{t}: row {last_id}')
    with lock:
        if bad:
            total = conn.execute(
                f'SELECT COUNT(*) FROM {t} WHERE book_id NOT IN ({",".join("?" * len(bad))})', sorted(bad)
            ).fetchone()[0]
        else:
            total = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        done = conn.execute(f'SELECT COUNT(*) FROM {staging}').fetchone()[0]
        if done != total:
            raise RuntimeError(f'v3 migration row count mismatch for {t}: {done}/{total}')
        for _sid, svec, ovec in conn.execute(
            f'SELECT s.id, s.vector, o.vector FROM {staging} s JOIN {t} o ON o.id=s.id LIMIT 5'
        ):
            if not _close(blob_to_vec(svec, True), blob_to_vec(ovec, False)):
                raise RuntimeError(f'v3 migration round-trip mismatch for {t}')
        conn.execute(f'DROP TABLE {t}')
        conn.execute(f'ALTER TABLE {staging} RENAME TO {t}')
        conn.execute(f'CREATE INDEX idx_{t}_book ON {t}(book_id)')
        prog['done'].append(t)
        prog['last'] = 0  # the anchor only ever describes the table in flight
        _save_marker(conn, prog, bad)
        conn.commit()


def _load_marker(backend):
    raw = backend.meta.get_meta(VEC_MIGRATE_KEY)
    if raw is None:
        return None
    try:
        p = json.loads(raw)
    except Exception:
        return None
    if isinstance(p, dict) and isinstance(p.get('done'), list) and isinstance(p.get('bad'), list):
        p['done'] = sorted(set(p['done']))
        p['bad'] = sorted(set(p['bad']))
        if not isinstance(p.get('last'), int):
            p['last'] = 0
        return p
    return None


def _save_marker(conn, prog, bad) -> None:
    conn.execute(
        'INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
        (VEC_MIGRATE_KEY, json.dumps({'done': sorted(set(prog['done'])), 'bad': sorted(bad), 'last': prog.get('last', 0)})),
    )


def _finite(vec) -> bool:
    """True when every component is a finite number (the ingest-time rule, applied to stored data)."""
    if np is not None:
        return bool(np.isfinite(vec).all())
    import math

    return all(math.isfinite(x) for x in vec)


def _close(a, b) -> bool:
    """Element-wise comparison with the half-precision rounding tolerance."""
    if np is not None:
        return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=2 ** -10, atol=2 ** -12))
    return all(abs(x - y) <= 2 ** -10 * max(1.0, abs(y)) for x, y in zip(a, b))
