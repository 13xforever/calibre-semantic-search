'''Schema migration v4: failed-book records -> dedicated table.

Brings the database from version 3 to version 4: the two JSON blobs that tracked
failed books in the meta table ('failed' for indexing failures, 'attr_failed' for
attribute-extraction failures) are moved into the `failed` table (one row per
book and failure kind), and the keys are deleted. The table itself is created by
META_SCHEMA on every open, so this step only moves data and flips user_version.

Runs deferred in VectorStore.finalize_schema, at the end of the 'schema' stage
after the chunk migrations: the version bump must land after v3 has settled the
blob format, because v3's pending check and the blob-width derivation both gate
on user_version >= 3. The GUI starts indexing only after finalize completes, so
no write path sees the half-migrated state (the records are still readable from
meta until this step deletes the keys).

One committed transaction (insert rows, delete keys, flip the version): a crash
either leaves everything as it was or finds user_version at 4, so no progress
marker is needed. Malformed legacy entries are skipped rather than failing the
migration.
'''

from __future__ import annotations

import json

# This step brings v3 -> v4. Later steps (v5) bump further, so target a fixed
# version rather than store.SCHEMA_VERSION to avoid re-running on newer DBs.
_TARGET = 4


def pending(meta) -> bool:
    """True when failed-book records still live in the meta table."""
    with meta._lock:
        return meta.conn.execute('PRAGMA user_version').fetchone()[0] < _TARGET


def _parse_entries(raw):
    """A legacy meta JSON blob -> {book_id: error}; malformed entries are skipped.

    The blobs mapped str(book_id) to {'error': str, 'at': float}; only the book id
    and the error survive the move (the timestamp was never read)."""
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    out = {}
    if not isinstance(data, dict):
        return out
    for key, info in data.items():
        try:
            bid = int(key)
        except (TypeError, ValueError):
            continue
        error = info.get('error') if isinstance(info, dict) else None
        if error is not None:
            out[bid] = str(error)
    return out


def upgrade(meta) -> bool:
    """Move the failed-book JSON blobs into the `failed` table and flip user_version.

    One committed transaction; returns True when the version was flipped."""
    with meta._lock:
        conn = meta.conn
        if conn.execute('PRAGMA user_version').fetchone()[0] >= _TARGET:
            return False
        rows = []
        for key, kind in (('failed', 'index'), ('attr_failed', 'attr')):
            for bid, error in _parse_entries(meta._meta_value_locked(key)).items():
                rows.append((bid, kind, error))
        if rows:
            conn.executemany(
                'INSERT INTO failed(book_id, kind, error) VALUES(?,?,?) '
                'ON CONFLICT(book_id, kind) DO UPDATE SET error=excluded.error',
                rows,
            )
        conn.execute("DELETE FROM meta WHERE key IN ('failed', 'attr_failed')")
        conn.execute(f'PRAGMA user_version={_TARGET}')
        conn.commit()
    return True
