'''Schema migration v5: indexing queue -> queue_indexing, add attribute queue.

Brings the database from version 4 to version 5: the legacy `dirty` table (the
indexing queue, still carrying a free-text reason and an added_at timestamp) is
replaced by queue_indexing, whose rows are keyed by a priority tier instead; the
persistent attribute-queue table queue_attrs is created. Both new tables are
created by META_SCHEMA on every open, so this step only moves the legacy dirty
rows (mapping them to the default priority tier and discarding reason/added_at),
drops the old table, and flips user_version.

Runs deferred in VectorStore.finalize_schema, at the end of the 'schema' stage
after v4: indexing has not started yet, so no read/write path sees the queue in
its half-migrated state. One committed transaction (move rows, drop the old
table, flip the version): a crash either leaves everything as it was or finds
user_version at 5, so no progress marker is needed.
'''

from __future__ import annotations

from ..store import PRIORITY_DEFAULT, SCHEMA_VERSION


def _user_version(conn) -> int:
    return int(conn.execute('PRAGMA user_version').fetchone()[0])


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def pending(meta) -> bool:
    """True when the version is still below v5."""
    with meta._lock:
        return _user_version(meta.conn) < SCHEMA_VERSION


def upgrade(meta) -> bool:
    """Move the legacy dirty rows into queue_indexing and flip user_version.

    One committed transaction; returns True when the version was flipped."""
    with meta._lock:
        conn = meta.conn
        if _user_version(conn) >= SCHEMA_VERSION:
            return False
        if _has_table(conn, 'dirty'):
            # Legacy rows are transient in-flight indexing work: reason/added_at are
            # dropped and every row maps to the default priority tier.
            conn.execute(
                'INSERT INTO queue_indexing(book_id, priority) SELECT book_id, ? FROM dirty',
                (PRIORITY_DEFAULT,),
            )
            conn.execute('DROP TABLE dirty')
        conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        conn.commit()
    return True
