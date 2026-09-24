'''LanceDB-side migration v1: repair oversized IVF_HNSW_SQ partitions.

lancedb 0.38 / lance 11 remap an existing vector index during optimize() with Arrow
fixed-size-list child indices (u32), so a partition whose rows * dimension exceeds
2**32 - 1 aborts the compaction with a RustPanic (lancedb#2866; the upstream fix is
unmerged). Indexes built by older plugin versions targeted ~1M rows per partition,
which overflows at common embedding dimensions such as 4096.

The dataset manifest records no usable partition count, so every index build also
records its rows/partition target in meta[INDEX_TARGET_PREFIX + table]. This step
repairs tables whose recorded target (or, for legacy indexes without a record, the
target the old build formula produced) overflows the limit: drop the index, compact,
rebuild with safe partitions. Resumable per table — an interrupted repair leaves the
table without an index and the normal build path finishes it; a crash in the small
window after the rebuild but before its marker write simply re-runs this step for
that table once.

Runs as part of the 'index' finalize stage, before routine index maintenance. store.py
imports this module at its call sites (function-local) to keep the import graph
acyclic; it imports pure helpers back from ..store.
'''

from __future__ import annotations

from datetime import timedelta

from ..store import INDEX_TARGET_PREFIX

_U32_LIMIT = 2**32 - 1
_LEGACY_TARGET_ROWS = 1_048_576  # partition target of pre-fix builds


def _target_rows(backend, name: str, n: int):
    """The rows/partition target in force for table `name`."""
    raw = backend.meta.get_meta(INDEX_TARGET_PREFIX + name)
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    # legacy index built before the marker existed: what the old build formula
    # (num_partitions = max(1, n // _LEGACY_TARGET_ROWS)) actually produced
    return n // max(1, n // _LEGACY_TARGET_ROWS)


def _flagged_tables(backend):
    """Yield (name, table, n, dim) for indexed tables whose partitions may overflow."""
    for name in sorted(backend._table_names()):
        t = backend._open_named(name)
        if t is None:
            continue
        try:
            n = t.count_rows()
        except Exception:
            continue
        if not n or backend._our_index(t) is None:
            continue
        dim = t.schema.field('vector').type.list_size
        target = _target_rows(backend, name, n)
        if target is not None and target * dim > _U32_LIMIT:
            yield name, t, n, dim


def needs_upgrade(backend) -> bool:
    """True when any indexed table may hold partitions that overflow the u32 limit."""
    return next(_flagged_tables(backend), None) is not None


def upgrade(backend, say=None):
    """Drop and rebuild every flagged table's index with safe partitions."""
    for name, t, n, _dim in _flagged_tables(backend):
        if say is not None:
            say('index', f'repairing oversized vector index for {name} ({n:,} rows)')
        cfg = backend._our_index(t)
        # drop before compacting: optimize() without an index never remaps, so the
        # rebuild path cannot re-trigger the overflow
        t.drop_index(cfg.name)
        t.optimize()
        t.optimize(cleanup_older_than=timedelta(0))
        backend._create_vector_index(t, n, name)
