'''Whole-database schema migrations, one self-contained module per version step.

v1.py  legacy meta tables -> current shape (registries, file_info,
       integer-second timestamps); runs synchronously on open.
v2.py  single `chunks` table -> slim per-model chunk tables; runs deferred
       in VectorStore.finalize_schema before indexing starts.

Each upgrade() brings the whole database from one version to the next (touching
exactly the parts that differ between those versions) and is structural and
resumable, so an interrupted migration simply re-runs the same step. store.py
imports these at its call sites (function-local) to keep the import graph acyclic:
the modules here import pure helpers back from ..store.
'''
