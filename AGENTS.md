# AGENTS.md

calibre **GUI plugin** (not a standalone app): meaning-based search + LLM-extracted book attributes. No build system, no typechecker — verification is ruff (pyflakes + isort) + `py_compile` + the stdlib unittest suite, all driven by `dev.py`. The installable ZIP mirrors `src/` exactly; dev dependencies live in `requirements-dev.txt` and are never shipped.

## Environment

- Dev environment is a project-local **`.venv`**: `python dev.py setup` creates it from the closest installed Python to what calibre ships (highest in 3.12..3.14 — see `_pick_interpreter`) and installs missing dev deps from `requirements-dev.txt` (the file's comments explain each pin). Re-running is a no-op when complete; nothing is installed system-wide; setup does NOT run the gate.
- Run everything through the venv interpreter: `.venv\Scripts\python dev.py ...`. The code uses nothing newer than Python 3.9.

## Commands (repo root, via `.venv\Scripts\python dev.py ...`)

- `setup` — create/refresh .venv + install missing dev deps (no tests)
- no args — full gate: compile + lint + test + build, stop on first failure
- `compile` / `lint` / `test` / `build` — individual steps (see the `dev.py` docstring)
- One test file: `.venv\Scripts\python -m unittest discover -s tests -p "test_store.py"`

**After any change to a shipped file**: run the full gate so tests pass and `semantic_search.zip` matches the working tree. Don't hand off with failing tests or a stale ZIP.

## Testing gotchas (easy to get wrong)

- `tests/` is **not** a package — always use `discover -s tests [-p file]`, never `unittest tests.test_x`.
- Self-contained modules load via `tests/util.py:load(name)` (by filename, no package). Modules with relative imports (`store`, `indexer`, `gui`, `dialog`, `config_widget`, `attributes`) must be loaded as a synthetic package with the `_loadpkg` pattern — see `tests/test_indexer.py`.
- `calibre.*` is unavailable in the test interpreter: any `calibre.ai` / `calibre.db` import must stay **inside functions**, never at module top level.
- The lancedb backend tests are **not skipped** — `lancedb` is a hard test dependency; without it the suite fails loudly on purpose. Don't wrap them in `skipUnless`.
- Simulating a *missing* optional package means patching `store._module_available` (the real module stays importable, mirroring an in-session uninstall where the loaded module still works while fresh opens must block). See `test_db_lifecycle.TestBlockedStates` / `TestCodecRecompress`.
- The chunker tests run **twice** in the gate: under the venv's PyPI lxml, and — when `calibre-debug` is found (via `.env` with `CALIBRE_DEBUG=path`, then PATH) — under calibre's Python, because the two bundle different libxml2 builds for the same pinned lxml version and `chunker.parse_page` must work on both. The calibre run is skipped with a note when unavailable; don't remove it or wrap it in `skipUnless`.

## Architecture

The module docstrings are the first stop for design context — especially `store.py`'s (per-library backend/codec choice, finalize stages, pre-flight) and `indexer.py`'s. Only the non-obvious parts are listed here:

- Entry chain: `__init__.py` → `actual_plugin = 'calibre_plugins.semantic_search.gui:SemanticSearchAction'`; real GUI in `gui.py`.
- **Asset loading** (`gui._plugin_icon`, `store._load_default_dict`): installed plugins load from the ZIP with a *virtual* `__file__`, so these accessors try the zip-aware `get_icons`/`get_resources` first and fall back to plain files. Don't "simplify" to filesystem-only — that silently breaks installed mode.
- **Attributes**: source of truth is the store's `attrs_raw` table; calibre custom columns are a *best-effort mirror* (runtime column creation in newAPI isn't reliably usable for an immediate `set_field`). Don't "fix" attribute storage by writing columns alone. The indexer batches the embedding and attribute phases so the two models aren't swapped mid-run; failed books sit in the store's `failed` table (one row per book and failure kind, `'index'`/`'attr'`) until retried.
- **Adding a setting touches three places**: the `Settings` dataclass (`utils.py`), the key list in `_settings_from_dict`, and the collect method in `config_widget.py`. Persisted under gprefs key `semantic_search_settings` (`utils.PREF_KEY`).
- **Store schema** (`store.py`): meta tables versioned via `PRAGMA user_version`; one self-contained module per step in `migrations/`. The heavy chunk-table migration is deferred to `VectorStore.finalize_schema()` (ordered stages `'schema'` → `'backend'` → `'codec'` → `'index'`, see `pending_stages()`), run in a background thread before indexing starts. Every stage is gated by meta markers (`backend_migrate`, `recompress`, `vec_migrate`) written **before** work begins, so a crash mid-transfer never loses data; the `'codec'` stage is all-or-nothing (one transaction, single end commit — fast on slow disks; a crash/cancel re-runs it from scratch), its read/update batches are sized against available RAM like the search (a whole table in one round trip when there is headroom), and the others resume in place; orphaned storage from an abandoned switch is swept on open only when no marker exists. The final `VACUUM` runs only when a stage freed pages (schema rebuilds, backend transfer) — an in-place codec-only run frees nothing and already drains its own WAL. Vector blobs are half floats in v3+ sqlite (width derived from `user_version`, never stored per row) and lancedb always stores halves; the `'index'` stage builds/maintains the lancedb IVF_HNSW_SQ index, whose state lives in the dataset manifest rather than a meta marker.
- **Per-library storage choices**: the `vector_backend` and `text_codec` meta keys are the per-library source of truth; global settings only seed brand-new libraries. lancedb stores plain text — zstd/zlib compression is sqlite-only. `ensure_codec_setup()` self-heals unusable codec records on open so a dataless library never blocks on a missing package, and detects a shipped default-dictionary change (stored zstd dict ≠ current asset) by writing the `recompress` marker — an in-flight conversion's marker is never clobbered.
- **Dependency pre-flight**: opening a store whose *stored data* needs a missing package raises `store.MissingDependencyError` (the has-chunks guard keeps dataless libraries unblocked). A pending switch *to* an unavailable backend does not block the open — only where the data actually is matters; the migration then fails with the same error when attempted. The GUI catches this into a blocked state (search/indexing disabled, storage combos locked while `has_data`); confirming settings retries the open, so a just-installed package unblocks the library without switching libraries.
- **When a new migration step is added** (a sqlite `migrations/vN.py` or a lancedb equivalent), update `tests/test_db_lifecycle.py`: it builds small reproducible DBs per scenario (legacy v0 migrated to latest, fresh sqlite at latest, lancedb) and runs the same full operation suite on each — the new step must be covered there too.
- **Chunker inline-run collection is load-bearing** (`chunker.html_to_units`): some real books (e.g. Project Gutenberg MOBIs) wrap their whole text in non-block tags like `<widger>` with `<br/>` line breaks and no `<p>` elements — without the inline-run fallback such books index as 0 chunks and are silently recorded as done. Don't "simplify" `walk()` back to block-tags-only; see `tests/test_chunker.py::TestInlineRuns`.

## Gotchas

- **Never delete files outside the scratch dir** (`%TEMP%\opencode`) without asking first — including test/benchmark artifacts like migrated DB copies on other drives. Cleaning up the scratch dir is fine; anything else needs explicit permission.
- **Benchmark with real parameters, never invented ones**: the target embedding dim is **4096**. Any perf/memory estimate (numpy vs pure-Python scoring, RAM-budget batch sizes, storage) must use that dim and realistic chunk counts — results scale linearly with dim, so a made-up dim produces wrong conclusions.
- Use **binary units** for all computer measurements: kB/MB/GB mean powers of 1024 (1 kB = 1024 B). Powers of ten are confusing in programming contexts — don't use them for byte counts.
- In WAL mode `PRAGMA auto_vacuum=INCREMENTAL` only takes effect **through a finished `VACUUM`** — that's why new DBs and migrations end with the pair, and why `needs_finalize()` treats `auto_vacuum != 2` as "previous VACUUM never completed" and re-runs it.
- **File mtimes are truncated to integer seconds, never rounded** (`migrations/v1.py`, `indexer._mtime_to_int`): rounding up would store a later mtime than the file's real one, making every book look modified.
- **Calibre's in-place plugin reload does NOT re-import submodules** (`calibre/customize/zipplugin.py`): Preferences → Plugins re-runs only `__init__.py`; already-loaded submodules stay stale until calibre restarts (replace the installed ZIP and restart). Never do a hard top-level `from .submodule import name` in `__init__.py` — with a stale submodule it raises `ImportError` and crashes the whole plugin load. Import the module and guard with `getattr` (see `_bootstrap_external_deps`).
- **Calibre's Qt binding marshals `QTableView.currentCellChanged` as four ints** `(row, col, prev_row, prev_col)`, NOT the stock-PyQt `(QModelIndex, QModelIndex)` — a slot's first arg is an `int`, so `.row()` on it raises. Handle both shapes (`hasattr(current, 'row')`); see `_ResultsTable._select_current_row`. Verify Qt signal/slot signatures by probing under `calibre-debug` with `QT_QPA_PLATFORM=offscreen`, not against stock-PyQt docs.
- The plugin targets calibre 9.x dev and uses `calibre.ai` structured-output + newAPI internals; a local calibre source checkout is useful for reading those APIs.
