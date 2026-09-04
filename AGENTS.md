# AGENTS.md

calibre **GUI plugin** (not a standalone app): meaning-based search + LLM-extracted book attributes. No build system, no package manifest, no linter/typechecker — verification is `py_compile` + the stdlib unittest suite.

## Commands (run from repo root)

- Tests (all): `python -m unittest discover -s tests`
- One file: `python -m unittest discover -s tests -p "test_store.py"`
- Compile check: `python -m py_compile __init__.py gui.py dialog.py store.py chunker.py indexer.py embed_client.py attributes.py config_widget.py utils.py`
- Build the installable ZIP (files must sit at the **ZIP root**, no subfolder prefix; the import-name `.txt` and `.png` are required):
  ```powershell
  python -m zipfile -c semantic_search.zip `
    __init__.py gui.py dialog.py store.py chunker.py indexer.py `
    embed_client.py attributes.py config_widget.py utils.py `
    plugin-import-name-semantic_search.txt semantic_search.png
  ```

**After any change to a shipped file**: run the full test suite + compile check, and rebuild `semantic_search.zip` so it always matches the working tree. Don't hand off a task with failing tests or a stale ZIP.

## Testing gotchas (easy to get wrong)

- `tests/` is **not** a package (no `__init__.py`). `python -m unittest tests.test_x` fails — always use `discover -s tests [-p file]`.
- Modules are loaded in tests via `tests/util.py:load(name)` (by filename, no package). That only works for self-contained modules (`store`, `chunker`, `utils`, `attributes`, `embed_client`).
- `indexer.py` and `gui.py` use **relative imports** (`from .store import ...`), so they can't be loaded with the plain helper. Load them as a synthetic package — see the `_loadpkg` pattern in `tests/test_indexer.py`. Copy that approach for new indexer/gui tests.
- The LLM/embedding paths are tested with fakes (no network). `calibre.*` imports are unavailable in the test interpreter, so any `calibre.ai` / `calibre.db` import must stay **inside functions** (lazy), never at module top level — a top-level calibre import breaks the whole suite.

## Architecture (not obvious from filenames)

- Entry chain: `__init__.py` (`SemanticSearch`, `InterfaceActionBase`) → `actual_plugin = 'calibre_plugins.semantic_search.gui:SemanticSearchAction'`. Real GUI lives in `gui.py`; installed as `calibre_plugins/semantic_search/...`.
- **Two-model pipeline** in `indexer.py`: the worker drains the embedding (dirty) queue first, then runs an attribute-extraction phase — deliberately batched so the two models aren't swapped mid-run. Attributes run automatically after indexing (`auto_extract_attributes`, default on) or via the "Extract attributes…" menu. Failed books are recorded in the `attr_failed` meta key and excluded from re-runs until retried.
- **Attribute storage source of truth is the store's `attrs_raw` table** (`store.set_attrs`/`get_attrs`). Calibre custom columns are a *best-effort mirror* only — runtime custom-column creation in calibre's newAPI isn't reliably usable for an immediate `set_field`, so don't "fix" things by writing columns alone.
- Adding a setting touches **three** places: the dataclass in `utils.py` (`Settings`), the key list in `_settings_from_dict`, and the collect method in `config_widget.py`. Settings persist under gprefs key `semantic_search_settings` (`utils.PREF_KEY`).

## Gotchas

- The store opens SQLite with `journal_mode=WAL` (`store.py`). Inspecting `semantic-search.db` directly can show **stale/empty tables** (rows live in `-wal` until a checkpoint). Don't "debug" persistence by eyeballing the raw file — verify via the store API or the *Index status* dialog's `Attributes stored: X/Y books` line.
- The plugin targets calibre 8.x dev and uses `calibre.ai` structured-output + newAPI internals. A sibling calibre source checkout (e.g. `E:\Work\git\calibre`) is useful for reading those APIs; the plugin itself ships only the files in the ZIP command above.
