# Disclaimer
> [!WARNING] 
> Everything here was written by Qwen3.8, do not trust it to keep your data safe.
> 
> This is purely a test/benchmark project, not intended for serious use.

# Semantic Search (calibre plugin)

Meaning-based search across your calibre library, with LLM-extracted book
attributes in custom columns.

- Books are chunked into paragraph groups that keep chapter context.
- Chunks are embedded via any **OpenAI-compatible** embedding server
  (Ollama, LM Studio, Unsloth, vLLM, ...).
- The search dialog ranks results by vector similarity and can open the
  matching passage in the calibre viewer.
- An optional LLM pass extracts configurable attributes (protagonist gender,
  orientation, romance elements, tropes, themes, POV, content warnings, ...)
  into **calibre custom columns**, so you can combine them with normal
  metadata searches and virtual libraries.

## Requirements

- calibre **9.x dev** (this plugin targets the current development tree; it
  uses `calibre.customize` base classes and the AI structured-output API).
- A local OpenAI-compatible embedding server, e.g. Ollama:
  - `ollama pull nomic-embed-text` (or any model you like)
  - default URL: `http://localhost:11434`
- Optional: a text-to-text AI provider configured under *Preferences > Plugins
  > AI Provider* (e.g. OpenAI compatible) for attribute extraction.
- Optional: the `lancedb` Python package if you want the LanceDB vector
  backend. The default SQLite backend needs nothing extra; numpy is used
  automatically if present. You can install lancedb yourself (`pip install
  lancedb`, into calibre's Python) or from inside the plugin: *Settings >
  Indexing* shows an **Install lancedb...** button that runs pip for you and
  falls back to a `--user` install on permission errors. Restart calibre after
  installing.

## Install

1. Build the ZIP (from this directory):

    ```powershell
    python dev.py build
    ```

    (`python dev.py` with no arguments runs the full gate: compile check, test
    suite, then the ZIP build.)

2. In calibre: *Preferences > Plugins > Custom plugins > `+`* and pick the ZIP.
3. Open the new **Semantic search** menu (toolbar dropdown or the icon in the
   search bar).

## Usage

1. **Settings** — point *Embeddings* at your server URL + model name, choose
   chunk size / format priority on the *Indexing* tab, and set both context
   limits (tokens): the **embedding** model's max input (caps chunk size) and
   the attribute model's window (sizes the sample/fulltext groups). Edit the
   attribute schema on the *Attributes* tab (add/remove fields, toggle them,
   `text` vs multi-value `tags`). New `tags` fields become multi-value custom
   columns (`#ss_*`).
2. On first start (and whenever books are added/changed) the indexer queues
   books automatically: extract -> chunk -> embed -> store. Watch progress in
   *Index status*.
3. **Search...** — type a natural-language query, e.g. *"a slow-burn romance
   between enemies"*. Double-click a row (or *Open in viewer*) to jump to the
   matching passage; *Restrict library to results* filters the library view to
   the matched books.
4. **Extract attributes...** — runs the LLM pass over indexed books that lack
   stored attributes and writes them to the custom columns. Use
   *Re-index all books* after changing the embedding model or chunk settings.

## Data & storage

- Per-library SQLite file `semantic-search.db` next to `metadata.db` holds
  book registry, dirty queue, metadata, raw attribute JSON, and (for the
  sqlite backend) chunk text + vectors — one table per embedding model, so
   different models never mix. Chunk text in the sqlite backend is stored
   compressed — zstd (with a bundled compression dictionary) or zlib, chosen per
   library in Settings and converted in the background when changed — which keeps
   the file noticeably smaller than the raw text would be. Search reads vectors
  in RAM-budgeted batches (half of the free RAM) instead of loading the whole
  index.
- Databases created by older versions migrate automatically on first start
  after upgrading. On large libraries this one-time migration can take a
  while; *Index status* shows "Migrating search database" until it finishes,
  and indexing starts only afterwards. The migration is resumable if calibre
  is closed mid-way. While it runs, intermediate data is flushed to the main
  file at phase boundaries, so the temporary WAL growth stays bounded even on
  slow disks.
- The **lancedb** backend keeps vectors in a hidden sibling LanceDB directory
  (`.semantic-search.lancedb`, also marked with the hidden attribute on
  Windows), one table per embedding model; bookkeeping stays in SQLite either
  way.
- Chunk tables left over from an old embedding model are dropped automatically
  once no indexed book uses that model anymore (after re-indexing / removal).
- Attributes are also written to calibre custom columns, so they survive even
  if the store file is deleted.

## Development environment

To run the test suite / build from a checkout on a new system:

1. A Python between **3.12 and 3.14** installed (check with `py --list`).
   Setup picks the one closest to what calibre ships — the highest version at or
   below 3.14 (calibre's current build runs CPython 3.14; staying at or below it
   is the safe direction). The code itself uses nothing newer than Python 3.9.
2. One command to provision a project-local virtual environment:

    ```powershell
    python dev.py setup
    ```

   It creates `.venv/` from that interpreter (reusing it if already present) and
   pip-installs any missing dev dependencies from `requirements-dev.txt` into the
   venv — and only those, so re-running on a complete machine does nothing.
   Nothing is installed system-wide.

3. Run everything through the venv's interpreter:

    ```powershell
    .\.venv\Scripts\python dev.py
    ```

   (or activate `.venv` first and use `python dev.py` as usual)

What each dev dependency is for:

- `ruff` — the lint step of the gate (pyflakes + isort rules; config in
  `ruff.toml`).
- `lancedb` — hard **test** dependency: the lancedb backend tests must run, not
  skip.
- `lxml==6.1.1` — the exact version calibre 9.14 bundles; the chunker's HTML
  walk runs against it in tests just as it does inside calibre. Test-only:
  production gets lxml from calibre itself.
- `numpy` — makes vector search in the sqlite backend ~36x faster. Optional at
  runtime (a pure-Python fallback exists) but calibre does not bundle it, so it
  is installed on demand from the Settings dialog (Dependencies tab); in this
  venv it is also needed to exercise the numpy code paths in tests (it arrives
  via lancedb's own dependencies too).
- `zstandard` — the zstd text codec; without it new stores fall back to zlib
  (a library can be converted to zstd later from Settings).

The plugin itself needs none of these to *run* inside calibre: it ships exactly
`src/` (no manifest) and uses the stdlib plus lxml (bundled by calibre). numpy
is optional — without it search falls back to a slower pure-Python path. lancedb
and zstandard are optional at runtime too; the Settings dialog's Dependencies tab
can install or uninstall them on demand.

Individual steps (same venv interpreter):
`.venv\Scripts\python dev.py compile | lint | test | build`.

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests
```

(or `.venv\Scripts\python dev.py test`; `dev.py` with no arguments runs compile
+ lint + test + ZIP build.)

(144 tests: chunker, store roundtrip/search/per-model tables/dirty queue, embed
client against a mock HTTP server, indexer phases/reconcile, settings
persistence, attribute extraction with a fake LLM, dialog behavior, plus full
store-lifecycle suites on three reproducible databases — legacy v0 migrated to
latest, fresh sqlite at latest, and lancedb.)
