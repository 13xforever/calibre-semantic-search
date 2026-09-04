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

- calibre **8.x dev** (this plugin targets the current development tree; it
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
   python -m zipfile -c semantic_search.zip `
     __init__.py gui.py dialog.py store.py chunker.py indexer.py `
     embed_client.py attributes.py config_widget.py utils.py `
      plugin-import-name-semantic_search.txt semantic_search.png `
      semantic_search-for-light-theme.png semantic_search-for-dark-theme.png
    ```

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
  sqlite backend) chunk text + vectors.
- The **lancedb** backend keeps vectors in a sibling LanceDB directory
  (`semantic-search-lancedb/`), one table per embedding model; bookkeeping
  stays in SQLite either way.
- Attributes are also written to calibre custom columns, so they survive even
  if the store file is deleted.

## Tests

```powershell
python -m unittest discover -s tests
```

(39 tests: chunker, store roundtrip/search/dirty queue, embed client against a
mock HTTP server, settings persistence, attribute extraction with a fake LLM.)
