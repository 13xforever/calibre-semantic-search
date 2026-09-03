'''Background indexing worker: dirty queue -> extract -> chunk -> embed -> store.

Runs as a daemon thread. All calibre/Qt access happens through the injected
`get_new_api()` callable so it works across library switches and is testable.
'''

from __future__ import annotations

import contextlib
import os
import threading
import time
import unicodedata

from .chunker import chunks_from_pages, split_plain_text
from .store import VectorStore


class IndexerError(RuntimeError):
    pass


def _default_log(msg: str) -> None:
    import sys

    print(f'[semantic-search] {msg}', file=sys.stderr)


def extract_book_pages(path: str, fmt: str):
    """Extract spine pages of an OEB-format book as a list of HTML strings.

    Returns (kind, payload): ('pages', [html, ...]) | ('text', plain_str) | ('error', msg).
    """
    from calibre.ebooks.oeb.iterator.book import extract_book
    from calibre.ebooks.oeb.polish.container import Container as ContainerBase
    from calibre.ptempfile import TemporaryDirectory

    class SimpleContainer(ContainerBase):
        tweak_mode = True

    fmt_u = fmt.upper()
    if fmt_u == 'PDF':
        try:
            from calibre.db.fts.text import pdftotext

            return ('text', pdftotext(path))
        except Exception as e:
            return ('error', f'PDF text extraction failed: {e}')
    if fmt_u in {'TXT', 'TEXT'}:
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            for enc in ('utf-8', 'utf-16', 'latin-1'):
                try:
                    return ('text', unicodedata.normalize('NFC', raw.decode(enc)))
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return ('error', 'could not decode text file')
        except Exception as e:
            return ('error', f'text read failed: {e}')

    try:
        with TemporaryDirectory() as tdir:
            book_fmt, opfpath, input_fmt = extract_book(path, tdir, log=_default_log)
            container = SimpleContainer(tdir, opfpath, _default_log)
            pages = []
            for name, is_linear in container.spine_names:
                root = container.parsed(name)
                if hasattr(root, 'xpath'):
                    from lxml import etree

                    pages.append(etree.tostring(root, encoding='unicode'))
            return ('pages', pages)
    except Exception as e:
        return ('error', f'extraction failed: {e}')


def pick_format(formats: dict[str, str], priority: list[str]) -> tuple[str, str] | None:
    """Choose the best (fmt, path) from a book's formats using the priority list."""
    if not formats:
        return None
    by_fmt = {f.upper(): (f, p) for f, p in formats.items()}
    for want in priority:
        hit = by_fmt.get(want.upper())
        if hit:
            return hit
    # fallback: first available extractable-ish format
    for f, p in formats.items():
        if f.upper() not in {'LSF', 'LRF'}:
            return (f, p)
    f, p = next(iter(formats.items()))
    return (f, p)


class Indexer(threading.Thread):
    """Daemon thread draining the store's dirty queue."""

    def __init__(self, store: VectorStore, get_new_api, settings_provider, status_cb=None):
        super().__init__(name='SemanticSearchIndexer', daemon=True)
        self.store = store
        self.get_new_api = get_new_api
        self.settings_provider = settings_provider
        self.status_cb = status_cb or (lambda s: None)
        self.stop_event = threading.Event()
        self.current_book_id: int | None = None

    def stop(self):
        self.stop_event.set()

    # -- public -------------------------------------------------------------

    def reconcile(self):
        """Compare library vs store; queue missing/changed books, drop removed ones."""
        import json

        api = self.get_new_api()
        if api is None:
            return
        settings = self.settings_provider()
        lib_ids = set(api.all_book_ids())
        indexed = {b['id']: b for b in self.store.indexed_books()}
        try:
            failed = json.loads(self.store.get_meta('failed', '{}') or '{}')
        except Exception:
            failed = {}
        # remove books that vanished from the library
        for bid, info in list(indexed.items()):
            if bid not in lib_ids:
                self.store.clear_book(bid)
                self.store.remove_dirty(bid)
                self.store.set_meta(f'fileinfo:{bid}', '')
                failed.pop(str(bid), None)
        for bid in lib_ids:
            formats = api.formats(bid)
            pick = pick_format(formats, settings.format_priority)
            if pick is None:
                continue
            fmt, path = pick
            try:
                md = api.format_metadata(bid, fmt)
            except Exception:
                continue
            fi_raw = self.store.get_meta(file_info_key(bid), '') or ''
            parts = fi_raw.split('|')
            info = indexed.get(bid)
            if info is None:
                # never indexed; skip if it failed before and the file is unchanged
                if str(bid) in failed and len(parts) == 3 and parts[0] == fmt and self._same_file(parts, md):
                    continue
                self.store.add_dirty(bid, fmt, 'added')
                continue
            if len(parts) == 3:
                changed = parts[0] != fmt or not self._same_file(parts, md)
            else:
                changed = info['fmt'] != fmt
            if changed:
                self.store.add_dirty(bid, fmt, 'changed')
        self.store.set_meta('failed', json.dumps(failed))

    @staticmethod
    def _same_file(parts: list[str], md: dict) -> bool:
        try:
            return int(parts[1]) == int(md.get('size') or -1) and float(parts[2]) == float(md.get('mtime') or -1)
        except (TypeError, ValueError):
            return False

    # -- worker loop ----------------------------------------------------------

    def run(self):
        while not self.stop_event.is_set():
            try:
                pending = self.store.dirty_book_ids()
                if not pending:
                    self.status_cb({'state': 'idle'})
                    self.stop_event.wait(2)
                    continue
                self._process_one(pending[0])
            except Exception as e:
                _default_log(f'indexer loop error: {e!r}')
                self.stop_event.wait(5)

    def _status(self, state, book_id=None, **kw):
        d = {'state': state}
        if book_id is not None:
            d['book_id'] = book_id
        d.update(kw)
        try:
            self.status_cb(d)
        except Exception:
            pass

    def _process_one(self, book_id: int):
        api = self.get_new_api()
        if api is None:
            return
        settings = self.settings_provider()
        formats = api.formats(book_id)
        if not formats:
            self.store.clear_book(book_id)
            self.store.remove_dirty(book_id)
            return
        pick = pick_format(formats, settings.format_priority)
        if pick is None:
            self.store.remove_dirty(book_id)
            return
        fmt, path = pick
        try:
            md = api.format_metadata(book_id, fmt)
        except Exception:
            md = {'size': None, 'mtime': None}

        self.current_book_id = book_id
        self._status('extracting', book_id, fmt=fmt)
        kind, payload = extract_book_pages(path, fmt)
        if kind == 'error':
            self._fail(book_id, payload)
            return
        if kind == 'pages' and not any((p or '').strip() for p in payload):
            self._status('done', book_id, note='no text found')
            self.store.clear_book(book_id)
            self.store.upsert_book(book_id, fmt, 0, settings.embed.model, 0)
            self.store.remove_dirty(book_id)
            return

        if kind == 'pages':
            chunks = chunks_from_pages(payload, settings.target_chars, settings.overlap_chars)
        else:
            chunks = split_plain_text(payload or '', settings.target_chars, settings.overlap_chars)
        if settings.max_chunks_per_book > 0:
            chunks = chunks[: settings.max_chunks_per_book]
        if not chunks:
            self.store.clear_book(book_id)
            self.store.upsert_book(book_id, fmt, 0, settings.embed.model, 0)
            self.store.remove_dirty(book_id)
            return

        from .embed_client import EmbedClient

        client = EmbedClient(
            base_url=settings.embed.base_url, model=settings.embed.model, api_key=settings.embed.api_key, timeout=settings.embed.timeout
        )
        texts = [c.text for c in chunks]
        done = 0

        def progress(d, total):
            nonlocal done
            done = d
            self._status('embedding', book_id, done=d, total=total)

        try:
            vectors = client.embed_batched(
                texts, batch_size=settings.embed.batch_size, concurrency=settings.embed.concurrency, progress=progress
            )
        except Exception as e:
            self._fail(book_id, str(e))
            return

        if len(vectors) != len(chunks):
            self._fail(book_id, f'embedding count mismatch: {len(vectors)} vs {len(chunks)}')
            return
        dim = len(vectors[0])
        self._status('saving', book_id)
        from .store import l2_normalize

        self.store.clear_book(book_id)
        for c, v in zip(chunks, vectors):
            self.store.insert_chunk(
                book_id, c, c.text, c.chapter_path, c.para_start, c.para_end, c.char_offset, settings.embed.model, dim, l2_normalize(v)
            )
        self.store.commit()
        self.store.upsert_book(book_id, fmt, len(chunks), settings.embed.model, dim)
        # record size/mtime for change detection
        self.store.set_meta(file_info_key(book_id), f'{fmt}|{md.get("size")}|{md.get("mtime")}')
        import json

        try:
            failed = json.loads(self.store.get_meta('failed', '{}') or '{}')
        except Exception:
            failed = {}
        if str(book_id) in failed:
            del failed[str(book_id)]
            self.store.set_meta('failed', json.dumps(failed))
        self.store.remove_dirty(book_id)
        self.current_book_id = None
        self._status('done', book_id, n_chunks=len(chunks))

    def _fail(self, book_id: int, msg: str):
        import json

        api = self.get_new_api()
        settings = self.settings_provider()
        formats = api.formats(book_id) if api is not None else {}
        pick = pick_format(formats, settings.format_priority) if formats else None
        if pick is not None and api is not None:
            try:
                md = api.format_metadata(book_id, pick[0])
                self.store.set_meta(file_info_key(book_id), f'{pick[0]}|{md.get("size")}|{md.get("mtime")}')
            except Exception:
                pass
        failed = {}
        try:
            failed = json.loads(self.store.get_meta('failed', '{}') or '{}')
        except Exception:
            failed = {}
        failed[str(book_id)] = {'error': msg, 'at': time.time()}
        self.store.set_meta('failed', json.dumps(failed))
        self.store.remove_dirty(book_id)
        self.current_book_id = None
        self._status('error', book_id, error=msg)


def file_info_key(book_id: int) -> str:
    return f'fileinfo:{book_id}'
