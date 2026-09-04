'''Background indexing worker: dirty queue -> extract -> chunk -> embed -> store.

Runs as a daemon thread. All calibre/Qt access happens through the injected
`get_new_api()` callable so it works across library switches and is testable.
'''

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import unicodedata

from .chunker import CONTEXT_OVERHEAD_TOKENS, chunks_from_pages, max_chunk_chars, split_plain_text
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
            book_fmt, opfpath, input_fmt = extract_book(path, tdir)
            container = SimpleContainer(tdir, opfpath)
            pages = []
            for name, is_linear in container.spine_names:
                root = container.parsed(name)
                if hasattr(root, 'xpath'):
                    from lxml import etree

                    pages.append(etree.tostring(root, encoding='unicode'))
            return ('pages', pages)
    except Exception as e:
        return ('error', f'extraction failed: {e}')


def pick_format(formats, priority: list[str]) -> str | None:
    """Choose the best format name from a book's available formats.

    `formats` is the tuple of format names returned by ``new_api.formats()``.
    Returns the chosen format name (string), or None if there are none.
    """
    if not formats:
        return None
    have = {f.upper(): f for f in formats}
    for want in priority:
        hit = have.get(want.upper())
        if hit is not None:
            return hit
    # fallback: first available extractable-ish format (we can't parse LSF/LRF)
    for f in formats:
        if f.upper() not in {'LSF', 'LRF'}:
            return f
    return next(iter(formats))


def _mtime_to_float(v):
    """Normalize an mtime value (number or datetime) to a float epoch timestamp."""
    if v is None:
        return None
    ts = getattr(v, 'timestamp', None)
    if callable(ts):
        try:
            return float(ts())
        except Exception:
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime

        return float(datetime.fromisoformat(str(v)).timestamp())
    except Exception:
        return None


def file_info_value(fmt: str, md: dict) -> str:
    """Serialize (fmt, size, mtime) for change detection; fmt-only if stat info is missing."""
    try:
        size = int(md.get('size'))
    except (TypeError, ValueError):
        size = None
    mtime = _mtime_to_float(md.get('mtime'))
    if size is not None and mtime is not None:
        return f'{fmt}|{size}|{mtime}'
    return fmt


class Indexer(threading.Thread):
    """Daemon thread draining the store's dirty queue."""

    def __init__(self, store: VectorStore, get_new_api, settings_provider, status_cb=None, attr_writer=None, attr_done_cb=None):
        super().__init__(name='SemanticSearchIndexer', daemon=True)
        self.store = store
        self.get_new_api = get_new_api
        self.settings_provider = settings_provider
        self.status_cb = status_cb or (lambda s: None)
        self.attr_writer = attr_writer  # GUI-thread-marshalled new_api for the custom-column mirror
        self.attr_done_cb = attr_done_cb or (lambda payload: None)
        self.stop_event = threading.Event()
        self.current_book_id: int | None = None
        self._reconcile_lock = threading.Lock()
        self._attr_requested = threading.Event()
        self._paused = threading.Event()

    def stop(self):
        self.stop_event.set()

    def pause(self):
        """Suspend the worker loop; queued work is left untouched."""
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # -- attribute extraction phase ---------------------------------------------

    def request_attributes(self):
        """Ask the worker loop to run the attribute-extraction phase."""
        self._attr_requested.set()

    def _pending_attr_books(self, settings) -> list[int]:
        from .attributes import pending_attribute_books

        try:
            failed = set(json.loads(self.store.get_meta('attr_failed', '{}') or '{}').keys())
        except Exception:
            failed = set()
        return [b for b in pending_attribute_books(self.store, settings) if str(b) not in failed]

    def _set_attr_failed(self, book_id: int, error: str | None):
        try:
            failed = json.loads(self.store.get_meta('attr_failed', '{}') or '{}')
        except Exception:
            failed = {}
        key = str(book_id)
        if error is None:
            failed.pop(key, None)
        else:
            failed[key] = {'error': error, 'at': time.time()}
        self.store.set_meta('attr_failed', json.dumps(failed))

    def _process_attributes(self, pending: list[int], settings, llm=None):
        """Extract attributes for every book in `pending`, reporting live progress.

        Runs only after the indexing (embedding) queue is empty, so all embedding
        work is batched before any LLM calls (the two use different models).
        Returns True if the phase ran to completion, False if it was interrupted
        by a pause or shutdown (remaining books are picked up on a later pass).
        """
        if llm is None:
            try:
                from calibre.ai import AICapabilities
                from calibre.ai.prefs import plugin_for_purpose

                llm = plugin_for_purpose(AICapabilities.text_to_text)
            except Exception as e:
                self._status('attr_error', error=f'AI provider unavailable: {e}')
                return True
        if llm is None:
            self._status('attr_error', error='no text-to-text AI provider configured')
            return True

        from .attributes import extract_book_attributes

        total = len(pending)
        errors: list[tuple[int, str]] = []
        for i, bid in enumerate(pending):
            if self.stop_event.is_set() or self._paused.is_set():
                return False  # interrupted; the phase resumes on a later pass
            self._status('attributes', bid, done=i + 1, total=total)
            try:
                extract_book_attributes(bid, self.attr_writer, self.store, settings, llm=llm)
                self._set_attr_failed(bid, None)
            except Exception as e:
                self._set_attr_failed(bid, repr(e))
                errors.append((bid, repr(e)))
        self._status('attributes_done', done=total, total=total)
        self.attr_done_cb((total, errors))
        return True

    # -- public -------------------------------------------------------------

    def reconcile(self):
        """Compare library vs store; queue missing/changed books, drop removed ones."""
        with self._reconcile_lock:
            self._reconcile_locked()

    def _reconcile_locked(self):
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
            fmt = pick_format(formats, settings.format_priority)
            if fmt is None:
                continue
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
            size = int(parts[1])
            cur_size = int(md.get('size'))
        except (IndexError, TypeError, ValueError):
            return False
        mtime = _mtime_to_float(parts[2])
        cur_mtime = _mtime_to_float(md.get('mtime'))
        if mtime is None or cur_mtime is None:
            return False
        return size == cur_size and abs(mtime - cur_mtime) < 1e-6

    # -- worker loop ----------------------------------------------------------

    IDLE_RECONCILE_SECONDS = 15

    def run(self):
        idle_since = None
        while not self.stop_event.is_set():
            try:
                if self._paused.is_set():
                    # Suspend without touching the queues; pick up where we left off
                    # on resume. Takes effect between books/phases, not mid-book.
                    self.status_cb({'state': 'paused'})
                    self.stop_event.wait(1)
                    continue
                pending = self.store.dirty_book_ids()
                if pending:
                    # Indexing (embedding) has priority: drain the whole dirty queue
                    # before any attribute work so the two models are used in batches.
                    idle_since = None
                    self._process_one(pending[0])
                    continue
                settings = self.settings_provider()
                if self._attr_requested.is_set() or getattr(settings, 'auto_extract_attributes', True):
                    attr_pending = self._pending_attr_books(settings)
                    if attr_pending:
                        idle_since = None
                        completed = self._process_attributes(attr_pending, settings)
                        # keep the request alive if we were paused mid-phase so it resumes
                        if completed:
                            self._attr_requested.clear()
                        continue
                # Fully idle.
                now = time.time()
                if idle_since is None:
                    idle_since = now
                elif now - idle_since >= self.IDLE_RECONCILE_SECONDS:
                    idle_since = now
                    self.reconcile()
                self.status_cb({'state': 'idle'})
                self.stop_event.wait(2)
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
        fmt = pick_format(formats, settings.format_priority)
        if fmt is None:
            self.store.remove_dirty(book_id)
            return
        try:
            md = api.format_metadata(book_id, fmt)
        except Exception:
            md = {'size': None, 'mtime': None}

        self.current_book_id = book_id
        self._status('extracting', book_id, fmt=fmt)
        try:
            path = api.format(book_id, fmt, as_path=True) or None
        except Exception as e:
            self._fail(book_id, f'could not read {fmt} file: {e}')
            return
        if not path:
            self._fail(book_id, f'no readable {fmt} file for book {book_id}')
            return
        try:
            kind, payload = extract_book_pages(path, fmt)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if kind == 'error':
            self._fail(book_id, payload)
            return
        if kind == 'pages' and not any((p or '').strip() for p in payload):
            self._status('done', book_id, note='no text found')
            self.store.clear_book(book_id)
            self.store.upsert_book(book_id, fmt, 0, settings.embed.model, 0)
            self.store.remove_dirty(book_id)
            return

        eff_target = min(settings.target_chars, max_chunk_chars(settings.embed_context_tokens))
        # hard per-chunk token cap so non-Latin scripts stay inside the model's context
        max_tokens = settings.embed_context_tokens - CONTEXT_OVERHEAD_TOKENS
        if kind == 'pages':
            chunks = chunks_from_pages(payload, eff_target, settings.overlap_chars, max_tokens=max_tokens)
        else:
            chunks = split_plain_text(payload or '', eff_target, settings.overlap_chars, max_tokens=max_tokens)
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
        self.store.set_meta(file_info_key(book_id), file_info_value(fmt, md))
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
        formats = api.formats(book_id) if api is not None else ()
        fmt = pick_format(formats, settings.format_priority) if formats else None
        if fmt is not None and api is not None:
            try:
                md = api.format_metadata(book_id, fmt)
                self.store.set_meta(file_info_key(book_id), file_info_value(fmt, md))
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
