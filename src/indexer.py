'''Background indexing worker: dirty queue -> extract -> chunk -> embed -> store.

Runs as a daemon thread. All calibre/Qt access happens through the injected
`get_new_api()` callable so it works across library switches and is testable.

Books that yield fewer than MIN_EXTRACTED_CHARS of text (scanned/image-only) or
that lose all their text in chunking fail explicitly via _fail instead of being
recorded as 0-chunk successes: reconcile skips failed books whose file is
unchanged, and 'Re-index new && failed' retries them on demand.
'''

from __future__ import annotations

import os
import threading
import time
import unicodedata

from .chunker import (
    CONTEXT_OVERHEAD_TOKENS,
    chunks_from_pages,
    max_chunk_chars,
    split_plain_text,
)
from .store import VectorStore

# Minimum amount of extractable text for a book to count as indexable content.
# Below this the file is treated as scanned/image-only (or broken) and indexing
# fails explicitly, instead of recording a silent 0-chunk "success" that never
# surfaces in the GUI. Char count (not chunk count) so the guard is independent
# of target_chars/overlap/max_chunks_per_book settings.
MIN_EXTRACTED_CHARS = 1000


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


def _mtime_to_int(v):
    """Normalize an mtime value (number or datetime) to integer epoch seconds (truncated,
    never rounded into the future)."""
    if v is None:
        return None
    ts = getattr(v, 'timestamp', None)
    if callable(ts):
        try:
            return int(float(ts()))
        except Exception:
            pass
    try:
        return int(float(v))
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(str(v)).timestamp())
    except Exception:
        return None


def file_info_from_md(fmt: str, md: dict):
    """(size, mtime_s) for change detection; None when the stat info is missing."""
    try:
        size = int(md.get('size'))
    except (TypeError, ValueError):
        return None
    mtime_s = _mtime_to_int(md.get('mtime'))
    if mtime_s is None:
        return None
    return size, mtime_s


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
        self._forced_attrs: set[int] = set()
        self._forced_lock = threading.Lock()
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

    def request_attributes(self, book_id=None):
        """Ask the worker loop to run the attribute-extraction phase.

        If book_id is given, that book is (re-)extracted even when its attributes
        are already stored; it stays in the force list until the phase has
        processed it (so a pause mid-phase doesn't lose the request).
        """
        if book_id is not None:
            with self._forced_lock:
                self._forced_attrs.add(int(book_id))
        self._attr_requested.set()

    def _pending_attr_books(self, settings) -> list[int]:
        from .attributes import pending_attribute_books

        failed = set(self.store.failed_book_ids('attr'))
        return [b for b in pending_attribute_books(self.store, settings) if b not in failed]

    def _attr_phase_books(self, settings) -> list[int]:
        """Books for the next attribute phase: normal pending books plus forced re-extractions."""
        attr_pending = self._pending_attr_books(settings)
        with self._forced_lock:
            forced = [b for b in sorted(self._forced_attrs) if b not in attr_pending]
        return attr_pending + forced

    def _drop_forced(self, book_id):
        with self._forced_lock:
            self._forced_attrs.discard(int(book_id))

    def _set_attr_failed(self, book_id: int, error: str | None):
        if error is None:
            self.store.clear_failed(book_id, 'attr')
        else:
            self.store.set_failed(book_id, 'attr', error)

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

            def progress(d, t, _bid=bid, _i=i):
                # per-book sub-progress (fulltext map steps) on top of book-level done/total
                self._status('attributes', _bid, done=_i + 1, total=total, sub_done=d, sub_total=t)

            try:
                extract_book_attributes(bid, self.attr_writer, self.store, settings, llm=llm, progress_cb=progress)
                self._set_attr_failed(bid, None)
            except Exception as e:
                self._set_attr_failed(bid, repr(e))
                errors.append((bid, repr(e)))
            finally:
                self._drop_forced(bid)
        self._status('attributes_done', done=total, total=total)
        self.attr_done_cb((total, errors))
        return True

    # -- public -------------------------------------------------------------

    def reconcile(self):
        """Compare library vs store; queue missing/changed books, drop removed ones."""
        with self._reconcile_lock:
            self._reconcile_locked()

    def _reconcile_locked(self):
        api = self.get_new_api()
        if api is None:
            return
        settings = self.settings_provider()
        lib_ids = set(api.all_book_ids())
        indexed = {b['id']: b for b in self.store.indexed_books()}
        failed_indexing = set(self.store.failed_book_ids('index'))
        # Every book id the store knows about, so a vanished book is cleaned up
        # even when it left no books/dirty row behind (failed indexing) and
        # orphans from older versions are swept on the first run.
        known = (
            set(indexed)
            | set(self.store.dirty_book_ids())
            | set(self.store.attr_book_ids())
            | set(self.store.file_info_book_ids())
            | set(self.store.failed_book_ids())
        )
        # remove books that vanished from the library
        for bid in sorted(known - lib_ids):
            self.store.clear_book(bid)
            self.store.remove_dirty(bid)
            self.store.clear_file_info(bid)
            self.store.clear_attrs(bid)
            self.store.clear_failed(bid)
        for bid in lib_ids:
            formats = api.formats(bid)
            fmt = pick_format(formats, settings.format_priority)
            if fmt is None:
                continue
            try:
                md = api.format_metadata(bid, fmt)
            except Exception:
                continue
            fi = self.store.get_file_info(bid)
            info = indexed.get(bid)
            if info is None:
                # never indexed; skip if it failed before and the file is unchanged
                if bid in failed_indexing and fi is not None and fi['fmt'] == fmt and self._same_file(fi, md):
                    continue
                self.store.add_dirty(bid, 'added')
                continue
            if fi is None:
                # no file info recorded (e.g. stat info was missing at index time)
                changed = info['fmt'] != fmt
            else:
                changed = fi['fmt'] != fmt or not self._same_file(fi, md)
            if changed:
                self.store.add_dirty(bid, 'changed')
        try:
            # removed books may have been the last ones of their model
            self.store.cleanup_stale_models(settings.embed.model)
        except Exception:
            pass

    @staticmethod
    def _same_file(fi: dict, md: dict) -> bool:
        try:
            size_ok = int(md.get('size')) == fi['size']
        except (TypeError, ValueError):
            size_ok = False
        mtime_s = _mtime_to_int(md.get('mtime'))
        return size_ok and mtime_s is not None and mtime_s == fi['mtime_s']

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
                    try:
                        # a re-indexed book may have been the last one of its old model
                        self.store.cleanup_stale_models(self.settings_provider().embed.model)
                    except Exception as e:
                        _default_log(f'stale model cleanup failed: {e!r}')
                    continue
                settings = self.settings_provider()
                if self._attr_requested.is_set() or getattr(settings, 'auto_extract_attributes', True):
                    phase_books = self._attr_phase_books(settings)
                    if phase_books:
                        idle_since = None
                        completed = self._process_attributes(phase_books, settings)
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
        try:
            self._index_book(book_id)
        except Exception as e:
            # An unexpected per-book crash must fail the book (visible in the
            # status dialog) instead of leaking into run(), which would retry it
            # forever and wedge the whole indexing run.
            try:
                self._fail(book_id, f'indexing failed unexpectedly: {e!r}')
            except Exception as e2:
                _default_log(f'_fail also failed for book {book_id}: {e2!r}')

    def _index_book(self, book_id: int):
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
        if kind == 'pages':
            extracted_chars = sum(len((p or '').strip()) for p in payload)
        else:
            extracted_chars = len((payload or '').strip())
        if extracted_chars < MIN_EXTRACTED_CHARS:
            self.store.clear_book(book_id)
            self._fail(
                book_id, f'no meaningful text extracted ({extracted_chars} chars from {fmt}): the file may be scanned or image-only'
            )
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
            # text was extracted but chunking lost it all (e.g. a non-standard
            # page structure): fail loudly instead of recording an empty index
            self.store.clear_book(book_id)
            self._fail(book_id, f'chunking produced no chunks from {extracted_chars} chars of {fmt} text')
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
        self._status('saving', book_id)
        from .store import l2_normalize

        self.store.clear_book(book_id)
        for c, v in zip(chunks, vectors):
            self.store.insert_chunk(book_id, c, settings.embed.model, l2_normalize(v))
        self.store.commit()
        self.store.upsert_book(book_id, fmt, len(chunks), settings.embed.model)
        # record size/mtime for change detection (no row when stat info is missing:
        # a missing file_info falls back to format-only comparison in reconcile)
        fi = file_info_from_md(fmt, md)
        if fi is not None:
            self.store.set_file_info(book_id, fmt, *fi)
        else:
            self.store.clear_file_info(book_id)
        self.store.clear_failed(book_id, 'index')
        self.store.remove_dirty(book_id)
        self.current_book_id = None
        self._status('done', book_id, n_chunks=len(chunks))

    def _fail(self, book_id: int, msg: str):
        api = None
        try:
            api = self.get_new_api()
            settings = self.settings_provider()
            formats = api.formats(book_id) if api is not None else ()
            fmt = pick_format(formats, settings.format_priority) if formats else None
        except Exception:
            fmt = None
        if fmt is not None and api is not None:
            try:
                md = api.format_metadata(book_id, fmt)
                fi = file_info_from_md(fmt, md)
                if fi is not None:
                    self.store.set_file_info(book_id, fmt, *fi)
                else:
                    self.store.clear_file_info(book_id)
            except Exception:
                pass
        self.store.set_failed(book_id, 'index', msg)
        self.store.remove_dirty(book_id)
        self.current_book_id = None
        self._status('error', book_id, error=msg)
