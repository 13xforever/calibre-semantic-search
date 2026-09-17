'''Semantic search dialog: query box, per-book results table, drill-down to a book's matches, viewer jump.'''

from __future__ import annotations

import re

from calibre.utils.localization import _
from qt.core import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QThread,
    QVBoxLayout,
    pyqtSignal,
)

# Hard cap on result rows (books) per search, regardless of the score threshold (keeps the
# table usable even if the user sets the minimum score to 0 on a large library).
MAX_RESULTS = 1000

# Strong refs to in-flight workers. Slots clear their worker attribute as soon as the
# results arrive, but that only drops one reference; this set keeps each QThread alive
# until its own `finished` signal (emitted after run() has returned and the thread has
# stopped). Deleting a QThread while it is still running aborts calibre with
# "QThread: Destroyed while thread ... is still running", so the last reference must not
# vanish before the thread has actually finished.
_live_workers = set()


class SearchWorker(QThread):
    finished_ok = pyqtSignal(object)  # list[SearchResult] (one per book)
    failed = pyqtSignal(str)

    def __init__(self, store, client, query: str, limit: int, min_score: float, model: str | None = None):
        super().__init__()
        self.store = store
        self.client = client
        self.query = query
        self.limit = limit
        self.min_score = min_score
        self.model = model
        self.query_vec = None  # set in run(); reused by BookMatchesWorker for drill-downs

    def run(self):
        try:
            vecs = self.client.embed([self.query])
            if not vecs:
                self.failed.emit(_('No embedding returned'))
                return
            self.query_vec = vecs[0]
            res = self.store.search(vecs[0], limit=self.limit, min_score=self.min_score, model=self.model)
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class BookMatchesWorker(QThread):
    """One book's full match list (store.search_book), reusing the search's query vector."""

    finished_ok = pyqtSignal(object)  # list[SearchResult]
    failed = pyqtSignal(str)

    def __init__(self, store, query_vec, book_id: int, min_score: float, model: str | None = None):
        super().__init__()
        self.store = store
        self.query_vec = query_vec
        self.book_id = book_id
        self.min_score = min_score
        self.model = model

    def run(self):
        try:
            res = self.store.search_book(self.query_vec, self.book_id, min_score=self.min_score, model=self.model)
            self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class DeleteChunkWorker(QThread):
    """Delete one chunk from the store, then re-fetch the book's match list with the
    search's cached query vector (no new embedding call)."""

    finished_ok = pyqtSignal(object)  # fresh match list, or None when there is no query vector
    failed = pyqtSignal(str)

    def __init__(self, store, query_vec, book_id: int, chunk_no: int, min_score: float, model: str | None = None):
        super().__init__()
        self.store = store
        self.query_vec = query_vec
        self.book_id = book_id
        self.chunk_no = chunk_no
        self.min_score = min_score
        self.model = model

    def run(self):
        try:
            self.store.delete_chunk(self.book_id, self.chunk_no)
            if self.query_vec is None:
                self.finished_ok.emit(None)
            else:
                res = self.store.search_book(self.query_vec, self.book_id, min_score=self.min_score, model=self.model)
                self.finished_ok.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


def best_search_phrase(chunk_text: str, max_len: int = 400) -> str:
    """Pick a long contiguous phrase from a chunk for the viewer's text search."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', chunk_text) if p.strip()]
    if not paras:
        return chunk_text[:max_len]
    best = max(paras, key=len)
    if len(best) > max_len:
        # middle slice is more distinctive than the start
        best = best[len(best) // 4 : len(best) // 4 + max_len]
    return best.strip()


class _ResultsTable(QTableWidget):
    """Results table where Home/End jump the row selection to the top/bottom row, and the
    two views are navigated with arrow keys: Enter / Right / Alt+Right go forward
    (`on_enter`, wired to open_selected — same as double-click), Backspace / Left /
    Alt+Left go back (`on_back`, wired to go_back).

    Qt's default for Home/End moves the current *cell* instead (Home -> first column,
    End -> last column, Ctrl+... -> corner cells), which is invisible under SelectRows;
    rows are the unit of interaction here, so all four should select the first/last row.
    The whole current row stays selected at all times (a Tab focus-in would otherwise
    highlight a single cell).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Tab focus lands on a single cell (the view highlights just that cell); rows are
        # the unit of interaction, so keep the whole current row selected.
        self.currentCellChanged.connect(self._select_current_row)

    def _select_current_row(self, current, *rest):
        # calibre's Qt binding marshals currentCellChanged as (row, col, prev_row, prev_col)
        # ints rather than (QModelIndex, QModelIndex); either way the row comes first.
        row = current.row() if hasattr(current, 'row') else current
        if row >= 0:
            self.selectRow(row)

    def keyPressEvent(self, e):
        mods = e.modifiers()
        # Only bare and Ctrl variants; anything else (e.g. Shift) keeps Qt's default.
        if e.key() in (Qt.Key.Key_Home, Qt.Key.Key_End) and not (mods & ~Qt.KeyboardModifier.ControlModifier):
            if self.rowCount() > 0:
                self.setCurrentCell(0 if e.key() == Qt.Key.Key_Home else self.rowCount() - 1, 0)
            e.accept()
            return
        key = e.key()
        alt_only = mods == Qt.KeyboardModifier.AltModifier
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not mods:
            cb = getattr(self, 'on_enter', None)
        elif key == Qt.Key.Key_Right and (not mods or alt_only):
            cb = getattr(self, 'on_enter', None)
        elif key == Qt.Key.Key_Backspace and not mods:
            cb = getattr(self, 'on_back', None)
        elif key == Qt.Key.Key_Left and (not mods or alt_only):
            cb = getattr(self, 'on_back', None)
        else:
            cb = None
        if cb is not None:
            cb()
            e.accept()
            return
        super().keyPressEvent(e)


class SemanticSearchDialog(QDialog):
    """Two views over one table: the per-book search results, and (after selecting a
    book) that book's full match list. Back returns to the books with its row and sort
    restored; a new search always starts in the books view."""

    def __init__(self, gui, action):
        super().__init__(gui)
        self.gui = gui
        self.action = action
        self.store = action.store
        self.books = []  # books-view rows: one SearchResult per book (its best chunk)
        self.matches = None  # matches-view rows for the active book; None until loaded
        self._active_book_id = None
        self._book_cursor = 0  # books-view row to restore on Back
        self._query_ctx = None  # (query_vec, model) reused by BookMatchesWorker
        self._sort_state = {'books': [2, False], 'matches': [2, False]}  # (col, asc) per view
        self._meta_cache = {}
        self._gen = 0  # bumped by do_search/drill_into/delete_selected; stale worker results are dropped
        self.worker = None
        self.matches_worker = None
        self.delete_worker = None
        self._deleting = None  # (book_id, chunk_no) of the in-flight deletion
        self.setWindowTitle(_('Semantic search'))
        self.resize(900, 560)

        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(_('Describe what you are looking for, e.g. "a slow-burn romance between enemies"'))
        self.edit.returnPressed.connect(self.do_search)
        top.addWidget(self.edit, 1)
        self.btn_search = QPushButton(_('&Search'))
        self.btn_search.clicked.connect(self.do_search)
        top.addWidget(self.btn_search)
        v.addLayout(top)

        mid = QHBoxLayout()
        self.status_label = QLabel('')
        mid.addWidget(self.status_label)
        self.btn_back = QPushButton(_('&Back'))
        self.btn_back.clicked.connect(self.go_back)
        mid.addWidget(self.btn_back)
        self.book_label = QLabel('')
        mid.addWidget(self.book_label, 1)
        v.addLayout(mid)

        self.table = _ResultsTable(0, 3)
        # Match is Stretch so it absorbs all leftover width: the table always spans the full
        # dialog and the snippet column grows/shrinks with the window (and when other columns
        # are dragged). The rest stay Interactive so Book/Chapter/Score remain user-resizable.
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 260)
        self.table.setColumnWidth(2, 65)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(lambda *_: self.open_selected())
        self.table.on_enter = self.open_selected  # Enter / Right / Alt+Right on a row == double-click
        self.table.on_back = self.go_back  # Backspace / Left / Alt+Left return to the books view
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        v.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.count_label = QLabel('')
        bottom.addWidget(self.count_label)
        bottom.addStretch(1)
        self.btn_delete = QPushButton(_('Delete &chunk'))
        self.btn_delete.setToolTip(
            _('Remove only this chunk from the book\'s search index. The book keeps its other chunks; re-indexing the book restores it.')
        )
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_restrict = QPushButton(_('&Restrict library to results'))
        self.btn_restrict.setToolTip(_('Set the library search so only books with matches are shown'))
        self.btn_restrict.clicked.connect(self.restrict_library)
        self.btn_context = QPushButton(_('Show &book matches'))
        bottom.addWidget(self.btn_delete)
        bottom.addSpacing(12)  # set chunk deletion apart from the view/filter buttons
        bottom.addWidget(self.btn_restrict)
        bottom.addWidget(self.btn_context)
        v.addLayout(bottom)
        self._update_view_chrome()

    # -- actions ---------------------------------------------------------------

    def do_search(self):
        q = self.edit.text().strip()
        if not q or self.worker is not None:
            return
        try:
            indexed = [b for b in self.store.indexed_books() if b['n_chunks'] > 0]
        except Exception:
            indexed = []
        if not indexed:
            self.status_label.setText(_('No books indexed yet — run indexing first (Semantic search menu).'))
            return
        settings = self.action.get_settings()
        from .embed_client import EmbedClient

        client = EmbedClient(
            base_url=settings.embed.base_url, model=settings.embed.model, api_key=settings.embed.api_key, timeout=settings.embed.timeout
        )
        min_score = min(1.0, max(0.0, float(settings.search_min_score)))
        self._gen += 1
        gen = self._gen
        # A new search always starts in the books view; drop any drill-down state.
        self.worker = None
        self.matches_worker = None
        self.books = []
        self.matches = None
        self._active_book_id = None
        self._book_cursor = 0
        self._query_ctx = None
        self._meta_cache = {}
        self.btn_search.setEnabled(False)
        self.status_label.setText(_('Searching...'))
        self._update_view_chrome()
        self._render_table()
        w = SearchWorker(self.store, client, q, limit=MAX_RESULTS, min_score=min_score, model=settings.embed.model)
        self.worker = w
        _live_workers.add(w)
        w.finished_ok.connect(lambda res, g=gen, w=w: self._on_results(res, g, w))
        w.failed.connect(lambda msg, g=gen: self._on_failed(msg, g))
        w.finished.connect(lambda w=w: _live_workers.discard(w))
        w.start()

    def _on_results(self, results, gen, worker):
        if gen != self._gen:
            return
        self.worker = None
        self.btn_search.setEnabled(True)
        vec = worker.query_vec
        self._query_ctx = (vec, worker.model) if vec is not None else None
        self.books = list(results)
        self._sort_state['books'] = [2, False]  # the store returns score-descending; reflect that in the header
        self.status_label.setText('')
        self._update_view_chrome()
        self._render_table()

    def drill_into(self):
        row = self.selected_row()
        if row is None or row >= len(self.books) or self.matches is not None:
            return
        r = self.books[row]
        if self._active_book_id == r.book_id:
            return  # this book's matches are already loading
        ctx = self._query_ctx
        if ctx is None:
            return
        vec, model = ctx
        settings = self.action.get_settings()
        min_score = min(1.0, max(0.0, float(settings.search_min_score)))
        title = self._book_title(r.book_id, self.action._api())
        self._gen += 1
        gen = self._gen
        self._active_book_id = r.book_id
        self._book_cursor = row
        self.book_label.setText(title)
        w = BookMatchesWorker(self.store, vec, r.book_id, min_score=min_score, model=model)
        self.matches_worker = w
        _live_workers.add(w)
        w.finished_ok.connect(lambda res, g=gen: self._on_book_results(res, g))
        w.failed.connect(lambda msg, g=gen: self._on_book_failed(msg, g))
        w.finished.connect(lambda w=w: _live_workers.discard(w))
        self._update_view_chrome()  # books rows stay on screen until the matches arrive
        w.start()

    def _on_book_results(self, results, gen):
        if gen != self._gen or self._active_book_id is None:
            return
        self.matches_worker = None
        self.matches = list(results)
        self._sort_state['matches'] = [2, False]  # search_book returns score-descending
        self._update_view_chrome()
        self._render_table()
        if results:
            # land on the first match so arrow keys / Enter work immediately
            self.table.setFocus()
            self.table.setCurrentCell(0, 0)

    def _on_book_failed(self, msg, gen):
        if gen != self._gen or self._active_book_id is None:
            return
        self.matches_worker = None
        self._active_book_id = None  # fall back to the books view and show the error there
        self.status_label.setText(_('Loading matches failed: ') + msg)
        self._update_view_chrome()
        self._render_table()

    def delete_selected(self):
        """Remove the selected match's chunk from the store (book-matches view only)."""
        if self.worker is not None or self.delete_worker is not None:
            return
        row = self.selected_row()
        if self.matches is None or row is None or row >= len(self.matches):
            return
        r = self.matches[row]
        ctx = self._query_ctx
        vec, model = (ctx if ctx is not None else (None, None))
        settings = self.action.get_settings()
        min_score = min(1.0, max(0.0, float(settings.search_min_score)))
        self._gen += 1
        gen = self._gen
        self._deleting = (r.book_id, r.chunk_no)
        w = DeleteChunkWorker(self.store, vec, r.book_id, r.chunk_no, min_score=min_score, model=model)
        self.delete_worker = w
        _live_workers.add(w)
        w.finished_ok.connect(lambda res, g=gen, w=w: self._on_chunk_deleted(res, g, w))
        w.failed.connect(lambda msg, g=gen, w=w: self._on_chunk_delete_failed(msg, g, w))
        w.finished.connect(lambda w=w: _live_workers.discard(w))
        self.status_label.setText(_('Deleting chunk...'))
        self._update_view_chrome()
        w.start()

    def _on_chunk_deleted(self, results, gen, worker):
        if self.delete_worker is worker:
            # clear the in-flight state even for a stale result (a newer search superseded
            # this deletion), or the button would stay disabled until the next view change
            self.delete_worker = None
            deleting = self._deleting
            self._deleting = None
        else:
            deleting = None
        if gen != self._gen or self._active_book_id is None:
            return
        if results is not None:
            self.matches = list(results)
            self._sort_state['matches'] = [2, False]  # search_book returns score-descending
        else:
            # no query vector to re-search with: drop the deleted row locally
            bid, cno = deleting or (None, None)
            self.matches = [m for m in self.matches if not (m.book_id == bid and m.chunk_no == cno)]
        self.status_label.setText('')
        self._update_view_chrome()
        self._render_table()

    def _on_chunk_delete_failed(self, msg, gen, worker):
        if self.delete_worker is worker:
            self.delete_worker = None
            self._deleting = None
        if gen != self._gen:
            return
        self.status_label.setText(_('Deleting chunk failed: ') + msg)
        self._update_view_chrome()
        self.status_label.setVisible(True)  # the matches view normally hides it; keep the error on screen

    def go_back(self):
        if self._active_book_id is None:
            return
        self._active_book_id = None
        self.matches = None
        self._gen += 1  # drop any in-flight book-matches result
        self._update_view_chrome()
        self._render_table()
        row = min(self._book_cursor, max(0, len(self.books) - 1))
        if row >= 0:
            self.table.setCurrentCell(row, 0)
            # setCurrentCell does not reliably re-scroll after a same-size re-render, so
            # make the restored row visible explicitly.
            self.table.scrollToItem(self.table.item(row, 0), QAbstractItemView.ScrollHint.EnsureVisible)

    def _on_failed(self, msg, gen):
        if gen != self._gen:
            return
        self.worker = None
        self.btn_search.setEnabled(True)
        self.status_label.setText(_('Search failed: ') + msg)

    # -- views -------------------------------------------------------------------

    def _update_view_chrome(self):
        """Refresh everything that depends on which view/data is showing: the middle row,
        the table's columns, the count label and the context button."""
        in_book = self._active_book_id is not None
        deleting = self.delete_worker is not None
        # the status label doubles as the deletion progress/error line; keep it up while
        # a deletion runs or fails, even though the matches view normally hides it
        self.status_label.setVisible(not in_book or deleting)
        self.btn_back.setVisible(in_book)
        self.book_label.setVisible(in_book)
        self.btn_delete.setVisible(self.matches is not None)
        self.btn_delete.setEnabled(self.matches is not None and not deleting)
        if self.matches is not None:
            self.table.setHorizontalHeaderLabels([_('Chapter'), _('Match'), _('Score')])
            self.table.setColumnWidth(0, 160)
            self.btn_context.setText(_('Open in &viewer'))
            self.count_label.setText(f'{len(self.matches)} matches')
        else:
            # books rows (still loading or not drilled at all): keep the books columns
            self.table.setHorizontalHeaderLabels([_('Book'), _('Match'), _('Score')])
            self.table.setColumnWidth(0, 260)
            self.btn_context.setText(_('Show &book matches'))
            self.count_label.setText(_('Loading matches...') if in_book else f'{len(self.books)} books')

    def _render_table(self):
        api = self.action._api()
        rows_list = self.matches if self.matches is not None else self.books
        key = 'matches' if self.matches is not None else 'books'
        self.table.setRowCount(0)
        for row, r in enumerate(rows_list):
            self.table.insertRow(row)
            if key == 'matches':
                self.table.setItem(row, 0, QTableWidgetItem(r.chapter_label or '—'))
            else:
                # Title only: Qt's item views paint just the first line of multi-line item
                # text, so a 'title\nauthors' label would silently drop the author line.
                title = self._book_title(r.book_id, api)
                it = QTableWidgetItem(title)
                it.setToolTip(title)  # full title on hover when the column elides it
                self.table.setItem(row, 0, it)
            snippet = r.text.replace('\n', ' ')
            if len(snippet) > 240:
                snippet = '…' + snippet[-240:]
            self.table.setItem(row, 1, QTableWidgetItem(snippet))
            self.table.setItem(row, 2, QTableWidgetItem(f'{r.score:.3f}'))
        col, asc = self._sort_state[key]
        order = Qt.SortOrder.AscendingOrder if asc else Qt.SortOrder.DescendingOrder
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setSortIndicator(col, order)

    def _book_title(self, book_id, api):
        if book_id not in self._meta_cache:
            title = '?'
            if api is not None:
                try:
                    title = api.get_metadata(book_id).title or '?'
                except Exception:
                    pass
            self._meta_cache[book_id] = title
        return self._meta_cache[book_id]

    def _on_header_clicked(self, col):
        # Hand-rolled instead of setSortingEnabled so we control the keys (case-insensitive).
        # All columns sort as text: scores are rendered fixed-width (.3f), so text order
        # equals numeric order. list.sort is stable, so rows tied on the clicked column keep
        # their previous relative order.
        key = 'matches' if self.matches is not None else 'books'
        state = self._sort_state[key]
        if state[0] == col:
            state[1] = not state[1]
        else:
            state[0] = col
            state[1] = col != 2  # score: best first; text columns: A to Z
        rows_list = self.matches if key == 'matches' else self.books
        idx = list(range(len(rows_list)))
        if state[1]:
            idx.sort(key=lambda i: self._column_key(i, col))
        else:
            idx.sort(key=lambda i: self._column_key(i, col), reverse=True)
        reordered = [rows_list[i] for i in idx]
        if key == 'matches':
            self.matches = reordered
        else:
            self.books = reordered
        self._render_table()

    def _column_key(self, i, col):
        item = self.table.item(i, col)
        text = item.text() if item is not None else ''
        return text.casefold()

    def selected_row(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else (self.table.currentRow() if self.table.currentRow() >= 0 else None)

    def open_selected(self):
        row = self.selected_row()
        if row is None:
            return
        if self.matches is not None:
            if row >= len(self.matches):
                return
            self.open_result(self.matches[row])
        else:
            if row >= len(self.books):
                return
            self.drill_into()

    def open_result(self, r):
        from calibre.gui2 import config as guiconfig

        gui = self.gui
        fmt = r.fmt
        if not fmt:
            try:
                formats = self.action._api().formats(r.book_id)
                fmt = next(iter(formats), '')
            except Exception:
                fmt = ''
        phrase = best_search_phrase(r.text)
        if fmt in guiconfig['internally_viewed_formats']:
            gui.iactions['View'].view_format_by_id(r.book_id, fmt, open_at=f'search:{phrase}')
        else:
            gui.iactions['View'].view_format_by_id(r.book_id, fmt)

    def restrict_library(self):
        ids = sorted({r.book_id for r in self.books})
        if not ids:
            return
        q = ' or '.join(f'id:{i}' for i in ids)
        self.gui.search.set_search_string(q, store_in_history=True)

    def closeEvent(self, e):
        for w in (self.worker, self.matches_worker, self.delete_worker):
            if w is not None and w.isRunning():
                w.wait(10000)
        super().closeEvent(e)
