'''Semantic search dialog: query box, results table, viewer jump.'''

from __future__ import annotations

import re

from qt.core import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QThread,
    QVBoxLayout,
    Qt,
    pyqtSignal,
)

from calibre.utils.localization import _


class SearchWorker(QThread):
    finished_ok = pyqtSignal(object)  # list[SearchResult]
    failed = pyqtSignal(str)

    def __init__(self, store, client, query: str, limit: int):
        super().__init__()
        self.store = store
        self.client = client
        self.query = query
        self.limit = limit

    def run(self):
        try:
            vecs = self.client.embed([self.query])
            if not vecs:
                self.failed.emit(_('No embedding returned'))
                return
            res = self.store.search(vecs[0], limit=self.limit)
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


class SemanticSearchDialog(QDialog):
    def __init__(self, gui, action):
        super().__init__(gui)
        self.gui = gui
        self.action = action
        self.store = action.store
        self.results = []
        self.worker = None
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

        self.status_label = QLabel('')
        v.addWidget(self.status_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([_('Book'), _('Chapter'), _('Match'), _('Score')])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(lambda *_: self.open_selected())
        v.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.btn_open = QPushButton(_('Open in &viewer'))
        self.btn_open.clicked.connect(self.open_selected)
        self.btn_restrict = QPushButton(_('&Restrict library to results'))
        self.btn_restrict.setToolTip(_('Set the library search so only books with matches are shown'))
        self.btn_restrict.clicked.connect(self.restrict_library)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_restrict)
        bottom.addWidget(self.btn_open)
        v.addLayout(bottom)

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
        self.btn_search.setEnabled(False)
        self.status_label.setText(_('Searching...'))
        self.worker = SearchWorker(self.store, client, q, limit=50)
        self.worker.finished_ok.connect(self._on_results)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_results(self, results):
        self.btn_search.setEnabled(True)
        self.results = results
        self.table.setRowCount(0)
        api = self.action._api()
        meta_cache: dict[int, object] = {}
        row = 0
        for r in results:
            if api is not None and r.book_id not in meta_cache:
                try:
                    mi = api.get_metadata(r.book_id)
                    meta_cache[r.book_id] = (mi.title or '?', ', '.join(mi.authors or []))
                except Exception:
                    meta_cache[r.book_id] = ('?', '')
            title, authors = meta_cache[r.book_id]
            label = f'{title}\n{authors}' if authors else title
            self.table.insertRow(row)
            it = QTableWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, r.book_id)
            self.table.setItem(row, 0, it)
            self.table.setItem(row, 1, QTableWidgetItem(r.chapter_label or '—'))
            snippet = r.text.replace('\n', ' ')
            if len(snippet) > 240:
                snippet = '…' + snippet[-240:]
            self.table.setItem(row, 2, QTableWidgetItem(snippet))
            self.table.setItem(row, 3, QTableWidgetItem(f'{r.score:.3f}'))
            row += 1
        self.status_label.setText(f'{len(results)} matches')

    def _on_failed(self, msg):
        self.btn_search.setEnabled(True)
        self.status_label.setText(_('Search failed: ') + msg)

    def selected_row(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else (self.table.currentRow() if self.table.currentRow() >= 0 else None)

    def open_selected(self):
        row = self.selected_row()
        if row is None or row >= len(self.results):
            return
        r = self.results[row]
        self.open_result(r)

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
        ids = sorted({r.book_id for r in self.results})
        if not ids:
            return
        q = ' or '.join(f'id:{i}' for i in ids)
        self.gui.search.set_search_string(q, store_in_history=True)

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(10000)
        super().closeEvent(e)
