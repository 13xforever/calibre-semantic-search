'''Main GUI action: toolbar/search-bar entry point, indexer lifecycle.'''

from __future__ import annotations

import os
import threading

from qt.core import QDialog, QDialogButtonBox, QTextEdit, QToolButton, QTimer, QVBoxLayout, pyqtSignal

from calibre.gui2.actions import InterfaceAction
from calibre.utils.localization import _

from .store import VectorStore
from .utils import load_settings, save_settings


def _plugin_icon(name):
    """Load a PNG shipped with the plugin (zip resource when installed, file when running from source)."""
    g = globals().get('get_icons')
    if callable(g):
        try:
            ic = g(name)
            if ic is not None and not ic.isNull():
                return ic
        except Exception:
            pass
    p = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(p):
        from qt.core import QIcon

        return QIcon(p)
    return None


class StatusDialog(QDialog):
    '''Live index-status view: fixed-size window, scrollable text, 1s refresh.'''

    def __init__(self, parent, action):
        super().__init__(parent)
        self.action = action
        self.setWindowTitle(_('Semantic search status'))
        self.resize(680, 440)
        lay = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.accepted.connect(self.close)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    def refresh(self):
        try:
            text = '\n'.join(self.action.status_lines())
        except Exception as e:
            text = f'status unavailable: {e!r}'
        if text == self.text.toPlainText():
            return
        sb = self.text.verticalScrollBar()
        pos = sb.value() if sb is not None else 0
        self.text.setPlainText(text)
        self.text.verticalScrollBar().setValue(pos)

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


class SemanticSearchAction(InterfaceAction):
    name = 'Semantic Search'
    action_spec = (_('Semantic search'), 'semantic_search.png', _('Search books by meaning (semantic search)'), None)
    popup_type = QToolButton.ToolButtonPopupMode.MenuButtonPopup
    action_add_menu = True
    action_type = 'global'

    _status_sig = pyqtSignal(object)

    def __init__(self, parent, site_customization):
        super().__init__(parent, site_customization)
        self.store: VectorStore | None = None
        self.indexer = None
        self.search_action = None
        self._reconcile_thread = None
        self._last_status = None
        self._status_dialog = None
        self._status_sig.connect(self._on_status)

    # -- lifecycle -----------------------------------------------------------

    def genesis(self):
        m = self.qaction.menu()
        assert m is not None
        icon = _plugin_icon('semantic_search.png')
        if icon is not None:
            self.qaction.setIcon(icon)
        # main button click opens the search dialog; the arrow shows this menu
        self.qaction.triggered.connect(self.open_dialog)
        ac_search = self.create_action(spec=(_('Search...'), 'semantic_search.png', _('Open the semantic search dialog'), None), attr='search')
        if icon is not None:
            ac_search.setIcon(icon)
        ac_search.triggered.connect(self.open_dialog)
        m.addAction(ac_search)
        ac_status = self.create_action(spec=(_('Index status'), 'book.png', _('Show indexing status'), None), attr='status')
        ac_status.triggered.connect(self.show_status)
        m.addAction(ac_status)
        ac_reindex = self.create_action(spec=(_('Re-index all books'), 'view-refresh.png', _('Queue every book for re-indexing'), None), attr='reindex')
        ac_reindex.triggered.connect(self.reindex_all)
        m.addAction(ac_reindex)
        ac_attrs = self.create_action(spec=(_('Extract attributes...'), 'ai.png', _('Run LLM attribute extraction on indexed books'), None), attr='attrs')
        ac_attrs.triggered.connect(self.extract_attributes_menu)
        m.addAction(ac_attrs)
        ac_settings = self.create_action(spec=(_('Settings'), 'config.png', _('Semantic search settings'), None), attr='settings')
        ac_settings.triggered.connect(self.open_settings)
        m.addAction(ac_settings)

    def initialization_complete(self):
        # called once at GUI startup; library_changed only fires on library switches
        # gui.search does not exist yet during genesis, so the search-bar icon goes here
        try:
            sb = getattr(self.gui, 'search', None)
            if sb is not None and hasattr(sb, 'add_action'):
                icon = _plugin_icon('semantic_search.png')
                self.search_action = sb.add_action(icon if icon is not None else 'search.png')
                self.search_action.triggered.connect(self.open_dialog)
        except Exception:
            self.search_action = None
        self._start_for_library()

    def _ensure_started(self) -> bool:
        if self.store is None:
            self._start_for_library(show_error=True)
        return self.store is not None

    def _start_for_library(self, show_error: bool = False):
        self._stop_indexer()
        if self.store is not None:
            try:
                self.store.close()
            except Exception:
                pass
        self.store = None
        try:
            db = self.gui.current_db
        except Exception:
            return
        if db is None:
            return
        try:
            libdir = os.path.dirname(db.backend.dbpath)
        except Exception:
            return
        settings = self.get_settings()
        try:
            self.store = VectorStore(os.path.join(libdir, 'semantic-search.db'), backend=settings.vector_backend)
        except Exception as e:
            if show_error:
                from calibre.gui2 import error_dialog

                error_dialog(self.gui, 'Semantic search', f'Failed to open the semantic search store:\n{e}', show=True)
            return
        from .indexer import Indexer

        self.indexer = Indexer(
            store=self.store,
            get_new_api=lambda: (self.gui.current_db.new_api if self.gui is not None else None),
            settings_provider=self.get_settings,
            status_cb=self._status_sig.emit,
        )
        self.indexer.start()
        # reconcile in a background thread so startup stays snappy
        t = threading.Thread(target=self._reconcile_safe, name='SSReconcile', daemon=True)
        self._reconcile_thread = t
        t.start()

    def library_changed(self, db):
        self._start_for_library()

    def _reconcile_safe(self):
        try:
            if self.indexer is not None:
                self.indexer.reconcile()
        except Exception as e:
            print(f'semantic search: reconcile failed: {e!r}')

    def shutting_down(self):
        self._stop_indexer()
        if self.store is not None:
            try:
                self.store.close()
            except Exception:
                pass
            self.store = None

    def _stop_indexer(self):
        if self.indexer is not None:
            self.indexer.stop()
            self.indexer = None

    # -- status ----------------------------------------------------------------

    def _on_status(self, d):
        self._last_status = d
        state = d.get('state', '')
        if state == 'idle':
            tip = _('Search books by meaning')
        else:
            tip = f"Indexing book {d.get('book_id')} ({state})"
            if d.get('total'):
                tip += f" {d.get('done')}/{d.get('total')}"
        self.qaction.setToolTip(tip)

    # -- helpers ---------------------------------------------------------------

    def get_settings(self):
        from calibre.gui2 import gprefs

        return load_settings(gprefs)

    def open_settings(self):
        from calibre.gui2 import gprefs

        from .config_widget import SettingsWidget

        w = SettingsWidget(self.get_settings())
        if w.exec() == 1:
            save_settings(gprefs, w.settings())

    def open_dialog(self):
        if not self._ensure_started():
            return
        from .dialog import SemanticSearchDialog

        d = SemanticSearchDialog(self.gui, self)
        text = getattr(self.gui, 'search', None)
        if text is not None:
            try:
                q = text.text()
            except Exception:
                q = ''
            if q and ':' not in q:
                d.edit.setText(q)
        d.show()

    def _book_label(self, bid, api=None):
        if api is None:
            api = self._api()
        if api is not None and bid is not None:
            try:
                md = api.get_metadata(bid)
            except Exception:
                md = None
            if md is not None:
                title = getattr(md, 'title', None) or '?'
                authors = [a for a in (getattr(md, 'authors', None) or []) if a]
                label = f'"{title}"' + (f' by {", ".join(authors)}' if authors else '')
                return f'{label} [id {bid}]'
        return f'book {bid}'

    def status_lines(self):
        if self.store is None:
            return ['Store not started.']
        books = self.store.indexed_books()
        dirty = self.store.dirty_book_ids()
        n_chunks = sum(b['n_chunks'] for b in books)
        api = self._api()
        lines = [f'Indexed books: {len(books)}']
        if api is not None:
            try:
                lines.append(f'Library books: {len(api.all_book_ids())}')
            except Exception:
                pass
        lines += [f'Total chunks: {n_chunks}', f'Pending (dirty): {len(dirty)}']
        st = self._last_status
        if st and st.get('state') in ('extracting', 'embedding', 'saving'):
            line = f"Currently indexing {self._book_label(st.get('book_id'), api)}: {st.get('state')}"
            if st.get('total'):
                line += f" {st.get('done')}/{st.get('total')}"
            lines.append(line)
        import json

        try:
            failed = json.loads(self.store.get_meta('failed', '{}') or '{}')
        except Exception:
            failed = {}
        if failed and api is not None:
            lines.append('')
            lines.append(f'Failed books: {len(failed)}')
            for bid_str, info in sorted(failed.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
                try:
                    bid = int(bid_str)
                except ValueError:
                    continue
                lines.append(f'{self._book_label(bid, api)}: {(info or {}).get("error", "unknown error")}')
        return lines

    def show_status(self):
        if not self._ensure_started():
            return
        d = self._status_dialog
        if d is not None and d.isVisible():
            d.raise_()
            d.activateWindow()
            return
        d = StatusDialog(self.gui, self)
        self._status_dialog = d
        d.show()

    def reindex_all(self):
        if not self._ensure_started():
            return
        api = self._api()
        if api is None:
            return
        settings = self.get_settings()
        from .indexer import pick_format

        for bid in sorted(api.all_book_ids()):
            formats = api.formats(bid)
            if not formats:
                continue
            fmt = pick_format(formats, settings.format_priority)
            if fmt is None:
                continue
            self.store.add_dirty(bid, fmt, 'reindex')

    def extract_attributes_menu(self):
        from calibre.gui2 import info_dialog

        if not self._ensure_started():
            return
        api = self._api()
        if api is None:
            return
        from .attributes import pending_attribute_books

        settings = self.get_settings()
        pending = pending_attribute_books(self.store, settings)
        if not pending:
            info_dialog(self.gui, 'Attribute extraction', 'All indexed books already have attributes.', show=True)
            return
        threading.Thread(target=self._extract_worker, args=(pending,), daemon=True).start()

    def _extract_worker(self, pending):
        from .attributes import extract_book_attributes

        api = self._api()
        settings = self.get_settings()
        for i, bid in enumerate(pending):
            try:
                extract_book_attributes(bid, api, self.store, settings)
            except Exception as e:
                print(f'semantic search: attribute extraction failed for {bid}: {e!r}')

    def _api(self):
        try:
            return self.gui.current_db.new_api
        except Exception:
            return None
