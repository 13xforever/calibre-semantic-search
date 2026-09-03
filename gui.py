'''Main GUI action: toolbar/search-bar entry point, indexer lifecycle.'''

from __future__ import annotations

import os
import threading

from qt.core import QToolButton, pyqtSignal

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

    def show_status(self):
        if not self._ensure_started():
            return
        books = self.store.indexed_books()
        dirty = self.store.dirty_book_ids()
        n_chunks = sum(b['n_chunks'] for b in books)
        lines = [
            f'Indexed books: {len(books)}',
            f'Total chunks: {n_chunks}',
            f'Pending (dirty): {len(dirty)}',
        ]
        if self.indexer is not None and self.indexer.current_book_id is not None:
            lines.append(f'Currently indexing book id {self.indexer.current_book_id}')
        from calibre.gui2 import info_dialog

        info_dialog(self.gui, 'Semantic search status', '\n'.join(lines), show=True)

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
            pick = pick_format(formats, settings.format_priority)
            if pick is None:
                continue
            self.store.add_dirty(bid, pick[0], 'reindex')

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
