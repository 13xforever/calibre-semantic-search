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
    """Load a PNG shipped with the plugin (zip resource when installed, file when running from source).

    Prefers calibre's theme-variant naming (<name>-for-dark-theme.png /
    <name>-for-light-theme.png) so the icon matches the active palette.
    """
    stem, ext = os.path.splitext(name)
    candidates = [name]
    try:
        from calibre.gui2 import is_dark_theme

        candidates.insert(0, f'{stem}-for-{"dark" if is_dark_theme() else "light"}-theme{ext}')
    except Exception:
        pass
    for cand in candidates:
        g = globals().get('get_icons')
        if callable(g):
            try:
                ic = g(cand)
                if ic is not None and not ic.isNull():
                    return ic
            except Exception:
                pass
        p = os.path.join(os.path.dirname(__file__), cand)
        if os.path.exists(p):
            from qt.core import QIcon

            return QIcon(p)
    return None


class _GuiDbProxy:
    '''Forwards calibre db writes to the GUI thread; reads pass through.

    The real new_api is resolved lazily via a getter so the proxy stays valid
    across library switches. calibre's newAPI is not safe to write from a worker
    thread, so create_custom_column and set_field are marshalled to the GUI thread
    synchronously; everything else (reads such as backend.custom_column_label_map)
    passes straight through.
    '''

    def __init__(self, action, get_real):
        self._action = action
        self._get_real = get_real

    def _real(self):
        r = self._get_real() if callable(self._get_real) else self._get_real
        if r is None:
            raise RuntimeError('no calibre db api available')
        return r

    def __getattr__(self, name):
        return getattr(self._real(), name)

    def create_custom_column(self, *a, **k):
        self._action._run_db_write('create_custom_column', a, k)

    def set_field(self, *a, **k):
        self._action._run_db_write('set_field', a, k)


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
    _db_sig = pyqtSignal(object)  # (method, args, kwargs, threading.Event)

    def __init__(self, parent, site_customization):
        super().__init__(parent, site_customization)
        self.store: VectorStore | None = None
        self.indexer = None
        self.search_action = None
        self._reconcile_thread = None
        self._last_status = None
        self._status_dialog = None
        self._status_sig.connect(self._on_status)
        self._db_sig.connect(self._on_db_write)

    # -- lifecycle -----------------------------------------------------------

    def _apply_theme_icon(self):
        # (re)resolve the plugin icon for the active light/dark theme and apply it
        # everywhere it is shown; also called on app.palette_changed so live
        # theme switches in Preferences > Appearance are picked up without a restart
        icon = _plugin_icon('semantic_search.png')
        if icon is not None:
            self.qaction.setIcon(icon)
            if getattr(self, '_search_menu_action', None) is not None:
                self._search_menu_action.setIcon(icon)
            if self.search_action is not None:
                self.search_action.setIcon(icon)
        # Qt's standard media icons (pause/resume) are generated from the active
        # palette at request time, so re-request them for the new theme; a fresh
        # or absent indexer is never paused
        paused = self.indexer is not None and self.indexer.paused
        self._set_pause_label(paused)

    def _hook_palette_changes(self):
        if getattr(self, '_palette_hooked', False):
            return
        try:
            from calibre.gui2 import qapplication_or_fail

            app = qapplication_or_fail()
            app.palette_changed.connect(self._apply_theme_icon)
            self._palette_hooked = True
        except Exception:
            pass

    def genesis(self):
        m = self.qaction.menu()
        assert m is not None
        # main button click opens the search dialog; the arrow shows this menu
        self.qaction.triggered.connect(self.open_dialog)
        ac_search = self.create_action(spec=(_('Search...'), 'semantic_search.png', _('Open the semantic search dialog'), None), attr='search')
        self._search_menu_action = ac_search
        ac_search.triggered.connect(self.open_dialog)
        m.addAction(ac_search)
        ac_status = self.create_action(spec=(_('Index status'), 'book.png', _('Show indexing status'), None), attr='status')
        ac_status.triggered.connect(self.show_status)
        m.addAction(ac_status)
        self._pause_action = self.create_action(
            spec=(_('Pause indexing'), None, _('Pause or resume indexing and attribute extraction'), None), attr='pause'
        )
        self._pause_action.triggered.connect(self.toggle_pause)
        self._set_pause_label(False)
        m.addAction(self._pause_action)
        ac_attrs = self.create_action(spec=(_('Extract attributes...'), 'ai.png', _('Run LLM attribute extraction on indexed books'), None), attr='attrs')
        ac_attrs.triggered.connect(self.extract_attributes_menu)
        m.addAction(ac_attrs)
        m.addSeparator()
        # && renders as a literal & in Qt action text (a single & would become a mnemonic)
        ac_reindex_new = self.create_action(
            spec=(_('Re-index new && failed books'), 'view-refresh.png', _('Queue books that are not indexed yet or failed before; skip already-indexed books'), None),
            attr='reindex_new',
        )
        ac_reindex_new.triggered.connect(self.reindex_new_and_failed)
        m.addAction(ac_reindex_new)
        ac_reindex = self.create_action(spec=(_('Re-index all books'), 'view-refresh.png', _('Queue every book for re-indexing'), None), attr='reindex')
        ac_reindex.triggered.connect(self.reindex_all)
        m.addAction(ac_reindex)
        m.addSeparator()
        ac_settings = self.create_action(spec=(_('Settings'), 'config.png', _('Semantic search settings'), None), attr='settings')
        ac_settings.triggered.connect(self.open_settings)
        m.addAction(ac_settings)
        self._apply_theme_icon()
        self._hook_palette_changes()

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
        try:
            self._hook_book_details_menu()
        except Exception as e:
            print(f'semantic search: book details menu hook failed: {e!r}')
        self._apply_theme_icon()
        self._hook_palette_changes()
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

        get_api = lambda: (self.gui.current_db.new_api if self.gui is not None else None)
        self.indexer = Indexer(
            store=self.store,
            get_new_api=get_api,
            settings_provider=self.get_settings,
            status_cb=self._status_sig.emit,
            attr_writer=_GuiDbProxy(self, get_api),
        )
        self.indexer.start()
        # a fresh indexer is never paused; make sure the menu label reflects that
        self._set_pause_label(False)
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
        try:
            self._unhook_book_details_menu()
        except Exception:
            pass
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
        elif state == 'paused':
            tip = _('Indexing paused')
        elif state == 'attributes':
            tip = f"Extracting attributes ({d.get('done')}/{d.get('total')})"
        elif state == 'attributes_done':
            tip = f"Attributes: {d.get('done')}/{d.get('total')} done"
        elif state == 'attr_error':
            tip = "Attribute extraction: " + str(d.get('error', ''))[:80]
        else:
            tip = f"Indexing book {d.get('book_id')} ({state})"
            if d.get('total'):
                tip += f" {d.get('done')}/{d.get('total')}"
        self.qaction.setToolTip(tip)

    # -- attribute extraction ------------------------------------------------------

    def _run_db_write(self, method, args, kwargs):
        '''Marshal a calibre db write to the GUI thread and wait for it.'''
        ev = threading.Event()
        self._db_sig.emit((method, args, kwargs, ev))
        if not ev.wait(timeout=30):
            raise RuntimeError('GUI thread did not process the db write in time')

    def _on_db_write(self, payload):
        method, args, kwargs, ev = payload
        try:
            api = self._api()
            if api is None:
                return
            getattr(api, method)(*args, **kwargs)
        except Exception as e:
            print(f'semantic search: db write {method} failed: {e!r}')
        finally:
            ev.set()

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
        lines = []
        if api is not None:
            try:
                lines.append(f'Books indexed: {len(books)}/{len(api.all_book_ids())}')
            except Exception:
                lines.append(f'Indexed books: {len(books)}')
        else:
            lines.append(f'Indexed books: {len(books)}')
        lines.append(f'Total chunks: {n_chunks}')
        st = self._last_status
        if st and st.get('state') == 'paused':
            lines.append(_('Indexing paused (use the menu to resume)'))
        elif st and st.get('state') in ('extracting', 'embedding', 'saving', 'attributes'):
            if st.get('state') == 'attributes':
                line = f"Extracting attributes ({st.get('done')}/{st.get('total')}): {self._book_label(st.get('book_id'), api)}"
            else:
                line = f"Currently indexing {self._book_label(st.get('book_id'), api)}: {st.get('state')}"
                if st.get('total'):
                    line += f" {st.get('done')}/{st.get('total')}"
            lines.append(line)
        try:
            settings = self.get_settings()
            from .attributes import pending_attribute_books

            with_chunks = [b for b in books if b['n_chunks'] > 0]
            pending_attrs = pending_attribute_books(self.store, settings)
            done_attrs = max(0, len(with_chunks) - len(pending_attrs))
            lines.append(f'Attributes stored: {done_attrs}/{len(with_chunks)} books')
        except Exception:
            pass
        lines.append(f'Pending: {len(dirty)}')
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
        try:
            attr_failed = json.loads(self.store.get_meta('attr_failed', '{}') or '{}')
        except Exception:
            attr_failed = {}
        if attr_failed and api is not None:
            lines.append('')
            lines.append(f'Attribute failures: {len(attr_failed)} (use "Extract attributes..." to retry)')
            for bid_str, info in sorted(attr_failed.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
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

    def toggle_pause(self):
        """Pause or resume the indexing/attribute-extraction worker."""
        if self.indexer is None:
            return
        if self.indexer.paused:
            self.indexer.resume()
        else:
            self.indexer.pause()
        self._set_pause_label(self.indexer.paused)

    def _set_pause_label(self, paused):
        act = getattr(self, '_pause_action', None)
        if act is not None:
            act.setText(_('Resume indexing') if paused else _('Pause indexing'))
            icon = self._pause_icon(paused)
            if icon is not None:
                act.setIcon(icon)

    def _pause_icon(self, paused):
        # Ship our own pause/play glyphs with light/dark variants — Qt's standard
        # media icons render as fixed near-black under both themes in this style.
        # Fall back to the standard icon if the shipped files are missing.
        name = 'semantic_play.png' if paused else 'semantic_pause.png'
        ic = _plugin_icon(name)
        if ic is not None:
            return ic
        from qt.core import QStyle

        sp = QStyle.StandardPixmap.SP_MediaPlay if paused else QStyle.StandardPixmap.SP_MediaPause
        try:
            return self.gui.style().standardIcon(sp)
        except Exception:
            return None

    # -- Book Details context menu -------------------------------------------------

    def _hook_book_details_menu(self):
        """Add per-book re-index/re-extract items to calibre's Book Details context menu.

        calibre builds that menu in calibre.gui2.book_details.details_context_menu_event
        and offers no extension point, so wrap the function: while the original runs,
        the module's QMenu is swapped for a subclass whose exec() injects our actions
        into the top-level menu just before it is shown.
        """
        try:
            from calibre.gui2 import book_details as bd
        except Exception:
            return
        if getattr(bd, '_ss_hooked', False):
            return
        orig = bd.details_context_menu_event
        action = self

        def hooked(view, ev, book_info, add_popup_action=False, edit_metadata=None):
            injected = [False]
            QMenuOrig = bd.QMenu

            class HookedQMenu(QMenuOrig):
                def exec(self, *a, **k):
                    if not injected[0]:
                        injected[0] = True
                        try:
                            action._add_book_details_actions(self)
                        except Exception as e:
                            print(f'semantic search: book details menu hook failed: {e!r}')
                    return super().exec(*a, **k)

            bd.QMenu = HookedQMenu
            try:
                return orig(view, ev, book_info, add_popup_action, edit_metadata)
            finally:
                bd.QMenu = QMenuOrig

        bd.details_context_menu_event = hooked
        bd._ss_orig_details_menu = orig
        bd._ss_hooked = True
        # The standalone Book Info dialog imports the function by name; patch that
        # binding too if the module is already loaded.
        import sys

        bi = sys.modules.get('calibre.gui2.dialogs.book_info')
        if bi is not None and getattr(bi, 'details_context_menu_event', None) is orig:
            bi.details_context_menu_event = hooked
            bi._ss_patched = True

    def _unhook_book_details_menu(self):
        try:
            from calibre.gui2 import book_details as bd
        except Exception:
            return
        orig = getattr(bd, '_ss_orig_details_menu', None)
        if orig is not None:
            bd.details_context_menu_event = orig
            del bd._ss_orig_details_menu
        if getattr(bd, '_ss_hooked', False):
            del bd._ss_hooked
        import sys

        bi = sys.modules.get('calibre.gui2.dialogs.book_info')
        if bi is not None and getattr(bi, '_ss_patched', False) and orig is not None:
            bi.details_context_menu_event = orig
            del bi._ss_patched

    def _add_book_details_actions(self, menu):
        """Items injected into the Book Details context menu (see _hook_book_details_menu)."""
        book_id = getattr(getattr(self.gui, 'library_view', None), 'current_id', None)
        if book_id is None:
            return
        from qt.core import QIcon

        menu.addSeparator()
        ac_embed = menu.addAction(_('Re-index this book for semantic search'))
        ac_embed.setToolTip(
            _('Queue this book to be re-read and re-embedded. Useful if the file changed outside calibre or you switched embedding models.')
        )
        # same refresh glyph as the "Re-index ..." items in the toolbar menu
        try:
            ic = QIcon.ic('view-refresh.png')
        except Exception:
            ic = None
        if ic is not None:
            ac_embed.setIcon(ic)
        ac_embed.triggered.connect(lambda checked=False, bid=book_id: self.reindex_book(bid))
        ac_attrs = menu.addAction(_('Re-extract attributes for this book'))
        ac_attrs.setToolTip(_('Run LLM attribute extraction on this book again and overwrite its stored attributes.'))
        try:
            ic2 = QIcon.ic('ai.png')
        except Exception:
            ic2 = None
        if ic2 is not None:
            ac_attrs.setIcon(ic2)
        ac_attrs.triggered.connect(lambda checked=False, bid=book_id: self.reextract_attributes_book(bid))

    def reindex_book(self, book_id):
        """Queue one book for re-embedding (Book Details context menu)."""
        if not self._ensure_started():
            return
        api = self._api()
        if api is None or book_id is None:
            return
        settings = self.get_settings()
        from .indexer import pick_format

        formats = api.formats(book_id)
        fmt = pick_format(formats, settings.format_priority) if formats else None
        if fmt is None:
            from calibre.gui2 import error_dialog

            error_dialog(self.gui, _('Semantic search'), f'No usable format for {self._book_label(book_id)}.', show=True)
            return
        self.store.add_dirty(book_id, fmt, 'reindex')
        from calibre.gui2 import info_dialog

        info_dialog(self.gui, _('Semantic search'), f'Queued {self._book_label(book_id)} for re-embedding.', show=True)

    def reextract_attributes_book(self, book_id):
        """Force LLM attribute extraction for one book (Book Details context menu)."""
        if not self._ensure_started():
            return
        if book_id is None:
            return
        indexed = {b['id']: b for b in self.store.indexed_books()}
        info = indexed.get(book_id)
        if info is None or info['n_chunks'] == 0:
            from calibre.gui2 import info_dialog

            info_dialog(
                self.gui,
                _('Semantic search'),
                f'{self._book_label(book_id)} has no indexed text yet.\n\n'
                'Use "Re-index this book for semantic search" first.',
                show=True,
            )
            return
        if not self._check_llm_provider():
            return
        self.indexer.request_attributes(book_id)
        self.show_status()

    def reindex_new_and_failed(self):
        """Queue books that are not indexed yet or previously failed; skip the rest."""
        if not self._ensure_started():
            return
        api = self._api()
        if api is None:
            return
        import json

        settings = self.get_settings()
        from .indexer import pick_format

        indexed = {b['id'] for b in self.store.indexed_books()}
        try:
            failed = set(json.loads(self.store.get_meta('failed', '{}') or '{}').keys())
        except Exception:
            failed = set()
        queued = 0
        for bid in sorted(api.all_book_ids()):
            if bid in indexed and str(bid) not in failed:
                continue  # already indexed without error -> skip
            formats = api.formats(bid)
            if not formats:
                continue
            fmt = pick_format(formats, settings.format_priority)
            if fmt is None:
                continue
            self.store.add_dirty(bid, fmt, 'reindex')
            queued += 1
        from calibre.gui2 import info_dialog

        if queued:
            info_dialog(self.gui, _('Semantic search'), f'Queued {queued} book(s) for (re-)indexing.', show=True)
        else:
            info_dialog(self.gui, _('Semantic search'), 'Nothing to do: every book is already indexed without errors.', show=True)

    def reindex_all(self):
        if not self._ensure_started():
            return
        api = self._api()
        if api is None:
            return
        from calibre.gui2 import question_dialog

        ok = question_dialog(
            self.gui,
            _('Semantic search'),
            'Re-index ALL books?\n\n'
            'This queues every book in the library for re-indexing, re-reading and re-embedding all of them. '
            'It can take a long time and uses many embedding API calls.',
            show=True,
        )
        if not ok:
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

    def _check_llm_provider(self):
        """Return True if a text-to-text AI provider is configured; show an error otherwise."""
        try:
            from calibre.ai import AICapabilities
            from calibre.ai.prefs import plugin_for_purpose

            llm = plugin_for_purpose(AICapabilities.text_to_text)
        except Exception as e:
            from calibre.gui2 import error_dialog

            error_dialog(self.gui, 'Attribute extraction', f'Could not check the AI provider:\n{e}', show=True)
            return False
        if llm is None:
            from calibre.gui2 import error_dialog

            error_dialog(
                self.gui,
                'Attribute extraction',
                'No text-to-text AI provider is configured.\n\n'
                'Set one up under Preferences > Plugins > AI Provider (e.g. an OpenAI-compatible provider), then try again.',
                show=True,
            )
            return False
        return True

    def extract_attributes_menu(self):
        """Force-run the attribute-extraction phase now and show live progress.

        Attribute extraction normally runs automatically right after indexing (see
        settings.auto_extract_attributes). This menu item re-runs it on demand: it
        resets previously-failed books, asks the indexer to run the phase, and
        raises the same status dialog used for indexing so progress is visible.
        """
        if not self._ensure_started():
            return
        if not self._check_llm_provider():
            return
        # Retry books that previously failed, then ask the indexer to run the
        # attribute phase; raise the live status dialog so progress is immediate.
        self.store.set_meta('attr_failed', '{}')
        self.indexer.request_attributes()
        self.show_status()

    def _api(self):
        try:
            return self.gui.current_db.new_api
        except Exception:
            return None
