import importlib.util
import os as _os
import sys as _sys
import types
import unittest

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SRC = _os.path.join(ROOT, 'src')
ASSETS = _os.path.join(SRC, 'assets')


def _install_qt_stubs():
    """gui.py imports calibre's Qt wrapper + a couple of calibre modules at module
    level; provide minimal stand-ins so it can be imported outside calibre."""
    qtcore = types.ModuleType('qt.core')
    for name in ('QDialog', 'QDialogButtonBox', 'QTextEdit', 'QTimer', 'QVBoxLayout'):
        setattr(qtcore, name, type(name, (), {'__init__': lambda self, *a, **k: None}))

    class _QToolButton(type('QToolButton', (), {'__init__': lambda self, *a, **k: None})):
        class ToolButtonPopupMode:
            MenuButtonPopup = 'MenuButtonPopup'

    qtcore.QToolButton = _QToolButton
    qtcore.pyqtSignal = lambda *a, **k: None

    class QIcon:
        def __init__(self, path=''):
            self.path = path

        def isNull(self):
            return not self.path

    qtcore.QIcon = QIcon
    _sys.modules['qt.core'] = qtcore

    cal = types.ModuleType('calibre')
    cal.__path__ = []
    gui2 = types.ModuleType('calibre.gui2')
    actions = types.ModuleType('calibre.gui2.actions')
    actions.InterfaceAction = type('InterfaceAction', (), {'__init__': lambda self, *a, **k: None})
    utils_pkg = types.ModuleType('calibre.utils')
    loc = types.ModuleType('calibre.utils.localization')
    loc._ = lambda s: s
    _sys.modules.update(
        {
            'calibre': cal,
            'calibre.gui2': gui2,
            'calibre.gui2.actions': actions,
            'calibre.utils': utils_pkg,
            'calibre.utils.localization': loc,
        }
    )


_install_qt_stubs()

# Load plugin modules as a synthetic package so relative imports resolve (same
# pattern as test_indexer.py).
_pkg = types.ModuleType('sspkg')
_pkg.__path__ = [SRC]
_sys.modules['sspkg'] = _pkg


def _loadpkg(name):
    key = 'sspkg.' + name
    if key in _sys.modules:
        return _sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _os.path.join(SRC, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'sspkg'
    _sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


gui = _loadpkg('gui')
utils = _loadpkg('utils')


class FakeStore:
    def __init__(self):
        self._books = [{'id': 1, 'n_chunks': 10}, {'id': 2, 'n_chunks': 5}]
        self._attrs = {1: {'gender': 'f', 'tropes': ['x']}}

    def indexed_books(self):
        return self._books

    def dirty_book_ids(self):
        return [3]

    def get_meta(self, k, d=None):
        return '{}' if k == 'failed' else (d or '')

    def get_attrs(self, bid):
        return self._attrs.get(bid, {})


class FakeApi:
    def all_book_ids(self):
        return [1, 2, 3]


def _status_lines(api):
    a = object.__new__(gui.SemanticSearchAction)
    a.store = FakeStore()
    a._last_status = {'state': 'attributes', 'done': 7, 'total': 19, 'book_id': 5}
    a._api = (lambda: api) if api is not None else (lambda: None)
    a.get_settings = lambda: utils.Settings()
    return gui.SemanticSearchAction.status_lines(a)


class TestStatusLines(unittest.TestCase):
    def test_order_and_no_crash(self):
        # Regression: status_lines once referenced `lines` before it was assigned.
        lines = _status_lines(FakeApi())
        self.assertTrue(lines, 'expected at least one line')
        self.assertEqual(lines[0], 'Books indexed: 2/3')
        self.assertTrue(lines[1].startswith('Total chunks:'))
        # live "Extracting attributes" line must precede the "Attributes stored" total
        extract = next(i for i, l in enumerate(lines) if l.startswith('Extracting attributes'))
        stored = next(i for i, l in enumerate(lines) if l.startswith('Attributes stored'))
        self.assertLess(extract, stored)
        # pending count is the last line
        self.assertTrue(lines[-1].startswith('Pending:'))

    def test_no_api_falls_back_to_indexed(self):
        lines = _status_lines(None)
        self.assertEqual(lines[0], 'Indexed books: 2')


class _FakeIconSink:
    def __init__(self):
        self.icons = []

    def setIcon(self, ic):
        self.icons.append(ic)


class TestThemeIconRefresh(unittest.TestCase):
    def _run(self, paused=None):
        a = object.__new__(gui.SemanticSearchAction)
        a.indexer = types.SimpleNamespace(paused=paused) if paused is not None else None
        a.qaction = _FakeIconSink()
        a.search_action = None
        seen = []
        a._set_pause_label = lambda p: seen.append(p)
        gui.SemanticSearchAction._apply_theme_icon(a)
        return seen

    def test_refresh_passes_unpaused_state(self):
        # palette_changed must re-request the pause/resume standard icon with
        # the current (unpaused) state so it is regenerated for the new theme
        self.assertEqual(self._run(), [False])

    def test_refresh_preserves_paused_state(self):
        self.assertEqual(self._run(True), [True])


class TestPluginIconTheme(unittest.TestCase):
    def _pick(self, dark):
        gui2 = _sys.modules['calibre.gui2']
        gui2.is_dark_theme = lambda: dark
        try:
            return gui._plugin_icon('semantic_search.png')
        finally:
            del gui2.is_dark_theme

    def test_dark_theme_picks_dark_variant(self):
        # Regression: the variant suffix must be inserted before the file
        # extension (semantic_search-for-dark-theme.png), not appended to the
        # whole filename.
        ic = self._pick(True)
        self.assertEqual(ic.path, _os.path.join(ASSETS, 'semantic_search-for-dark-theme.png'))

    def test_light_theme_picks_light_variant(self):
        ic = self._pick(False)
        self.assertEqual(ic.path, _os.path.join(ASSETS, 'semantic_search-for-light-theme.png'))


class TestPluginIconZipBranch(unittest.TestCase):
    # Installed plugins load from the ZIP via calibre's loader, which injects
    # get_icons into the module; _plugin_icon must use it with assets/ arcnames.

    def _pick(self, dark, found):
        calls = []

        class _ZipIcon:
            def __init__(self, name):
                self.name = name

            def isNull(self):
                return not found

        def fake_get_icons(name):
            calls.append(name)
            return _ZipIcon(name) if found else None

        gui2 = _sys.modules['calibre.gui2']
        gui2.is_dark_theme = lambda: dark
        old = getattr(gui, 'get_icons', None)
        gui.get_icons = fake_get_icons
        try:
            ic = gui._plugin_icon('semantic_search.png')
        finally:
            if old is None:
                del gui.get_icons
            else:
                gui.get_icons = old
            del gui2.is_dark_theme
        return ic, calls

    def test_zip_resource_used_with_assets_prefix(self):
        ic, calls = self._pick(True, found=True)
        self.assertEqual(ic.name, 'assets/semantic_search-for-dark-theme.png')
        self.assertEqual(calls, ['assets/semantic_search-for-dark-theme.png'])

    def test_falls_back_to_file_when_zip_resource_missing(self):
        ic, calls = self._pick(False, found=False)
        self.assertEqual(ic.path, _os.path.join(ASSETS, 'semantic_search-for-light-theme.png'))
        self.assertEqual(calls, ['assets/semantic_search-for-light-theme.png'])


class TestPauseIconTheme(unittest.TestCase):
    def _path(self, paused, dark):
        gui2 = _sys.modules['calibre.gui2']
        gui2.is_dark_theme = lambda: dark
        try:
            a = object.__new__(gui.SemanticSearchAction)
            ic = gui.SemanticSearchAction._pause_icon(a, paused)
            return ic.path
        finally:
            del gui2.is_dark_theme

    def test_pause_glyph_light(self):
        self.assertEqual(self._path(False, False), _os.path.join(ASSETS, 'semantic_pause-for-light-theme.png'))

    def test_pause_glyph_dark(self):
        # Qt's standard media icons are fixed near-black in both themes, so the
        # pause/resume glyphs must come from the shipped themed variants
        self.assertEqual(self._path(False, True), _os.path.join(ASSETS, 'semantic_pause-for-dark-theme.png'))

    def test_play_glyph_light(self):
        self.assertEqual(self._path(True, False), _os.path.join(ASSETS, 'semantic_play-for-light-theme.png'))

    def test_play_glyph_dark(self):
        self.assertEqual(self._path(True, True), _os.path.join(ASSETS, 'semantic_play-for-dark-theme.png'))


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, cb):
        self.slots.append(cb)


class _FakeQtAction:
    def __init__(self, text=''):
        self.text = text
        self.tip = ''
        self.icon = None
        self.triggered = _FakeSignal()

    def setToolTip(self, t):
        self.tip = t

    def setIcon(self, ic):
        self.icon = ic


class _FakeMenu:
    def __init__(self):
        self.actions = []
        self.separators = 0

    def addSeparator(self):
        self.separators += 1

    def addAction(self, text=''):
        ac = _FakeQtAction(text)
        self.actions.append(ac)
        return ac


def _details_action(current_id=5):
    a = object.__new__(gui.SemanticSearchAction)
    calls = []
    a.gui = types.SimpleNamespace(library_view=types.SimpleNamespace(current_id=current_id))
    a.reindex_book = lambda bid: calls.append(('reindex', bid))
    a.reextract_attributes_book = lambda bid: calls.append(('attrs', bid))
    return a, calls


class TestBookDetailsMenuActions(unittest.TestCase):
    def test_adds_two_actions_for_current_book(self):
        a, calls = _details_action()
        m = _FakeMenu()
        gui.SemanticSearchAction._add_book_details_actions(a, m)
        self.assertEqual(m.separators, 1)
        texts = [x.text for x in m.actions]
        self.assertIn('Re-index this book for semantic search', texts)
        self.assertIn('Re-extract attributes for this book', texts)
        # both actions must be wired to the captured book id
        for ac in m.actions:
            for slot in ac.triggered.slots:
                slot(False)
        self.assertEqual(sorted(calls), [('attrs', 5), ('reindex', 5)])

    def test_no_current_book_adds_nothing(self):
        a, calls = _details_action(current_id=None)
        m = _FakeMenu()
        gui.SemanticSearchAction._add_book_details_actions(a, m)
        self.assertEqual(m.actions, [])
        self.assertEqual(m.separators, 0)


class TestBookDetailsMenuHook(unittest.TestCase):
    def _install_fake_bd(self):
        class FakeQMenu:
            instances = []

            def __init__(self, *a, **k):
                self.actions = []
                FakeQMenu.instances.append(self)

            def addSeparator(self):
                pass

            def addAction(self, text=''):
                ac = _FakeQtAction(text)
                self.actions.append(ac)
                return ac

            def exec(self, *a, **k):
                return 'executed'

        bd = types.ModuleType('calibre.gui2.book_details')
        bd.QMenu = FakeQMenu

        def orig(view, ev, book_info, add_popup_action=False, edit_metadata=None):
            # mimic calibre: build the menu via the module global, then show it
            menu = bd.QMenu()
            return menu.exec((0, 0))

        bd.details_context_menu_event = orig
        _sys.modules['calibre.gui2.book_details'] = bd
        _sys.modules['calibre.gui2'].book_details = bd
        return bd, FakeQMenu

    def tearDown(self):
        _sys.modules.pop('calibre.gui2.book_details', None)
        if hasattr(_sys.modules['calibre.gui2'], 'book_details'):
            delattr(_sys.modules['calibre.gui2'], 'book_details')

    def test_hook_injects_actions_before_exec(self):
        bd, FakeQMenu = self._install_fake_bd()
        a, _ = _details_action(current_id=9)
        gui.SemanticSearchAction._hook_book_details_menu(a)
        self.assertTrue(bd._ss_hooked)
        self.assertEqual(bd.details_context_menu_event(None, None, None), 'executed')
        # the module's QMenu must be restored after the call returns
        self.assertIs(bd.QMenu, FakeQMenu)
        menu = FakeQMenu.instances[-1]
        texts = [x.text for x in menu.actions]
        self.assertIn('Re-index this book for semantic search', texts)
        self.assertIn('Re-extract attributes for this book', texts)
        # opening the menu again must not duplicate the items
        bd.details_context_menu_event(None, None, None)
        menu2 = FakeQMenu.instances[-1]
        self.assertEqual(sum(1 for t in (x.text for x in menu2.actions) if 'semantic search' in t), 1)

    def test_hook_is_idempotent(self):
        bd, _ = self._install_fake_bd()
        a, _ = _details_action()
        gui.SemanticSearchAction._hook_book_details_menu(a)
        first = bd.details_context_menu_event
        gui.SemanticSearchAction._hook_book_details_menu(a)
        self.assertIs(bd.details_context_menu_event, first)

    def test_unhook_restores_original(self):
        bd, _ = self._install_fake_bd()
        orig = bd.details_context_menu_event
        a, _ = _details_action()
        gui.SemanticSearchAction._hook_book_details_menu(a)
        gui.SemanticSearchAction._unhook_book_details_menu(a)
        self.assertIs(bd.details_context_menu_event, orig)
        self.assertFalse(hasattr(bd, '_ss_hooked'))


if __name__ == '__main__':
    unittest.main()
