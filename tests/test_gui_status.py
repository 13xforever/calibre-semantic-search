import os as _os
import sys as _sys
import types
import importlib.util
import unittest

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


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
_pkg.__path__ = [ROOT]
_sys.modules['sspkg'] = _pkg


def _loadpkg(name):
    key = 'sspkg.' + name
    if key in _sys.modules:
        return _sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _os.path.join(ROOT, name + '.py'))
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
        self.assertEqual(ic.path, _os.path.join(ROOT, 'semantic_search-for-dark-theme.png'))

    def test_light_theme_picks_light_variant(self):
        ic = self._pick(False)
        self.assertEqual(ic.path, _os.path.join(ROOT, 'semantic_search-for-light-theme.png'))


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
        self.assertEqual(self._path(False, False), _os.path.join(ROOT, 'semantic_pause-for-light-theme.png'))

    def test_pause_glyph_dark(self):
        # Qt's standard media icons are fixed near-black in both themes, so the
        # pause/resume glyphs must come from the shipped themed variants
        self.assertEqual(self._path(False, True), _os.path.join(ROOT, 'semantic_pause-for-dark-theme.png'))

    def test_play_glyph_light(self):
        self.assertEqual(self._path(True, False), _os.path.join(ROOT, 'semantic_play-for-light-theme.png'))

    def test_play_glyph_dark(self):
        self.assertEqual(self._path(True, True), _os.path.join(ROOT, 'semantic_play-for-dark-theme.png'))


if __name__ == '__main__':
    unittest.main()
