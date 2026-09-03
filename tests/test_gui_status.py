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


if __name__ == '__main__':
    unittest.main()
