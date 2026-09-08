import importlib.util
import os as _os
import sys as _sys
import types
import unittest

SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src')


def _install_stubs():
    """config_widget imports calibre's Qt wrapper + localization at module level;
    provide minimal stand-ins so it can be imported outside calibre."""
    qtcore = types.ModuleType('qt.core')

    class _Sig:
        def __init__(self, *a, **k):
            self.slots = []

        def connect(self, cb):
            self.slots.append(cb)

        def emit(self, *a):
            for cb in list(self.slots):
                cb(*a)

    qtcore.pyqtSignal = _Sig

    class _W:
        """Generic no-op widget: any method not defined below does nothing."""

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    class QDialog(_W):
        pass

    class QObject(_W):
        pass

    class QThread(_W):
        def start(self):
            pass

        def isRunning(self):
            return False

        def wait(self, ms=0):
            return True

    class QPushButton(_W):
        def __init__(self, text=''):
            self.text = text
            self.enabled = True
            self.clicked = _Sig()

        def setEnabled(self, v):
            self.enabled = bool(v)

        def setText(self, t):
            self.text = t

    class QLabel(_W):
        def __init__(self, text=''):
            self.text = text

        def setText(self, t):
            self.text = t

    class QComboBox(_W):
        def __init__(self):
            self._current = ''
            self.currentTextChanged = _Sig()

        def addItems(self, items):
            pass

        def setCurrentText(self, t):
            self._current = t

        def currentText(self):
            return self._current

        def set_current(self, t):
            """Test helper: change the selection and fire the signal like Qt does."""
            self._current = t
            self.currentTextChanged.emit(t)

    class QFormLayout(_W):
        FieldGrowthPolicy = types.SimpleNamespace(ExpandingFieldsGrow=1)

    class _Header:
        def setSectionResizeMode(self, *a):
            pass

    class QTableWidget(_W):
        def __init__(self, rows=0, cols=0):
            self._cols = cols
            self._grid = [[None] * cols for _ in range(rows)]

        def horizontalHeader(self):
            return _Header()

        def setItem(self, r, c, item):
            while len(self._grid) <= r:
                self._grid.append([None] * self._cols)
            self._grid[r][c] = item

        def item(self, r, c):
            return self._grid[r][c]

        def rowCount(self):
            return len(self._grid)

    class QTableWidgetItem:
        def __init__(self, text=''):
            self._text = text
            self._check = 0

        def text(self):
            return self._text

        def setCheckState(self, v):
            self._check = v

        def checkState(self):
            return self._check

    class QDialogButtonBox(_W):
        StandardButton = types.SimpleNamespace(Ok=1, Cancel=2)

        def __init__(self, *a, **k):
            self.accepted = _Sig()
            self.rejected = _Sig()

    for name in ('QCheckBox', 'QDoubleSpinBox', 'QGridLayout', 'QHBoxLayout', 'QLineEdit',
                 'QMessageBox', 'QPlainTextEdit', 'QSpinBox', 'QTabWidget', 'QTextEdit',
                 'QVBoxLayout', 'QWidget'):
        setattr(qtcore, name, type(name, (_W,), {}))

    qtcore.QDialog = QDialog
    qtcore.QObject = QObject
    qtcore.QThread = QThread
    qtcore.QPushButton = QPushButton
    qtcore.QLabel = QLabel
    qtcore.QComboBox = QComboBox
    qtcore.QFormLayout = QFormLayout
    qtcore.QTableWidget = QTableWidget
    qtcore.QTableWidgetItem = QTableWidgetItem
    qtcore.QDialogButtonBox = QDialogButtonBox
    qtcore.QHeaderView = type('QHeaderView', (), {'ResizeMode': types.SimpleNamespace(Stretch=1)})
    qtcore.QEvent = types.SimpleNamespace(Type=types.SimpleNamespace(Enter=6))
    qtcore.Qt = types.SimpleNamespace(CheckState=types.SimpleNamespace(Checked=2, Unchecked=0))

    _sys.modules['qt.core'] = qtcore

    cal = types.ModuleType('calibre')
    cal.__path__ = []
    utils_pkg = types.ModuleType('calibre.utils')
    loc = types.ModuleType('calibre.utils.localization')
    loc._ = lambda s: s
    _sys.modules.update({'calibre': cal, 'calibre.utils': utils_pkg, 'calibre.utils.localization': loc})


_install_stubs()

# Load plugin modules as a synthetic package so relative imports resolve (same
# pattern as test_dialog.py / test_indexer.py).
_pkg = types.ModuleType('sscfgpkg')
_pkg.__path__ = [SRC]
_sys.modules['sscfgpkg'] = _pkg


def _loadpkg(name):
    key = 'sscfgpkg.' + name
    if key in _sys.modules:
        return _sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _os.path.join(SRC, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'sscfgpkg'
    _sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


cfgw = _loadpkg('config_widget')
utils = _loadpkg('utils')


class TestLancedbUninstallGate(unittest.TestCase):
    """lancedb's uninstall button must be disabled while lancedb is selected as the
    vector backend, so a library cannot be left blocked by its own settings dialog."""

    def setUp(self):
        self._saved = (cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status)
        cfgw.external_deps_disabled = lambda: False
        cfgw.lancedb_status = lambda: (True, 'stub')

    def tearDown(self):
        cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status = self._saved

    def _widget(self, present, lib_backend=None, vector_backend='sqlite', **kw):
        cfgw.dep_in_root = lambda dep: dep in present
        s = utils.Settings()
        s.vector_backend = vector_backend
        return cfgw.SettingsWidget(s, library_backend=lib_backend, **kw)

    def _lancedb_row(self, w):
        return w._dep_rows['lancedb']

    def test_disabled_when_library_uses_lancedb(self):
        w = self._widget({'lancedb'}, lib_backend='lancedb')
        row = self._lancedb_row(w)
        self.assertEqual(row['button'].text, 'Uninstall...')
        self.assertFalse(row['button'].enabled)
        self.assertIn('vector backend', row['note'].text)

    def test_enabled_when_library_uses_sqlite(self):
        w = self._widget({'lancedb'}, lib_backend='sqlite')
        row = self._lancedb_row(w)
        self.assertEqual(row['button'].text, 'Uninstall...')
        self.assertTrue(row['button'].enabled)

    def test_install_stays_available_when_selected_but_missing(self):
        # selected but not installed: the button offers Install and must stay clickable
        w = self._widget(set(), lib_backend='lancedb')
        row = self._lancedb_row(w)
        self.assertEqual(row['button'].text, 'Install...')
        self.assertTrue(row['button'].enabled)

    def test_live_switch_toggles_button(self):
        w = self._widget({'lancedb'}, lib_backend='sqlite')
        row = self._lancedb_row(w)
        self.assertTrue(row['button'].enabled)
        w.i_backend.set_current('lancedb')
        self.assertFalse(row['button'].enabled)
        self.assertIn('vector backend', row['note'].text)
        w.i_backend.set_current('sqlite')
        self.assertTrue(row['button'].enabled)

    def test_no_library_context_uses_global_default(self):
        # no library open: the combo shows the global default for new libraries
        w = self._widget({'lancedb'}, lib_backend=None, vector_backend='lancedb')
        self.assertFalse(self._lancedb_row(w)['button'].enabled)
        w2 = self._widget({'lancedb'}, lib_backend=None, vector_backend='sqlite')
        self.assertTrue(self._lancedb_row(w2)['button'].enabled)

    def test_blocked_library_offers_install(self):
        # lancedb data exists but the package is missing: nothing to uninstall yet
        w = self._widget(set(), lib_backend='lancedb', blocked_dep='lancedb', blocked_has_data=True)
        row = self._lancedb_row(w)
        self.assertEqual(row['button'].text, 'Install...')
        self.assertTrue(row['button'].enabled)

    def test_numpy_coupling_still_applies(self):
        # regression: with both present and sqlite selected, only numpy is locked
        w = self._widget({'numpy', 'lancedb'}, lib_backend='sqlite')
        self.assertFalse(w._dep_rows['numpy']['button'].enabled)
        self.assertTrue(self._lancedb_row(w)['button'].enabled)


if __name__ == '__main__':
    unittest.main()
