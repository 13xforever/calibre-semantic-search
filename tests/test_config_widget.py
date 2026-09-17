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
        """Generic no-op widget: any public method not defined below does nothing.
        Missing private attributes raise AttributeError like on a real object, so
        hasattr/getattr-with-default behave normally."""

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            if name.startswith('_'):
                raise AttributeError(name)
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
        InsertPolicy = types.SimpleNamespace(NoInsert=0)

        def __init__(self):
            self._current = ''
            self._items = []
            self.enabled = True
            self.currentTextChanged = _Sig()

        def addItems(self, items):
            self._items.extend(items)
            if not self._current and self._items:
                self._current = self._items[0]  # like Qt: the first added item is selected

        def count(self):
            return len(self._items)

        def itemText(self, i):
            return self._items[i]

        def setCurrentText(self, t):
            self._current = t

        def setEditText(self, t):
            # like an editable combo in real Qt: the edit text becomes the current text
            self._current = t

        def currentText(self):
            return self._current

        def setEnabled(self, v):
            self.enabled = bool(v)

        def set_current(self, t):
            """Test helper: change the selection and fire the signal like Qt does."""
            self._current = t
            self.currentTextChanged.emit(t)

    class QLineEdit(_W):
        def __init__(self, text=''):
            self._text = text

        def setText(self, t):
            self._text = t

        def text(self):
            return self._text

    class QPlainTextEdit(_W):
        def __init__(self, text=''):
            self._text = text

        def toPlainText(self):
            return self._text

    class QFormLayout(_W):
        FieldGrowthPolicy = types.SimpleNamespace(ExpandingFieldsGrow=1)

    class _Header:
        def setSectionResizeMode(self, *a):
            pass

    class QTableWidget(_W):
        def __init__(self, rows=0, cols=0):
            self._cols = cols
            self._grid = [[None] * cols for _ in range(rows)]
            self._cells = {}

        def horizontalHeader(self):
            return _Header()

        def setItem(self, r, c, item):
            while len(self._grid) <= r:
                self._grid.append([None] * self._cols)
            self._grid[r][c] = item

        def item(self, r, c):
            return self._grid[r][c]

        def setCellWidget(self, r, c, w):
            self._cells[(r, c)] = w

        def cellWidget(self, r, c):
            return self._cells.get((r, c))

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

    for name in ('QCheckBox', 'QDoubleSpinBox', 'QGridLayout', 'QHBoxLayout',
                 'QMessageBox', 'QSpinBox', 'QTabWidget', 'QTextEdit',
                 'QVBoxLayout', 'QWidget'):
        setattr(qtcore, name, type(name, (_W,), {}))

    class _NoopMeta(type):
        # QMessageBox is used via class-level calls (QMessageBox.critical(...)), which a
        # plain no-op widget class cannot answer; make those no-ops too.
        def __getattr__(cls, name):
            return lambda *a, **k: None

    qtcore.QMessageBox = _NoopMeta('QMessageBox', (_W,), {})

    qtcore.QLineEdit = QLineEdit
    qtcore.QPlainTextEdit = QPlainTextEdit
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


class TestZstandardUninstallGate(unittest.TestCase):
    """The compression combo and zstandard's uninstall button follow what the library
    actually stores: the combo edits the stored codec (disabled without a library
    context, on lancedb, while zstandard is missing, or while a zstd library is
    blocked), and uninstall is disabled while the chunks are stored as zstd."""

    def setUp(self):
        self._saved = (cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.zstandard_status)
        cfgw.external_deps_disabled = lambda: False
        cfgw.zstandard_status = lambda: (True, 'stub')

    def tearDown(self):
        cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.zstandard_status = self._saved

    def _widget(self, present, lib_backend=None, lib_codec=None, zstd_ok=True, **kw):
        cfgw.dep_in_root = lambda dep: dep in present
        cfgw.zstandard_status = lambda: (zstd_ok, 'stub')
        s = utils.Settings()
        return cfgw.SettingsWidget(s, library_backend=lib_backend, library_codec=lib_codec, **kw)

    def _zstd_row(self, w):
        return w._dep_rows['zstandard']

    # -- uninstall gate ----------------------------------------------------------

    def test_uninstall_disabled_when_library_stores_zstd(self):
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec='zstd')
        row = self._zstd_row(w)
        self.assertEqual(row['button'].text, 'Uninstall...')
        self.assertFalse(row['button'].enabled)
        self.assertIn('zstd', row['note'].text)

    def test_uninstall_enabled_when_library_stores_zlib(self):
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec='zlib')
        row = self._zstd_row(w)
        self.assertEqual(row['button'].text, 'Uninstall...')
        self.assertTrue(row['button'].enabled)

    def test_uninstall_enabled_when_library_uses_lancedb(self):
        # no sqlite chunk data: uninstalling cannot block this library
        w = self._widget({'zstandard'}, lib_backend='lancedb', lib_codec=None)
        row = self._zstd_row(w)
        self.assertTrue(row['button'].enabled)

    def test_install_stays_available_when_missing(self):
        w = self._widget(set(), lib_backend='sqlite', lib_codec='zstd')
        row = self._zstd_row(w)
        self.assertEqual(row['button'].text, 'Install...')
        self.assertTrue(row['button'].enabled)

    def test_blocked_zstd_library_offers_install(self):
        # zstd data exists but the package is missing: nothing to uninstall yet
        w = self._widget(set(), lib_backend='sqlite', lib_codec='zstd', blocked_dep='zstandard', blocked_has_data=True)
        row = self._zstd_row(w)
        self.assertEqual(row['button'].text, 'Install...')
        self.assertTrue(row['button'].enabled)

    # -- compression combo ---------------------------------------------------------

    def test_combo_disabled_without_library_context(self):
        w = self._widget({'zstandard'}, lib_backend=None)
        self.assertFalse(w.i_compress.enabled)
        self.assertIsNone(w.codec_choice())

    def test_combo_shows_availability_default_without_data(self):
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec=None)
        self.assertEqual(w.i_compress.currentText(), 'zstd')  # the package is available here
        self.assertTrue(w.i_compress.enabled)
        w2 = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec=None, zstd_ok=False)
        self.assertEqual(w2.i_compress.currentText(), 'zlib')
        self.assertFalse(w2.i_compress.enabled)

    def test_combo_shows_stored_codec(self):
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec='zlib')
        self.assertEqual(w.i_compress.currentText(), 'zlib')
        self.assertTrue(w.i_compress.enabled)
        self.assertEqual(w.codec_choice(), 'zlib')

    def test_combo_disabled_for_lancedb(self):
        w = self._widget({'zstandard'}, lib_backend='lancedb', lib_codec=None)
        self.assertFalse(w.i_compress.enabled)

    def test_combo_live_follows_backend_combo(self):
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec='zlib')
        self.assertTrue(w.i_compress.enabled)
        w.i_backend.set_current('lancedb')
        self.assertFalse(w.i_compress.enabled)
        w.i_backend.set_current('sqlite')
        self.assertTrue(w.i_compress.enabled)

    def test_combo_disabled_when_package_missing(self):
        # stored as zlib and the package is gone: zstd is not selectable, so there is nothing to change
        w = self._widget({'zstandard'}, lib_backend='sqlite', lib_codec='zlib', zstd_ok=False)
        self.assertFalse(w.i_compress.enabled)

    def test_combo_locked_for_blocked_zstd_library(self):
        # blocked with data: the combo shows zstd and cannot be changed until readable
        w = self._widget(set(), lib_backend='sqlite', lib_codec='zstd', blocked_dep='zstandard', blocked_has_data=True)
        self.assertEqual(w.i_compress.currentText(), 'zstd')
        self.assertFalse(w.i_compress.enabled)

    def test_lancedb_gate_still_applies(self):
        # regression: the two gates are independent — lancedb selected locks only lancedb
        w = self._widget({'zstandard', 'lancedb'}, lib_backend='lancedb', lib_codec=None)
        self.assertFalse(w._dep_rows['lancedb']['button'].enabled)
        self.assertTrue(self._zstd_row(w)['button'].enabled)


class TestAttrTableCollect(unittest.TestCase):
    """The attribute table's Enabled checkbox and Type dropdown feed the saved schema."""

    def setUp(self):
        self._saved = (cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status, cfgw.zstandard_status)
        cfgw.external_deps_disabled = lambda: False
        cfgw.lancedb_status = lambda: (True, 'stub')
        cfgw.zstandard_status = lambda: (True, 'stub')

    def tearDown(self):
        cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status, cfgw.zstandard_status = self._saved

    def _widget(self):
        cfgw.dep_in_root = lambda dep: False
        return cfgw.SettingsWidget(utils.Settings())

    def test_defaults_collected(self):
        w = self._widget()
        defaults = utils.Settings().attributes
        w._collect_and_accept()
        attrs = w.settings().attributes
        self.assertEqual(len(attrs), len(defaults))
        # checkbox states, types and languages round-trip the defaults
        self.assertEqual([a.enabled for a in attrs], [a.enabled for a in defaults])
        self.assertEqual([a.type for a in attrs], [a.type for a in defaults])
        self.assertEqual([a.language for a in attrs], [a.language for a in defaults])

    def test_enabled_checkbox_and_type_combo(self):
        w = self._widget()
        # disable the first field and switch its type via the dropdown
        w.attr_table.item(0, 0).setCheckState(cfgw.Qt.CheckState.Unchecked)
        w.attr_table.cellWidget(0, 2).setCurrentText('tags')
        w._collect_and_accept()
        a = w.settings().attributes[0]
        self.assertFalse(a.enabled)
        self.assertEqual(a.type, 'tags')
        # the other rows are untouched
        self.assertTrue(w.settings().attributes[1].enabled)

    def test_type_combo_items(self):
        w = self._widget()
        combo = w.attr_table.cellWidget(0, 2)
        items = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(items, ['text', 'category', 'tags'])

    def test_language_combo_items_and_blurb_default(self):
        w = self._widget()
        combo = w.attr_table.cellWidget(0, 3)
        items = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(items, ['Default', 'Match book'])
        # the blurb row ships with match-book selected
        row = next(r for r in range(w.attr_table.rowCount()) if w.attr_table.item(r, 1).text() == 'blurb')
        self.assertEqual(w.attr_table.cellWidget(row, 3).currentText(), 'Match book')

    def test_language_collect_mappings(self):
        # typed free text is kept as a forced language name
        w = self._widget()
        w.attr_table.cellWidget(0, 3).setCurrentText('russian')
        w._collect_and_accept()
        self.assertEqual(w.settings().attributes[0].language, 'russian')
        # the presets map to their sentinels
        w2 = self._widget()
        w2.attr_table.cellWidget(0, 3).setCurrentText('Match book')
        w2._collect_and_accept()
        self.assertEqual(w2.settings().attributes[0].language, 'book')
        w3 = self._widget()
        w3.attr_table.cellWidget(0, 3).setCurrentText('Default')
        w3._collect_and_accept()
        self.assertEqual(w3.settings().attributes[0].language, '')

    def test_add_attr_defaults(self):
        w = self._widget()
        n = w.attr_table.rowCount()
        w._add_attr()
        self.assertEqual(w.attr_table.rowCount(), n + 1)
        self.assertEqual(w.attr_table.item(n, 1).text(), 'new_field')
        self.assertEqual(w.attr_table.item(n, 0).checkState(), cfgw.Qt.CheckState.Checked)
        self.assertEqual(w.attr_table.cellWidget(n, 2).currentText(), 'text')
        self.assertEqual(w.attr_table.cellWidget(n, 3).currentText(), 'Default')


class TestTemplateKwargsSetting(unittest.TestCase):
    """The extra template kwargs field round-trips into the saved settings, and a value
    that is not a JSON object blocks accept so a bad setting can never be persisted."""

    def setUp(self):
        self._saved = (cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status, cfgw.zstandard_status)
        cfgw.external_deps_disabled = lambda: False
        cfgw.lancedb_status = lambda: (True, 'stub')
        cfgw.zstandard_status = lambda: (True, 'stub')

    def tearDown(self):
        cfgw.dep_in_root, cfgw.external_deps_disabled, cfgw.lancedb_status, cfgw.zstandard_status = self._saved

    def _widget(self, **kw):
        cfgw.dep_in_root = lambda dep: False
        s = utils.Settings()
        for k, v in kw.items():
            setattr(s, k, v)
        return cfgw.SettingsWidget(s)

    def test_init_from_settings(self):
        w = self._widget(attr_template_kwargs='{"enable_thinking": false}')
        self.assertEqual(w.e_template_kwargs.text(), '{"enable_thinking": false}')

    def test_collect_roundtrip_strips_whitespace(self):
        w = self._widget()
        w.e_template_kwargs.setText('  {"reasoning_effort": "low"}  ')
        w._collect_and_accept()
        self.assertEqual(w.settings().attr_template_kwargs, '{"reasoning_effort": "low"}')

    def test_clearing_collects_empty(self):
        w = self._widget(attr_template_kwargs='{"a": 1}')
        w.e_template_kwargs.setText('   ')
        w._collect_and_accept()
        self.assertEqual(w.settings().attr_template_kwargs, '')

    def test_invalid_json_blocks_accept(self):
        w = self._widget()
        w.e_template_kwargs.setText('not json')
        w._collect_and_accept()
        self.assertFalse(hasattr(w, '_result'))  # the dialog was not accepted
        self.assertEqual(w.settings().attr_template_kwargs, '')  # still the loaded settings

    def test_non_object_json_blocks_accept(self):
        for raw in ('[1, 2]', '42', '"no"'):
            w = self._widget()
            w.e_template_kwargs.setText(raw)
            w._collect_and_accept()
            self.assertFalse(hasattr(w, '_result'))


if __name__ == '__main__':
    unittest.main()
