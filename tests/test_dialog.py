import importlib.util
import os as _os
import sys as _sys
import types
import unittest

SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src')


def _install_stubs():
    """dialog.py imports calibre's Qt wrapper + localization at module level;
    provide minimal stand-ins so it can be imported outside calibre."""
    qtcore = types.ModuleType('qt.core')
    for name in ('QAbstractItemView', 'QDialog', 'QHBoxLayout', 'QHeaderView', 'QLabel', 'QLineEdit', 'QPushButton', 'QTableWidget', 'QVBoxLayout'):
        setattr(qtcore, name, type(name, (), {'__init__': lambda self, *a, **k: None}))

    class QThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def isRunning(self):
            return False

        def wait(self, ms=0):
            return True

    qtcore.QThread = QThread

    class _Sig:
        def __init__(self, *a, **k):
            self.slots = []

        def connect(self, cb):
            self.slots.append(cb)

        def emit(self, *a):
            for cb in list(self.slots):
                cb(*a)

    qtcore.pyqtSignal = _Sig
    QThread.finished = _Sig()  # real QThread exposes `finished`; stub it for the keep-alive wiring
    qtcore.Qt = types.SimpleNamespace(
        ItemDataRole=types.SimpleNamespace(UserRole=256),
        SortOrder=types.SimpleNamespace(AscendingOrder=0, DescendingOrder=1),
        Key=types.SimpleNamespace(
            Key_Up=81, Key_Down=82, Key_Home=167, Key_End=168, Key_Return=16777221, Key_Enter=65431, Key_Backspace=16777216, Key_Left=16777234, Key_Right=16777235
        ),
        KeyboardModifier=types.SimpleNamespace(NoModifier=0, ShiftModifier=0x02000000, ControlModifier=0x04000000, AltModifier=0x08000000),
    )

    class _Item:
        def __init__(self, text=''):
            self._text = text
            self.data = None
            self.tooltip = None

        def text(self):
            return self._text

        def setToolTip(self, text):
            self.tooltip = text

        def setData(self, role, value):
            self.data = value

    qtcore.QTableWidgetItem = _Item
    qtcore.QTableWidget.currentCellChanged = _Sig()  # real QTableView emits it; instances connect in __init__
    qtcore.QTableWidget.selectRow = lambda self, r: None
    qtcore.QAbstractItemView.ScrollHint = types.SimpleNamespace(EnsureVisible=0, PositionAtTop=1, PositionAtCenter=2)
    _sys.modules['qt.core'] = qtcore

    cal = types.ModuleType('calibre')
    cal.__path__ = []
    utils_pkg = types.ModuleType('calibre.utils')
    loc = types.ModuleType('calibre.utils.localization')
    loc._ = lambda s: s
    _sys.modules.update({'calibre': cal, 'calibre.utils': utils_pkg, 'calibre.utils.localization': loc})


_install_stubs()

# Load plugin modules as a synthetic package so relative imports resolve (same
# pattern as test_indexer.py / test_gui_status.py).
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


dialog = _loadpkg('dialog')
utils = _loadpkg('utils')


class _Edit:
    def __init__(self, text='a query'):
        self._text = text

    def text(self):
        return self._text


class _Btn:
    def __init__(self, text=''):
        self.enabled = True
        self.visible = True
        self.text = text

    def setEnabled(self, v):
        self.enabled = bool(v)

    def setVisible(self, v):
        self.visible = bool(v)

    def setText(self, t):
        self.text = t


class _Label:
    def __init__(self):
        self.text = ''
        self.visible = True

    def setText(self, t):
        self.text = t

    def setVisible(self, v):
        self.visible = bool(v)


class _Header:
    def __init__(self):
        self.indicator = None
        self.shown = False

    def setSortIndicatorShown(self, v):
        self.shown = bool(v)

    def setSortIndicator(self, col, order):
        self.indicator = (col, order)


class _Table:
    def __init__(self):
        self.grid = []
        self.labels = []
        self.col_widths = {}
        self.header = _Header()
        self._current_cell = None
        self._current_row = -1
        self.focused = False
        self.selected_rows = []
        self.scrolled_to = None

    def horizontalHeader(self):
        return self.header

    def setRowCount(self, n):
        self.grid = []

    def insertRow(self, r):
        self.grid.append([None] * 3)

    def setItem(self, row, col, item):
        self.grid[row][col] = item

    def item(self, row, col):
        return self.grid[row][col]

    def setHorizontalHeaderLabels(self, labels):
        self.labels = list(labels)

    def setColumnWidth(self, col, w):
        self.col_widths[col] = w

    def setCurrentCell(self, r, c):
        self._current_cell = (r, c)
        self._current_row = r

    def selectionModel(self):
        return None

    def currentRow(self):
        return self._current_row

    def setFocus(self):
        self.focused = True

    def selectRow(self, r):
        self.selected_rows.append(r)

    def scrollToItem(self, item, hint=None):
        for r, row_items in enumerate(self.grid):
            for c, it in enumerate(row_items):
                if it is item:
                    self.scrolled_to = (r, c)
                    return


class _Store:
    def __init__(self, books=None):
        self._books = [{'id': 1, 'n_chunks': 5}] if books is None else books

    def indexed_books(self):
        return self._books


def _result(i=0, score=0.5):
    r = types.SimpleNamespace()
    r.book_id = i
    r.fmt = 'EPUB'
    r.chunk_no = 0
    r.text = f'some matched text {i}'
    r.chapter_path = ['Chapter One']
    r.chapter_label = 'Chapter One'
    r.para_start = 0
    r.para_end = 2
    r.char_offset = 0
    r.score = score
    return r


class _FakeApi:
    def get_metadata(self, bid):
        return types.SimpleNamespace(title=f'Title {bid}')


def _make_dialog(min_score=0.2, api=None, books=None):
    d = object.__new__(dialog.SemanticSearchDialog)
    d.books = []
    d.matches = None
    d._active_book_id = None
    d._book_cursor = 0
    d._query_ctx = None
    d._sort_state = {'books': [2, False], 'matches': [2, False]}
    d._meta_cache = {}
    d._gen = 0
    d.worker = None
    d.matches_worker = None
    d.edit = _Edit()
    d.btn_search = _Btn()
    d.status_label = _Label()
    d.book_label = _Label()
    d.count_label = _Label()
    d.btn_back = _Btn()
    d.btn_context = _Btn('Show &book matches')
    d.table = _Table()
    settings = utils.Settings()
    settings.search_min_score = min_score
    d.action = types.SimpleNamespace(get_settings=lambda: settings, _api=lambda: api)
    d.store = _Store(books)
    return d


def _searched(d, results):
    """Run do_search and deliver `results` synchronously (with a query vector)."""
    dialog.SemanticSearchDialog.do_search(d)
    w = d.worker
    w.query_vec = [1.0] * 4
    dialog.SemanticSearchDialog._on_results(d, results, d._gen, w)
    return w


class TestDoSearch(unittest.TestCase):
    def test_creates_worker_with_settings(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        w = d.worker
        self.assertIsNotNone(w)
        self.assertEqual(w.query, 'a query')
        self.assertEqual(w.limit, dialog.MAX_RESULTS)
        self.assertAlmostEqual(w.min_score, 0.2)
        self.assertFalse(d.btn_search.enabled)
        self.assertEqual(d.status_label.text, 'Searching...')

    def test_min_score_from_settings(self):
        d = _make_dialog(min_score=0.45)
        dialog.SemanticSearchDialog.do_search(d)
        self.assertAlmostEqual(d.worker.min_score, 0.45)

    def test_min_score_clamped_to_unit_range(self):
        d = _make_dialog(min_score=5.0)
        dialog.SemanticSearchDialog.do_search(d)
        self.assertEqual(d.worker.min_score, 1.0)
        dialog.SemanticSearchDialog._on_failed(d, 'reset', d._gen)
        d.edit = _Edit('another query')
        d2_settings = d.action.get_settings()
        d2_settings.search_min_score = -1.0
        dialog.SemanticSearchDialog.do_search(d)
        self.assertEqual(d.worker.min_score, 0.0)

    def test_blocked_while_worker_alive(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        w1 = d.worker
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIs(d.worker, w1)

    def test_no_indexed_books_shows_hint(self):
        d = _make_dialog(books=[])
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIsNone(d.worker)
        self.assertIn('No books indexed', d.status_label.text)


class TestRepeatSearch(unittest.TestCase):
    """Regression: a finished worker used to block every subsequent search."""

    def test_can_search_again_after_results(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        w1 = d.worker
        dialog.SemanticSearchDialog._on_results(d, [_result()], d._gen, w1)
        self.assertIsNone(d.worker)
        self.assertTrue(d.btn_search.enabled)
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIsNotNone(d.worker)
        self.assertIsNot(d.worker, w1)

    def test_can_search_again_after_failure(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        dialog.SemanticSearchDialog._on_failed(d, 'boom', d._gen)
        self.assertIsNone(d.worker)
        self.assertEqual(d.status_label.text, 'Search failed: boom')
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIsNotNone(d.worker)


class TestStaleResults(unittest.TestCase):
    """Regression: the previous search's rows used to linger in the table until the new
    results arrived."""

    def test_old_rows_cleared_when_searching_again(self):
        d = _make_dialog()
        _searched(d, [_result(1), _result(2)])
        self.assertEqual(len(d.table.grid), 2)
        dialog.SemanticSearchDialog.do_search(d)
        self.assertEqual(d.books, [])
        self.assertEqual(len(d.table.grid), 0)

    def test_stale_generation_is_ignored(self):
        d = _make_dialog()
        w = _searched(d, [_result(1)])
        dialog.SemanticSearchDialog.do_search(d)  # bumps the generation
        stale_gen = d._gen - 1
        dialog.SemanticSearchDialog._on_results(d, [_result(9)], stale_gen, w)
        self.assertEqual(d.books, [])
        self.assertIsNotNone(d.worker)


class TestHomeEndRowJump(unittest.TestCase):
    """Regression: Qt's default Home/End moves the current *cell* between columns (invisible
    under SelectRows); rows are the unit of interaction, so Home/Ctrl+Home and End/Ctrl+End
    must jump the row selection to the top/bottom row."""

    def _table(self, n_rows):
        t = dialog._ResultsTable(0, 3)
        t.rowCount = lambda: n_rows
        t.current = None
        t.setCurrentCell = lambda r, c: setattr(t, 'current', (r, c))
        return t

    def _event(self, key, modifiers=0):
        e = types.SimpleNamespace()
        e.key = lambda: key
        e.modifiers = lambda: modifiers
        e.accepted = False
        e.accept = lambda: setattr(e, 'accepted', True)
        return e

    def test_home_jumps_to_first_row(self):
        t = self._table(5)
        e = self._event(dialog.Qt.Key.Key_Home)
        dialog._ResultsTable.keyPressEvent(t, e)
        self.assertEqual(t.current, (0, 0))
        self.assertTrue(e.accepted)

    def test_end_jumps_to_last_row(self):
        t = self._table(5)
        e = self._event(dialog.Qt.Key.Key_End)
        dialog._ResultsTable.keyPressEvent(t, e)
        self.assertEqual(t.current, (4, 0))
        self.assertTrue(e.accepted)

    def test_ctrl_variants_match(self):
        ctrl = dialog.Qt.KeyboardModifier.ControlModifier
        t = self._table(5)
        dialog._ResultsTable.keyPressEvent(t, self._event(dialog.Qt.Key.Key_Home, ctrl))
        self.assertEqual(t.current, (0, 0))
        t = self._table(5)
        dialog._ResultsTable.keyPressEvent(t, self._event(dialog.Qt.Key.Key_End, ctrl))
        self.assertEqual(t.current, (4, 0))

    def test_empty_table_is_noop(self):
        t = self._table(0)
        e = self._event(dialog.Qt.Key.Key_Home)
        dialog._ResultsTable.keyPressEvent(t, e)
        self.assertIsNone(t.current)
        self.assertTrue(e.accepted)

    def test_other_keys_and_modifiers_pass_through(self):
        t = self._table(5)
        seen = []
        base = dialog.QTableWidget
        original = getattr(base, 'keyPressEvent', None)
        had_key_press = hasattr(base, 'keyPressEvent')
        base.keyPressEvent = lambda self, e: seen.append(e)
        try:
            shift = dialog.Qt.KeyboardModifier.ShiftModifier
            for key, mods in ((dialog.Qt.Key.Key_Up, 0), (dialog.Qt.Key.Key_Down, 0), (dialog.Qt.Key.Key_Home, shift)):
                dialog._ResultsTable.keyPressEvent(t, self._event(key, mods))
        finally:
            if had_key_press:
                base.keyPressEvent = original
            else:
                delattr(base, 'keyPressEvent')
        self.assertEqual(len(seen), 3)


class TestForwardNavigation(unittest.TestCase):
    """Enter / Right / Alt+Right on the results table open the selected book's matches
    (or the viewer in the matches view); searching is only triggered from the query box."""

    def _table(self, n_rows=2):
        t = dialog._ResultsTable(0, 3)
        t.rowCount = lambda: n_rows
        t.current = None
        t.setCurrentCell = lambda r, c: setattr(t, 'current', (r, c))
        return t

    def _event(self, key, modifiers=0):
        e = types.SimpleNamespace()
        e.key = lambda: key
        e.modifiers = lambda: modifiers
        e.accepted = False
        e.accept = lambda: setattr(e, 'accepted', True)
        return e

    def test_forward_keys_call_on_enter(self):
        cases = (
            (dialog.Qt.Key.Key_Return, 0),
            (dialog.Qt.Key.Key_Enter, 0),
            (dialog.Qt.Key.Key_Right, 0),
            (dialog.Qt.Key.Key_Right, dialog.Qt.KeyboardModifier.AltModifier),
        )
        for key, mods in cases:
            t = self._table()
            called = []
            t.on_enter = lambda: called.append(1)
            e = self._event(key, mods)
            dialog._ResultsTable.keyPressEvent(t, e)
            self.assertEqual(called, [1])
            self.assertTrue(e.accepted)

    def test_modified_forward_keys_pass_through(self):
        t = self._table()
        called = []
        t.on_enter = lambda: called.append(1)
        seen = []
        base = dialog.QTableWidget
        base.keyPressEvent = lambda self, e: seen.append(e)
        try:
            cases = (
                (dialog.Qt.Key.Key_Return, dialog.Qt.KeyboardModifier.ShiftModifier),
                (dialog.Qt.Key.Key_Right, dialog.Qt.KeyboardModifier.ShiftModifier),
                (dialog.Qt.Key.Key_Right, dialog.Qt.KeyboardModifier.AltModifier | dialog.Qt.KeyboardModifier.ControlModifier),
            )
            for key, mods in cases:
                dialog._ResultsTable.keyPressEvent(t, self._event(key, mods))
        finally:
            delattr(base, 'keyPressEvent')
        self.assertEqual(called, [])
        self.assertEqual(len(seen), 3)

    def test_enter_without_callback_passes_through(self):
        t = self._table()
        seen = []
        base = dialog.QTableWidget
        base.keyPressEvent = lambda self, e: seen.append(e)
        try:
            e = self._event(dialog.Qt.Key.Key_Return)
            dialog._ResultsTable.keyPressEvent(t, e)
        finally:
            delattr(base, 'keyPressEvent')
        self.assertEqual(seen, [e])
        self.assertFalse(e.accepted)


class TestBackNavigation(unittest.TestCase):
    """Bare Backspace / Alt+Left on the results table return to the books view."""

    def _table(self, n_rows=2):
        t = dialog._ResultsTable(0, 3)
        t.rowCount = lambda: n_rows
        t.current = None
        t.setCurrentCell = lambda r, c: setattr(t, 'current', (r, c))
        return t

    def _event(self, key, modifiers=0):
        e = types.SimpleNamespace()
        e.key = lambda: key
        e.modifiers = lambda: modifiers
        e.accepted = False
        e.accept = lambda: setattr(e, 'accepted', True)
        return e

    def test_back_keys_call_on_back(self):
        cases = (
            (dialog.Qt.Key.Key_Backspace, 0),
            (dialog.Qt.Key.Key_Left, 0),
            (dialog.Qt.Key.Key_Left, dialog.Qt.KeyboardModifier.AltModifier),
        )
        for key, mods in cases:
            t = self._table()
            called = []
            t.on_back = lambda: called.append(1)
            e = self._event(key, mods)
            dialog._ResultsTable.keyPressEvent(t, e)
            self.assertEqual(called, [1])
            self.assertTrue(e.accepted)

    def test_other_variants_pass_through(self):
        t = self._table()
        called = []
        t.on_back = lambda: called.append(1)
        seen = []
        base = dialog.QTableWidget
        base.keyPressEvent = lambda self, e: seen.append(e)
        try:
            cases = (
                (dialog.Qt.Key.Key_Backspace, dialog.Qt.KeyboardModifier.ShiftModifier),
                (dialog.Qt.Key.Key_Left, dialog.Qt.KeyboardModifier.AltModifier | dialog.Qt.KeyboardModifier.ControlModifier),
                (dialog.Qt.Key.Key_Right, 0),  # Right goes forward, never back
            )
            for key, mods in cases:
                dialog._ResultsTable.keyPressEvent(t, self._event(key, mods))
        finally:
            delattr(base, 'keyPressEvent')
        self.assertEqual(called, [])
        self.assertEqual(len(seen), 3)

    def test_back_without_callback_passes_through(self):
        t = self._table()
        seen = []
        base = dialog.QTableWidget
        base.keyPressEvent = lambda self, e: seen.append(e)
        try:
            e = self._event(dialog.Qt.Key.Key_Backspace)
            dialog._ResultsTable.keyPressEvent(t, e)
        finally:
            delattr(base, 'keyPressEvent')
        self.assertEqual(seen, [e])
        self.assertFalse(e.accepted)


class TestRowSelectionFollowsCursor(unittest.TestCase):
    """Tab focus lands on a single cell; the whole current row must stay selected."""

    def _table(self):
        t = dialog._ResultsTable(0, 3)
        t.selected_rows = []
        t.selectRow = lambda r: t.selected_rows.append(r)
        return t

    def test_focus_in_landing_selects_full_row(self):
        # calibre's Qt binding marshals the signal as (row, col, prev_row, prev_col) ints
        t = self._table()
        t.currentCellChanged.emit(0, 0, -1, -1)
        self.assertEqual(t.selected_rows, [0])

    def test_cursor_movement_keeps_row_selected(self):
        t = self._table()
        for row in (3, 7):
            t.currentCellChanged.emit(row, 0, row - 1, 0)
        self.assertEqual(t.selected_rows, [3, 7])

    def test_qmodelindex_shape_is_handled_too(self):
        # stock PyQt marshals the signal as (QModelIndex, QModelIndex)
        t = self._table()
        t.currentCellChanged.emit(types.SimpleNamespace(row=lambda: 5), None)
        self.assertEqual(t.selected_rows, [5])

    def test_invalid_index_selects_nothing(self):
        t = self._table()
        t.currentCellChanged.emit(-1, -1, -1, -1)
        self.assertEqual(t.selected_rows, [])


class TestWorkerLifetime(unittest.TestCase):
    """Regression: clearing self.worker on results used to drop the last reference to the
    QThread while it was still running, aborting calibre with 'QThread: Destroyed while
    thread is still running'. The worker must stay alive until its `finished` signal."""

    def test_worker_kept_alive_until_finished(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        w = d.worker
        self.assertIn(w, dialog._live_workers)
        # results arrive -> slot cleared, but the thread must still be kept alive
        dialog.SemanticSearchDialog._on_results(d, [_result()], d._gen, w)
        self.assertIsNone(d.worker)
        self.assertIn(w, dialog._live_workers)
        # the thread finishes -> released for GC (now safe to delete)
        w.finished.emit()
        self.assertNotIn(w, dialog._live_workers)


class TestOnResults(unittest.TestCase):
    def test_without_api_does_not_crash(self):
        # Regression: with _api() returning None the meta cache was never filled,
        # so the lookup below raised KeyError for every row.
        d = _make_dialog()
        r = _result()
        _searched(d, [r])
        self.assertEqual(d.books, [r])
        self.assertEqual(len(d.table.grid), 1)
        self.assertEqual(sum(item is not None for row in d.table.grid for item in row), 3)
        self.assertEqual(d.count_label.text, '1 books')

    def test_with_api_uses_metadata(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(7)])
        book_item = d.table.grid[0][0]
        self.assertEqual(book_item.text(), 'Title 7')
        self.assertEqual(book_item.tooltip, 'Title 7')

    def test_book_cell_is_single_line_title(self):
        # Qt's item views paint only the first line of multi-line item text, so the
        # book cell must never carry a '\n' (the author line would be silently dropped).
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(7)])
        self.assertNotIn('\n', d.table.grid[0][0].text())

    def test_capped_status(self):
        # hitting the row cap looks like any other count: no special-casing
        d = _make_dialog()
        results = [_result(i) for i in range(dialog.MAX_RESULTS)]
        _searched(d, results)
        self.assertEqual(d.count_label.text, f'{dialog.MAX_RESULTS} books')


class TestHeaderSort(unittest.TestCase):
    def test_initial_indicator_is_score_desc(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        self.assertEqual(d._sort_state['books'], [2, False])
        self.assertEqual(d.table.header.indicator, (2, 1))  # col 2, DescendingOrder

    def test_sort_by_book_is_stable(self):
        d = _make_dialog(api=_FakeApi())
        a_hi = _result(1, score=0.9)
        b = _result(2, score=0.8)
        a_lo = _result(1, score=0.7)
        _searched(d, [a_hi, b, a_lo])
        dialog.SemanticSearchDialog._on_header_clicked(d, 0)
        self.assertEqual([r.book_id for r in d.books], [1, 1, 2])
        # stable: the two book-1 rows keep their previous (score) relative order
        self.assertEqual([r.score for r in d.books], [0.9, 0.7, 0.8])

    def test_sort_by_score_toggles_direction(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1, score=0.9), _result(2, score=0.8), _result(3, score=0.7)])
        dialog.SemanticSearchDialog._on_header_clicked(d, 2)  # was desc (initial) -> now asc
        self.assertEqual([r.score for r in d.books], [0.7, 0.8, 0.9])
        self.assertTrue(d._sort_state['books'][1])
        dialog.SemanticSearchDialog._on_header_clicked(d, 2)  # back to desc
        self.assertEqual([r.score for r in d.books], [0.9, 0.8, 0.7])

    def test_sort_by_chapter_case_insensitive(self):
        d = _make_dialog(api=_FakeApi())
        zebra = _result(1, score=0.9)
        zebra.chapter_label = 'Zebra'
        apple = _result(2, score=0.8)
        apple.chapter_label = 'apple'
        d.matches = [zebra, apple]
        dialog.SemanticSearchDialog._render_table(d)
        dialog.SemanticSearchDialog._on_header_clicked(d, 0)
        self.assertEqual([r.book_id for r in d.matches], [2, 1])

    def test_row_to_result_mapping_after_sort(self):
        # open_selected() maps table rows onto the view's rows; sorting must keep them in sync
        d = _make_dialog(api=_FakeApi())
        b = _result(2, score=0.9)
        a = _result(1, score=0.8)
        _searched(d, [b, a])
        dialog.SemanticSearchDialog._on_header_clicked(d, 0)  # 'Title 1' < 'Title 2' -> reorders
        self.assertEqual([r.book_id for r in d.books], [1, 2])
        for row in range(2):
            self.assertEqual(d.table.grid[row][0].text(), f'Title {d.books[row].book_id}')


class TestDrillDown(unittest.TestCase):
    """The matches view: drill into a book, load its full match list, go back."""

    def test_drill_creates_matches_worker(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1), _result(2)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        w = d.matches_worker
        self.assertIsNotNone(w)
        self.assertEqual(w.book_id, 1)
        self.assertEqual(w.query_vec, [1.0] * 4)
        self.assertAlmostEqual(w.min_score, 0.2)
        self.assertEqual(d._active_book_id, 1)
        self.assertEqual(d._book_cursor, 0)
        self.assertEqual(d.book_label.text, 'Title 1')
        # books rows stay on screen while the matches load
        self.assertEqual(d.count_label.text, 'Loading matches...')
        self.assertTrue(d.btn_back.visible)
        self.assertFalse(d.status_label.visible)

    def test_book_results_switch_to_matches_view(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1), _result(2)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        m1, m2 = _result(1, score=0.9), _result(1, score=0.4)
        dialog.SemanticSearchDialog._on_book_results(d, [m1, m2], d._gen)
        self.assertIsNone(d.matches_worker)
        self.assertEqual(d.matches, [m1, m2])
        self.assertEqual(d.count_label.text, '2 matches')
        self.assertEqual(d.btn_context.text, 'Open in &viewer')
        self.assertEqual(d.table.labels[0], 'Chapter')
        self.assertEqual(d._sort_state['matches'], [2, False])

    def test_go_back_restores_books_view(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1), _result(2)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        dialog.SemanticSearchDialog._on_book_results(d, [_result(1)], d._gen)
        gen = d._gen
        dialog.SemanticSearchDialog.go_back(d)
        self.assertIsNone(d._active_book_id)
        self.assertIsNone(d.matches)
        self.assertGreater(d._gen, gen)  # in-flight results must be dropped
        self.assertEqual([r.book_id for r in d.books], [1, 2])
        self.assertEqual(d.table.labels[0], 'Book')
        self.assertEqual(d.count_label.text, '2 books')
        self.assertFalse(d.btn_back.visible)
        self.assertEqual(d.table._current_cell, (0, 0))

    def test_stale_book_results_dropped(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        stale_gen = d._gen - 1
        dialog.SemanticSearchDialog._on_book_results(d, [_result(1)], stale_gen)
        self.assertIsNone(d.matches)

    def test_book_results_focus_first_row(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        dialog.SemanticSearchDialog._on_book_results(d, [_result(1), _result(1)], d._gen)
        self.assertTrue(d.table.focused)
        self.assertEqual(d.table._current_cell, (0, 0))

    def test_empty_book_results_do_not_focus(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        d.table.focused = False
        d.table._current_cell = None
        dialog.SemanticSearchDialog._on_book_results(d, [], d._gen)
        self.assertFalse(d.table.focused)
        self.assertIsNone(d.table._current_cell)

    def test_go_back_scrolls_to_restored_row(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(i) for i in range(50)])
        d.table.setCurrentCell(40, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        dialog.SemanticSearchDialog._on_book_results(d, [_result(1)] * 30, d._gen)
        d.table.setCurrentCell(25, 0)
        dialog.SemanticSearchDialog.go_back(d)
        self.assertEqual(d.table._current_cell, (40, 0))
        self.assertEqual(d.table.scrolled_to, (40, 0))

    def test_go_back_in_books_view_is_noop(self):
        # Backspace/Alt+Left land here too; without a drill-down nothing may change
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        gen = d._gen
        dialog.SemanticSearchDialog.go_back(d)
        self.assertEqual(d._gen, gen)
        self.assertEqual([r.book_id for r in d.books], [1])

    def test_book_failed_falls_back_to_books_view(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1), _result(2)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        dialog.SemanticSearchDialog._on_book_failed(d, 'boom', d._gen)
        self.assertIsNone(d.matches_worker)
        self.assertIsNone(d._active_book_id)
        self.assertIsNone(d.matches)
        self.assertEqual(d.status_label.text, 'Loading matches failed: boom')
        self.assertEqual([r.book_id for r in d.books], [1, 2])


class TestOpenSelected(unittest.TestCase):
    def test_books_view_drills(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.open_selected(d)
        self.assertIsNotNone(d.matches_worker)

    def test_matches_view_opens_result(self):
        d = _make_dialog(api=_FakeApi())
        _searched(d, [_result(1)])
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.drill_into(d)
        m = _result(1)
        dialog.SemanticSearchDialog._on_book_results(d, [m], d._gen)
        opened = []
        d.open_result = lambda r: opened.append(r)
        d.table.setCurrentCell(0, 0)
        dialog.SemanticSearchDialog.open_selected(d)
        self.assertEqual(opened, [m])


class TestRestrictLibrary(unittest.TestCase):
    def test_uses_books_view_ids(self):
        d = _make_dialog()
        calls = []
        d.gui = types.SimpleNamespace(
            search=types.SimpleNamespace(set_search_string=lambda q, store_in_history=False: calls.append((q, store_in_history)))
        )
        d.books = [_result(2), _result(1)]
        dialog.SemanticSearchDialog.restrict_library(d)
        self.assertEqual(calls, [('id:1 or id:2', True)])

    def test_empty_books_is_noop(self):
        d = _make_dialog()
        calls = []
        d.gui = types.SimpleNamespace(search=types.SimpleNamespace(set_search_string=lambda *a, **k: calls.append(1)))
        dialog.SemanticSearchDialog.restrict_library(d)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
