import os as _os
import sys as _sys
import types
import importlib.util
import unittest

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


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
    )

    class _Item:
        def __init__(self, text=''):
            self._text = text
            self.data = None

        def text(self):
            return self._text

        def setData(self, role, value):
            self.data = value

    qtcore.QTableWidgetItem = _Item
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


dialog = _loadpkg('dialog')
utils = _loadpkg('utils')


class _Edit:
    def __init__(self, text='a query'):
        self._text = text

    def text(self):
        return self._text


class _Btn:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = bool(v)


class _Label:
    def __init__(self):
        self.text = ''

    def setText(self, t):
        self.text = t


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
        self.rows = 0
        self.grid = []
        self.header = _Header()

    def horizontalHeader(self):
        return self.header

    def setRowCount(self, n):
        self.rows = n
        self.grid = []

    def insertRow(self, r):
        self.grid.append([None] * 4)

    def setItem(self, row, col, item):
        self.grid[row][col] = item

    def item(self, row, col):
        return self.grid[row][col]


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
        return types.SimpleNamespace(title=f'Title {bid}', authors=[f'Author {bid}'])


def _make_dialog(min_score=0.2, api=None, books=None):
    d = object.__new__(dialog.SemanticSearchDialog)
    d.results = []
    d._meta_cache = {}
    d._sort_col = None
    d._sort_asc = False
    d.worker = None
    d.edit = _Edit()
    d.btn_search = _Btn()
    d.status_label = _Label()
    d.table = _Table()
    settings = utils.Settings()
    settings.search_min_score = min_score
    d.action = types.SimpleNamespace(get_settings=lambda: settings, _api=lambda: api)
    d.store = _Store(books)
    return d


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
        dialog.SemanticSearchDialog._on_failed(d, 'reset')
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
        dialog.SemanticSearchDialog._on_results(d, [_result()])
        self.assertIsNone(d.worker)
        self.assertTrue(d.btn_search.enabled)
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIsNotNone(d.worker)
        self.assertIsNot(d.worker, w1)

    def test_can_search_again_after_failure(self):
        d = _make_dialog()
        dialog.SemanticSearchDialog.do_search(d)
        dialog.SemanticSearchDialog._on_failed(d, 'boom')
        self.assertIsNone(d.worker)
        self.assertEqual(d.status_label.text, 'Search failed: boom')
        dialog.SemanticSearchDialog.do_search(d)
        self.assertIsNotNone(d.worker)


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
        dialog.SemanticSearchDialog._on_results(d, [_result()])
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
        dialog.SemanticSearchDialog._on_results(d, [r])
        self.assertEqual(d.results, [r])
        self.assertEqual(sum(len(row) for row in d.table.grid), 4)
        self.assertEqual(d.status_label.text, '1 matches')

    def test_with_api_uses_metadata(self):
        d = _make_dialog(api=_FakeApi())
        dialog.SemanticSearchDialog._on_results(d, [_result(7)])
        book_item = d.table.grid[0][0]
        self.assertEqual(book_item.text(), 'Title 7\nAuthor 7')
        self.assertEqual(book_item.data, 7)

    def test_capped_status(self):
        d = _make_dialog()
        results = [_result(i) for i in range(dialog.MAX_RESULTS)]
        dialog.SemanticSearchDialog._on_results(d, results)
        self.assertEqual(
            d.status_label.text, f'{dialog.MAX_RESULTS} matches — showing the top {dialog.MAX_RESULTS}; raise "Minimum match score" in settings to narrow the list'
        )


class TestHeaderSort(unittest.TestCase):
    def test_initial_indicator_is_score_desc(self):
        d = _make_dialog(api=_FakeApi())
        dialog.SemanticSearchDialog._on_results(d, [_result(1)])
        self.assertEqual(d._sort_col, 3)
        self.assertFalse(d._sort_asc)
        self.assertEqual(d.table.header.indicator, (3, 1))  # col 3, DescendingOrder

    def test_sort_by_book_is_stable(self):
        d = _make_dialog(api=_FakeApi())
        a_hi = _result(1, score=0.9)
        b = _result(2, score=0.8)
        a_lo = _result(1, score=0.7)
        dialog.SemanticSearchDialog._on_results(d, [a_hi, b, a_lo])
        dialog.SemanticSearchDialog._on_header_clicked(d, 0)
        self.assertEqual([r.book_id for r in d.results], [1, 1, 2])
        # stable: the two book-1 rows keep their previous (score) relative order
        self.assertEqual([r.score for r in d.results], [0.9, 0.7, 0.8])

    def test_sort_by_score_toggles_direction(self):
        d = _make_dialog(api=_FakeApi())
        dialog.SemanticSearchDialog._on_results(d, [_result(1, score=0.9), _result(2, score=0.8), _result(3, score=0.7)])
        dialog.SemanticSearchDialog._on_header_clicked(d, 3)  # was desc (initial) -> now asc
        self.assertEqual([r.score for r in d.results], [0.7, 0.8, 0.9])
        self.assertTrue(d._sort_asc)
        dialog.SemanticSearchDialog._on_header_clicked(d, 3)  # back to desc
        self.assertEqual([r.score for r in d.results], [0.9, 0.8, 0.7])

    def test_sort_by_chapter_case_insensitive(self):
        d = _make_dialog(api=_FakeApi())
        zebra = _result(1, score=0.9)
        zebra.chapter_label = 'Zebra'
        apple = _result(2, score=0.8)
        apple.chapter_label = 'apple'
        dialog.SemanticSearchDialog._on_results(d, [zebra, apple])
        dialog.SemanticSearchDialog._on_header_clicked(d, 1)
        self.assertEqual([r.book_id for r in d.results], [2, 1])

    def test_row_to_result_mapping_after_sort(self):
        # open_selected() maps table rows onto self.results; sorting must keep them in sync
        d = _make_dialog(api=_FakeApi())
        b = _result(2, score=0.9)
        a = _result(1, score=0.8)
        dialog.SemanticSearchDialog._on_results(d, [b, a])
        dialog.SemanticSearchDialog._on_header_clicked(d, 0)  # 'Title 1' < 'Title 2' -> reorders
        self.assertEqual([r.book_id for r in d.results], [1, 2])
        for row in range(2):
            self.assertEqual(d.table.grid[row][0].data, d.results[row].book_id)


if __name__ == '__main__':
    unittest.main()
