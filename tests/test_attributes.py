import importlib.util
import os as _os
import sys as _sys
import types
import unittest

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src')

# attributes does a relative `from .chunker import ...`, so it must be loaded as a
# synthetic package (like indexer). A distinct package name keeps this file's module
# instances isolated from other tests' copies.
_pkg = types.ModuleType('ssattrpkg')
_pkg.__path__ = [SRC]
_sys.modules['ssattrpkg'] = _pkg


def _loadpkg(name):
    key = 'ssattrpkg.' + name
    if key in _sys.modules:
        return _sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _os.path.join(SRC, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'ssattrpkg'
    _sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


attributes = _loadpkg('attributes')
utils = load('utils')


class FakeStore:
    def __init__(self, chunks_by_book=None):
        self.chunks_by_book = chunks_by_book or {}
        self.meta = {}
        self.attrs = {}

    def set_meta(self, k, v):
        self.meta[k] = v

    def get_meta(self, k, default=None):
        return self.meta.get(k, default)

    def set_attrs(self, book_id, values):
        self.attrs[book_id] = dict(values)

    def get_attrs(self, book_id):
        return self.attrs.get(book_id, {})

    def all_attrs(self):
        return {k: dict(v) for k, v in self.attrs.items()}

    # attributes._chunks_for_book pokes at store.conn; emulate via monkeypatch in tests


class FakeLLM:
    def __init__(self, data=None):
        self.calls = 0
        self.data = data or {}

    def generate_structured_output(self, prompt, schema, instructions=''):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(data=SimpleNamespace(**{f.name: self.data.get(f.name) for f in _FIELDS}), exception=None, error_details='')


class FakeLLMSequence(FakeLLM):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.i = 0

    def generate_structured_output(self, prompt, schema, instructions=''):
        from types import SimpleNamespace

        self.calls += 1
        data = self.responses[self.i % len(self.responses)]
        self.i += 1
        return SimpleNamespace(data=SimpleNamespace(**{f.name: data.get(f.name) for f in _FIELDS}), exception=None, error_details='')


_FIELDS = [utils.AttrField('gender', 'ss_gender', 'text', 'g'), utils.AttrField('tropes', 'ss_tropes', 'tags', 't')]


class FakeApi:
    def __init__(self):
        self.columns = {}
        self.fields = {}

    @property
    def backend(self):
        class B:
            pass

        b = B()
        b.custom_column_label_map = self.columns
        return b

    def create_custom_column(self, label, name, datatype, is_multiple):
        self.columns[label] = {'label': label}

    def delete_custom_column(self, label=None, num=None):
        if label is not None and label in self.columns:
            del self.columns[label]
        elif label is not None:
            raise ValueError('No such column')

    def set_field(self, key, mapping):
        self.fields.setdefault(key, {}).update(mapping)


class TestNormalize(unittest.TestCase):
    def test_text_values(self):
        self.assertEqual(attributes._normalize_value('female', 'text'), 'female')
        self.assertEqual(attributes._normalize_value(None, 'text'), '')
        self.assertEqual(attributes._normalize_value('unknown', 'text'), '')

    def test_tags_from_list_and_string(self):
        self.assertEqual(attributes._normalize_value(['a', 'b'], 'tags'), ['a', 'b'])
        self.assertEqual(attributes._normalize_value('a, b ,c', 'tags'), ['a', 'b', 'c'])
        self.assertEqual(attributes._normalize_value(None, 'tags'), [])

    def test_tags_list_coerced_for_text(self):
        self.assertEqual(attributes._normalize_value(['x', 'y'], 'text'), 'x, y')


class TestNormalizeTags(unittest.TestCase):
    def test_exact_duplicates_removed_in_order(self):
        self.assertEqual(attributes.normalize_tags(['a', 'b', 'a', 'B']), ['a', 'b'])

    def test_case_and_whitespace_variants(self):
        self.assertEqual(attributes.normalize_tags(['Slow Burn', 'slow  burn']), ['Slow Burn'])

    def test_separator_variants(self):
        self.assertEqual(
            attributes.normalize_tags(['sexual assault/rape', 'sexual assault / rape']),
            ['sexual assault/rape'],
        )
        self.assertEqual(
            attributes.normalize_tags(['bullying and harassment', 'bullying/harassment']),
            ['bullying/harassment'],
        )

    def test_trailing_parenthetical_collapses_to_base(self):
        tags = [
            'non-explicit sex scenes (brief implied encounter between Holsten and Lain)',
            'violence depicted (gun battles, combat, drone destruction)',
            'skinship moments',
            'non-explicit sex scenes',
        ]
        # the lone detailed tag keeps its display; only the duplicated base form collapses
        self.assertEqual(
            attributes.normalize_tags(tags),
            ['non-explicit sex scenes', 'violence depicted (gun battles, combat, drone destruction)', 'skinship moments'],
        )

    def test_distinct_tags_untouched(self):
        tags = ['slow burn', 'enemies to lovers', 'found family']
        self.assertEqual(attributes.normalize_tags(tags), tags)

    def test_blank_entries_dropped(self):
        self.assertEqual(attributes.normalize_tags(['  ', '', 'ok']), ['ok'])


class TestSampling(unittest.TestCase):
    def test_sample_small(self):
        chunks = ['a' * 100, 'b' * 100]
        self.assertEqual(attributes.sample_text(chunks, 1000), 'a' * 100 + '\n\n' + 'b' * 100)

    def test_sample_large(self):
        chunks = ['x' * 500 for _ in range(20)]
        out = attributes.sample_text(chunks, 1000)
        self.assertLessEqual(len(out), 1600)  # capped roughly
        self.assertIn('x', out)

    def test_map_split(self):
        chunks = ['a' * 4000, 'b' * 4000, 'c' * 400]
        groups = attributes._split_for_map(chunks, group_chars=8000)
        self.assertEqual(len(groups), 2)
        self.assertIn('a', groups[0])
        self.assertIn('b', groups[0])
        self.assertIn('c', groups[1])


class TestBudget(unittest.TestCase):
    def test_larger_context_gives_larger_budget(self):
        self.assertGreater(attributes.text_budget_chars(8192), attributes.text_budget_chars(4096))

    def test_derivation_formula(self):
        self.assertEqual(
            attributes.text_budget_chars(8192),
            int((8192 - attributes.OVERHEAD_TOKENS) * attributes.CHARS_PER_TOKEN),
        )

    def test_floor_for_tiny_context(self):
        self.assertGreaterEqual(attributes.text_budget_chars(0), attributes.MIN_TEXT_CHARS)


class TestTokenBudget(unittest.TestCase):
    def test_latin_estimate(self):
        self.assertEqual(attributes.estimate_tokens('x' * 350), 100)

    def test_cyrillic_estimate(self):
        self.assertEqual(attributes.estimate_tokens('ж' * 1000), 667)

    def test_sample_all_fits(self):
        chunks = ['ж' * 100 for _ in range(3)]  # 40 tokens each
        out = attributes.sample_text(chunks, max_tokens=1000)
        self.assertEqual(out, '\n\n'.join(chunks))

    def test_sample_cyrillic_respects_token_budget(self):
        chunks = ['ж' * 1200 for _ in range(10)]  # 800 tokens each
        out = attributes.sample_text(chunks, max_tokens=1700)
        self.assertIn('ж', out)
        self.assertLessEqual(attributes.estimate_tokens(out), 1700 + 5)

    def test_map_split_token_budget(self):
        chunks = ['ж' * 1200 for _ in range(4)]  # 800 tokens each
        groups = attributes._split_for_map(chunks, group_tokens=1700)
        self.assertEqual(len(groups), 2)
        for g in groups:
            self.assertLessEqual(attributes.estimate_tokens(g), 1700 + 5)


class TestBuildSchema(unittest.TestCase):
    def test_annotations(self):
        cls = attributes.build_schema_class(_FIELDS)
        anns = cls.__annotations__
        self.assertIn('gender', anns)
        self.assertIn('tropes', anns)
        # defaults exist
        self.assertIsNone(cls.gender)
        self.assertIsNone(cls.tropes)

    def test_accepts_kwargs_like_calibre_instantiate(self):
        # calibre.ai.structured.instantiate builds the object via cls(**parsed_json);
        # a plain class has no such __init__ and raises "takes no arguments".
        cls = attributes.build_schema_class(_FIELDS)
        obj = cls(gender='female', tropes=['slow burn'])
        self.assertEqual(obj.gender, 'female')
        self.assertEqual(list(obj.tropes), ['slow burn'])


class TestExtract(unittest.TestCase):
    def test_sampled_extraction_writes_columns(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female', 'tropes': ['slow burn']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]

        # emulate chunk fetch
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        try:
            values = attributes.extract_book_attributes(1, api, store, settings, llm=llm)
        finally:
            pass
        self.assertEqual(values['gender'], 'female')
        self.assertEqual(values['tropes'], ['slow burn'])
        # source of truth is the plugin store (attrs_raw), not the columns
        self.assertEqual(store.attrs[1]['gender'], 'female')
        self.assertEqual(store.attrs[1]['tropes'], ['slow burn'])
        self.assertIn('ss_gender', api.columns)
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})
        self.assertEqual(api.fields['#ss_tropes'], {1: ['slow burn']})

    def test_fulltext_map_reduce(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'male', 'tropes': ['x']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attr_mode = 'fulltext'
        # 3000 tokens -> ~6916 char budget, so [5000,5000,100] splits into 2 groups
        settings.attr_context_tokens = 3000
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        try:
            values = attributes.extract_book_attributes(2, api, store, settings, llm=llm)
        finally:
            pass
        self.assertEqual(llm.calls, 2)  # two map groups
        self.assertEqual(values['gender'], 'male')

    def test_sampled_extraction_dedupes_near_duplicate_tags(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female', 'tropes': ['slow burn', 'Slow Burn (implied)', 'angst/whump']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        values = attributes.extract_book_attributes(10, api, store, settings, llm=llm)
        self.assertEqual(values['tropes'], ['slow burn', 'angst/whump'])
        self.assertEqual(store.attrs[10]['tropes'], ['slow burn', 'angst/whump'])

    def test_fulltext_merge_dedupes_near_duplicate_tags(self):
        store = FakeStore()
        api = FakeApi()
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attr_mode = 'fulltext'
        # 3000 tokens -> ~6916 char budget, so [5000,5000,100] splits into 2 groups
        settings.attr_context_tokens = 3000
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        llm = FakeLLMSequence([
            {'gender': 'male', 'tropes': ['slow burn (early chapters)', 'enemies to lovers']},
            {'gender': None, 'tropes': ['Slow Burn', 'enemies to  lovers']},
        ])
        values = attributes.extract_book_attributes(11, api, store, settings, llm=llm)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(values['tropes'], ['Slow Burn', 'enemies to lovers'])
        self.assertEqual(store.attrs[11]['tropes'], ['Slow Burn', 'enemies to lovers'])

    def test_fulltext_reports_progress_per_group(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'male'})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attr_mode = 'fulltext'
        # 3000 tokens -> ~6916 char budget, so [5000,5000,100] splits into 2 groups
        settings.attr_context_tokens = 3000
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        calls = []
        attributes.extract_book_attributes(2, api, store, settings, llm=llm, progress_cb=lambda d, t: calls.append((d, t)))
        self.assertEqual(calls, [(1, 2), (2, 2)])

    def test_sampled_mode_emits_no_progress(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female'})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        calls = []
        attributes.extract_book_attributes(3, api, store, settings, llm=llm, progress_cb=lambda d, t: calls.append((d, t)))
        self.assertEqual(calls, [])

    def test_extraction_stores_but_skips_unexposed_column(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female', 'tropes': ['slow burn']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attributes[1].exposed = False  # tropes: stored internally only
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        values = attributes.extract_book_attributes(20, api, store, settings, llm=llm)
        # source of truth gets everything...
        self.assertEqual(values['gender'], 'female')
        self.assertEqual(store.attrs[20]['tropes'], ['slow burn'])
        # ...but only the exposed field is mirrored into a column
        self.assertIn('ss_gender', api.columns)
        self.assertNotIn('ss_tropes', api.columns)
        self.assertEqual(api.fields['#ss_gender'], {20: 'female'})
        self.assertNotIn('#ss_tropes', api.fields)


class TestSyncColumns(unittest.TestCase):
    """sync_attribute_columns converges the library's columns on the schema."""

    def _settings(self, gender_exposed=True, tropes_exposed=True):
        s = utils.Settings()
        g = utils.AttrField('gender', 'ss_gender', 'text', 'g')
        t = utils.AttrField('tropes', 'ss_tropes', 'tags', 't')
        t.exposed = tropes_exposed
        g.exposed = gender_exposed
        s.attributes = [g, t]
        return s

    def test_creates_columns_and_backfills_from_store(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['slow burn']}
        store.attrs[2] = {'gender': '', 'tropes': []}  # empty values are skipped
        api = FakeApi()
        attributes.sync_attribute_columns(api, store, self._settings())
        self.assertEqual(sorted(api.columns), ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})
        self.assertEqual(api.fields['#ss_tropes'], {1: ['slow burn']})

    def test_deletes_columns_of_unexposed_fields(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_gender'] = {'label': 'ss_gender'}
        api.columns['ss_tropes'] = {'label': 'ss_tropes'}
        attributes.sync_attribute_columns(api, store, self._settings(gender_exposed=False, tropes_exposed=False))
        self.assertEqual(api.columns, {})
        self.assertEqual(api.fields, {})

    def test_mixed_exposure(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_tropes'] = {'label': 'ss_tropes'}  # stale column of the un-exposed field
        attributes.sync_attribute_columns(api, store, self._settings(tropes_exposed=False))
        self.assertNotIn('ss_tropes', api.columns)  # deleted
        self.assertIn('ss_gender', api.columns)  # created
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})
        self.assertNotIn('#ss_tropes', api.fields)

    def test_idempotent_noop_when_converged(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_gender'] = {'label': 'ss_gender'}
        api.columns['ss_tropes'] = {'label': 'ss_tropes'}
        attributes.sync_attribute_columns(api, store, self._settings())
        # re-running must not duplicate or drop anything
        attributes.sync_attribute_columns(api, store, self._settings())
        self.assertEqual(sorted(api.columns), ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})


if __name__ == '__main__':
    unittest.main()
