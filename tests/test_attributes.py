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


class ScriptedLLM(FakeLLM):
    """Serves queued responses in order (one dict per call), recording each call's
    prompt and schema. `fail_on` (1-based call number) makes that call raise."""

    def __init__(self, responses=None, data=None, fail_on=None):
        super().__init__(data)
        self.responses = list(responses) if responses is not None else []
        self.i = 0
        self.fail_on = fail_on
        self.prompts = []
        self.schemas = []

    def generate_structured_output(self, prompt, schema, instructions=''):
        from types import SimpleNamespace

        self.calls += 1
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if self.fail_on is not None and self.calls == self.fail_on:
            return SimpleNamespace(data=None, exception=RuntimeError('boom'), error_details='boom')
        raw = self.responses[self.i % len(self.responses)] if self.responses else self.data
        self.i += 1
        return SimpleNamespace(data=SimpleNamespace(**raw), exception=None, error_details='')


_FIELDS = [utils.AttrField('gender', 'ss_gender', 'text', 'g'), utils.AttrField('tropes', 'ss_tropes', 'tags', 't')]


class FakeApi:
    def __init__(self):
        self.columns = {}  # label -> {'label', 'datatype', 'is_multiple'}
        self.fields = {}
        self.created = []  # (label, name, datatype, is_multiple), in creation order
        self.deleted = []  # labels, in deletion order
        self.book_values = {}  # label -> {book_id: value}, served by get_custom

    @property
    def backend(self):
        class B:
            pass

        b = B()
        b.custom_column_label_map = self.columns
        return b

    def create_custom_column(self, label, name, datatype, is_multiple):
        self.columns[label] = {'label': label, 'datatype': datatype, 'is_multiple': is_multiple}
        self.created.append((label, name, datatype, is_multiple))

    def delete_custom_column(self, label=None, num=None):
        if label is not None and label in self.columns:
            del self.columns[label]
            self.deleted.append(label)
        elif label is not None:
            raise ValueError('No such column')

    def set_field(self, key, mapping):
        self.fields.setdefault(key, {}).update(mapping)

    def all_ids(self):
        ids = set()
        for values in self.book_values.values():
            ids.update(values)
        return sorted(ids)

    def get_custom(self, idx, label=None, num=None, index_is_id=False):
        return self.book_values.get(label, {}).get(idx)


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


class TestTextTokenBudget(unittest.TestCase):
    """The per-call text budget accounts for the schema prompt + a safety margin."""

    def _fields(self, desc='g'):
        return [utils.AttrField('gender', 'ss_gender', 'text', desc)]

    def test_longer_descriptions_shrink_budget(self):
        ctx = 163840
        short = attributes._text_token_budget(ctx, self._fields('g'))
        long = attributes._text_token_budget(ctx, self._fields('a very long field description ' * 25))
        self.assertLess(long, short)

    def test_larger_window_gives_larger_budget(self):
        f = self._fields()
        self.assertGreater(attributes._text_token_budget(163840, f), attributes._text_token_budget(8192, f))

    def test_worst_case_message_stays_under_window(self):
        # even if a full-budget group tokenizes at the safety rate (estimate undershoot),
        # prompt + text still fit inside the configured window
        ctx = 163840
        f = self._fields()
        b = attributes._text_token_budget(ctx, f)
        prefix = attributes.estimate_tokens(attributes._prompt_for('', f))
        self.assertLess(b * attributes.EST_SAFETY_FACTOR + prefix, ctx)


class TestBalancedSplit(unittest.TestCase):
    """fulltext groups are balanced (not full,full,...,leftover) and stay within budget."""

    def test_partition_is_contiguous_and_in_order(self):
        chunks = [f'chunk{i} ' + 'a' * 900 for i in range(30)]
        groups = attributes._split_for_map(chunks, group_tokens=4000)
        flat = []
        for g in groups:
            flat.extend(g.split('\n\n'))
        self.assertEqual(flat, chunks)

    def test_groups_are_evenly_sized(self):
        chunks = ['a' * 1000 for _ in range(40)]  # ~286 est tokens each
        groups = attributes._split_for_map(chunks, group_tokens=5000)
        ests = [attributes.estimate_tokens(g) for g in groups]
        self.assertGreater(len(groups), 1)
        self.assertLess(max(ests), 2 * min(ests))  # balanced, not full+leftover

    def test_no_group_exceeds_budget(self):
        chunks = ['a' * 1000 for _ in range(40)]
        budget = 5000
        groups = attributes._split_for_map(chunks, group_tokens=budget)
        for g in groups:  # each chunk (~286) is far under the budget
            self.assertLessEqual(attributes.estimate_tokens(g), budget)


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
        # window 3000 -> ~1564-token text budget, so [5000,5000,100] (~2887 est tokens) splits into 2 groups
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
        # window 3000 -> ~1564-token text budget, so [5000,5000,100] (~2887 est tokens) splits into 2 groups
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
        # window 3000 -> ~1564-token text budget, so [5000,5000,100] (~2887 est tokens) splits into 2 groups
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

    def test_extraction_mirrors_all_enabled_fields(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female', 'tropes': ['slow burn']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        values = attributes.extract_book_attributes(20, api, store, settings, llm=llm)
        # source of truth gets everything and every enabled field is mirrored into a column
        self.assertEqual(values['gender'], 'female')
        self.assertEqual(store.attrs[20]['tropes'], ['slow burn'])
        self.assertEqual(sorted(api.columns), ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.fields['#ss_gender'], {20: 'female'})
        self.assertEqual(api.fields['#ss_tropes'], {20: ['slow burn']})

    def test_extraction_skips_disabled_fields(self):
        store = FakeStore()
        api = FakeApi()
        llm = FakeLLM({'gender': 'female', 'tropes': ['slow burn']})
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attributes[1].enabled = False  # tropes: disabled, no column
        attributes._chunks_for_book = lambda s, bid: ['para one', 'para two']
        values = attributes.extract_book_attributes(20, api, store, settings, llm=llm)
        # the disabled field is not extracted or stored at all...
        self.assertEqual(values['gender'], 'female')
        self.assertNotIn('tropes', values)
        self.assertNotIn('tropes', store.attrs[20])
        # ...and gets no column either
        self.assertIn('ss_gender', api.columns)
        self.assertNotIn('ss_tropes', api.columns)
        self.assertEqual(api.fields['#ss_gender'], {20: 'female'})
        self.assertNotIn('#ss_tropes', api.fields)


class TestFulltextReduce(unittest.TestCase):
    """fulltext mode merges disagreeing text partials with one reduce LLM call."""

    def _settings(self, context_tokens=3000):
        settings = utils.Settings()
        settings.attributes = [f.clone() for f in _FIELDS]
        settings.attr_mode = 'fulltext'
        # window 3000 -> ~1564-token text budget, so [5000,5000,100] (~2887 est tokens) splits into 2 groups
        settings.attr_context_tokens = context_tokens
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        return settings

    def test_identical_text_partials_skip_reduce(self):
        store, api = FakeStore(), FakeApi()
        llm = ScriptedLLM([{'gender': 'first person', 'tropes': ['x']}, {'gender': 'first person', 'tropes': ['y']}])
        values = attributes.extract_book_attributes(1, api, store, self._settings(), llm=llm)
        self.assertEqual(llm.calls, 2)  # map only, no reduce
        self.assertEqual(values['gender'], 'first person')

    def test_case_variant_partials_skip_reduce_keeps_first_spelling(self):
        store, api = FakeStore(), FakeApi()
        llm = ScriptedLLM([{'gender': 'First Person', 'tropes': ['x']}, {'gender': 'first person', 'tropes': ['y']}])
        values = attributes.extract_book_attributes(2, api, store, self._settings(), llm=llm)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(values['gender'], 'First Person')

    def test_disagreeing_partials_run_one_reduce_call(self):
        store, api = FakeStore(), FakeApi()
        llm = ScriptedLLM([
            {'gender': 'first person', 'tropes': ['x']},
            {'gender': 'third person limited', 'tropes': ['y']},
            {'gender': 'merged pov'},
        ])
        values = attributes.extract_book_attributes(3, api, store, self._settings(), llm=llm)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(values['gender'], 'merged pov')
        prompt = llm.prompts[2]
        # both partials in book order, with the field description
        self.assertLess(prompt.index('first person'), prompt.index('third person limited'))
        self.assertIn(_FIELDS[0].description, prompt)
        # the reduce schema carries only the disagreeing field
        self.assertEqual(set(llm.schemas[2].__annotations__), {'gender'})

    def test_reduce_failure_raises(self):
        store, api = FakeStore(), FakeApi()
        llm = ScriptedLLM(
            [
                {'gender': 'first person', 'tropes': ['x']},
                {'gender': 'third person limited', 'tropes': ['y']},
                {'gender': 'unreachable'},
            ],
            fail_on=3,
        )
        with self.assertRaises(RuntimeError):
            attributes.extract_book_attributes(4, api, store, self._settings(), llm=llm)

    def test_reduce_null_falls_back_to_first_partial(self):
        store, api = FakeStore(), FakeApi()
        llm = ScriptedLLM([{'gender': 'first person', 'tropes': ['x']}, {'gender': 'third person limited', 'tropes': ['y']}, {}])
        values = attributes.extract_book_attributes(5, api, store, self._settings(), llm=llm)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(values['gender'], 'first person')

    def test_over_budget_drops_middle_portions(self):
        store, api = FakeStore(), FakeApi()
        settings = self._settings(context_tokens=1500)  # small window -> 8 one-chunk map groups + a reduce that drops middle portions
        attributes._chunks_for_book = lambda s, bid: ['x' * 1000] * 8  # 8 groups of ~286 tokens
        parts = [f'partial number {i} ' + 'x' * 350 for i in range(8)]
        llm = ScriptedLLM([{'gender': p, 'tropes': [f't{i}']} for i, p in enumerate(parts)] + [{'gender': 'final merged'}])
        values = attributes.extract_book_attributes(6, api, store, settings, llm=llm)
        self.assertEqual(llm.calls, 9)
        self.assertEqual(values['gender'], 'final merged')
        prompt = llm.prompts[8]
        n_portions = prompt.count('  portion ')
        self.assertLess(n_portions, 8)
        self.assertIn(parts[0], prompt)  # first kept
        self.assertIn(parts[7], prompt)  # last kept


class TestSyncColumns(unittest.TestCase):
    """sync_attribute_columns converges the library's columns on the schema."""

    def _settings(self, gender_enabled=True, tropes_enabled=True):
        s = utils.Settings()
        g = utils.AttrField('gender', 'ss_gender', 'category', 'g')
        t = utils.AttrField('tropes', 'ss_tropes', 'tags', 't')
        g.enabled = gender_enabled
        t.enabled = tropes_enabled
        s.attributes = [g, t]
        return s

    def test_creates_columns_and_backfills_from_store(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['slow burn']}
        store.attrs[2] = {'gender': '', 'tropes': []}  # empty values are skipped
        api = FakeApi()
        attributes.sync_attribute_columns(api, store, self._settings())
        # category -> single-value text; tags -> multi-value text (calibre's #tags mechanism)
        self.assertEqual(api.created, [('ss_gender', 'Gender', 'text', False), ('ss_tropes', 'Tropes', 'text', True)])
        self.assertEqual(sorted(api.columns), ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})
        self.assertEqual(api.fields['#ss_tropes'], {1: ['slow burn']})

    def test_deletes_columns_of_disabled_fields(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_gender'] = {'label': 'ss_gender', 'datatype': 'text', 'is_multiple': False}
        api.columns['ss_tropes'] = {'label': 'ss_tropes', 'datatype': 'text', 'is_multiple': True}
        attributes.sync_attribute_columns(api, store, self._settings(gender_enabled=False, tropes_enabled=False))
        self.assertEqual(api.columns, {})
        self.assertEqual(api.deleted, ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.fields, {})

    def test_mixed_enabled(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_tropes'] = {'label': 'ss_tropes', 'datatype': 'text', 'is_multiple': True}  # stale column of the disabled field
        attributes.sync_attribute_columns(api, store, self._settings(tropes_enabled=False))
        self.assertNotIn('ss_tropes', api.columns)  # deleted
        self.assertIn('ss_gender', api.columns)  # created
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})
        self.assertNotIn('#ss_tropes', api.fields)

    def test_idempotent_noop_when_converged(self):
        store = FakeStore()
        store.attrs[1] = {'gender': 'female', 'tropes': ['a']}
        api = FakeApi()
        api.columns['ss_gender'] = {'label': 'ss_gender', 'datatype': 'text', 'is_multiple': False}
        api.columns['ss_tropes'] = {'label': 'ss_tropes', 'datatype': 'text', 'is_multiple': True}
        attributes.sync_attribute_columns(api, store, self._settings())
        # re-running must not duplicate or drop anything
        attributes.sync_attribute_columns(api, store, self._settings())
        self.assertEqual(sorted(api.columns), ['ss_gender', 'ss_tropes'])
        self.assertEqual(api.created, [])
        self.assertEqual(api.deleted, [])
        self.assertEqual(api.fields['#ss_gender'], {1: 'female'})


class TestColumnConversion(unittest.TestCase):
    """ensure_columns converts a column whose stored calibre type no longer matches the field."""

    def _settings(self, name='blurb', label='ss_blurb', ftype='text'):
        s = utils.Settings()
        s.attributes = [utils.AttrField(name, label, ftype, 'd')]
        return s

    def test_text_field_creates_comments_column(self):
        api = FakeApi()
        colmap = attributes.ensure_columns(api, self._settings())
        self.assertEqual(colmap['blurb'], '#ss_blurb')
        self.assertEqual(api.created, [('ss_blurb', 'Blurb', 'comments', False)])

    def test_category_field_creates_single_text_column(self):
        api = FakeApi()
        attributes.ensure_columns(api, self._settings('pov', 'ss_pov', 'category'))
        self.assertEqual(api.created, [('ss_pov', 'Pov', 'text', False)])

    def test_matching_column_left_alone(self):
        api = FakeApi()
        api.columns['ss_blurb'] = {'label': 'ss_blurb', 'datatype': 'comments', 'is_multiple': False}
        attributes.ensure_columns(api, self._settings())
        self.assertEqual(api.created, [])
        self.assertEqual(api.deleted, [])

    def test_text_column_converted_to_comments_values_preserved(self):
        # the pre-rename default: ss_blurb existed as a single-value text column
        api = FakeApi()
        api.columns['ss_blurb'] = {'label': 'ss_blurb', 'datatype': 'text', 'is_multiple': False}
        api.book_values['ss_blurb'] = {1: 'A storm is coming.', 2: ''}
        attributes.ensure_columns(api, self._settings())
        self.assertEqual(api.deleted, ['ss_blurb'])
        self.assertEqual(api.created, [('ss_blurb', 'Blurb', 'comments', False)])
        self.assertEqual(api.fields['#ss_blurb'], {1: 'A storm is coming.'})  # empty value dropped

    def test_tags_column_converted_to_category_joins_values(self):
        api = FakeApi()
        api.columns['ss_tropes'] = {'label': 'ss_tropes', 'datatype': 'text', 'is_multiple': True}
        api.book_values['ss_tropes'] = {1: ['slow burn', 'found family'], 2: []}
        attributes.ensure_columns(api, self._settings('tropes', 'ss_tropes', 'category'))
        self.assertEqual(api.created, [('ss_tropes', 'Tropes', 'text', False)])
        self.assertEqual(api.fields['#ss_tropes'], {1: 'slow burn, found family'})

    def test_category_column_converted_to_tags_splits_single_value(self):
        api = FakeApi()
        api.columns['ss_pov'] = {'label': 'ss_pov', 'datatype': 'text', 'is_multiple': False}
        api.book_values['ss_pov'] = {1: 'first person'}
        attributes.ensure_columns(api, self._settings('pov', 'ss_pov', 'tags'))
        self.assertEqual(api.created, [('ss_pov', 'Pov', 'text', True)])
        self.assertEqual(api.fields['#ss_pov'], {1: ['first person']})


class TestLanguageNote(unittest.TestCase):
    """The per-field language setting becomes a prompt instruction."""

    def _field(self, lang=''):
        return utils.AttrField('blurb', 'ss_blurb', 'text', 'A short summary.', True, lang)

    def test_default_adds_no_note(self):
        p = attributes._prompt_for('TEXT', [self._field()])
        self.assertNotIn('Write all values in', p)

    def test_book_language_note(self):
        p = attributes._prompt_for('TEXT', [self._field('book')])
        self.assertIn('Write all values in the original language of the book text.', p)

    def test_explicit_language_note(self):
        p = attributes._prompt_for('TEXT', [self._field('Russian')])
        self.assertIn('Write all values in Russian.', p)

    def test_tags_field_gets_note_too(self):
        f = utils.AttrField('tropes', 'ss_tropes', 'tags', 'Notable tropes.', True, 'book')
        p = attributes._prompt_for('TEXT', [f])
        self.assertIn('Write all values in the original language of the book text.', p)

    def test_reduce_prompt_keeps_language(self):
        # merged prose must stay in the field's target language
        p = attributes._reduce_prompt([(self._field('Russian'), ['a', 'b'])], 4000)
        self.assertIn('Write all values in Russian.', p)


if __name__ == '__main__':
    unittest.main()
