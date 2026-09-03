import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

attributes = load('attributes')
utils = load('utils')


class FakeStore:
    def __init__(self, chunks_by_book=None):
        self.chunks_by_book = chunks_by_book or {}
        self.meta = {}

    def set_meta(self, k, v):
        self.meta[k] = v

    def get_meta(self, k, default=None):
        return self.meta.get(k, default)

    # attributes._chunks_for_book pokes at store.conn; emulate via monkeypatch in tests


class FakeLLM:
    def __init__(self, data=None):
        self.calls = 0
        self.data = data or {}

    def generate_structured_output(self, prompt, schema, instructions=''):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(data=SimpleNamespace(**{f.name: self.data.get(f.name) for f in _FIELDS}), exception=None, error_details='')


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


class TestBuildSchema(unittest.TestCase):
    def test_annotations(self):
        cls = attributes.build_schema_class(_FIELDS)
        anns = cls.__annotations__
        self.assertIn('gender', anns)
        self.assertIn('tropes', anns)
        # defaults exist
        self.assertIsNone(cls.gender)
        self.assertIsNone(cls.tropes)


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
        attributes._chunks_for_book = lambda s, bid: ['a' * 5000, 'b' * 5000, 'c' * 100]
        try:
            values = attributes.extract_book_attributes(2, api, store, settings, llm=llm)
        finally:
            pass
        self.assertEqual(llm.calls, 2)  # two map groups
        self.assertEqual(values['gender'], 'male')


if __name__ == '__main__':
    unittest.main()
