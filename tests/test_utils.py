import sys
import unittest

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

utils = load('utils')


class TestSettingsRoundtrip(unittest.TestCase):
    def _prefs(self):
        d = {}
        return (lambda k, default=None: d.get(k, default)), (lambda k, v: d.__setitem__(k, v))


    def test_defaults_loadable(self):
        get, set_ = self._prefs()
        s = utils.load_settings(get)
        self.assertEqual(s.embed.base_url, 'http://localhost:11434')
        self.assertEqual(len(s.attributes), 7)
        self.assertTrue(all(a.enabled for a in s.attributes))
        self.assertEqual(s.search_min_score, 0.2)

    def test_save_load_roundtrip(self):
        get, set_ = self._prefs()
        s = utils.load_settings(get)
        s.embed.base_url = 'http://10.0.0.5:8080'
        s.embed.model = 'bge-m3'
        s.vector_backend = 'lancedb'
        s.target_chars = 777
        s.embed_context_tokens = 4096
        s.search_min_score = 0.35
        s.attr_mode = 'fulltext'
        s.attributes[0].enabled = False
        utils.save_settings(set_, s)

        s2 = utils.load_settings(get)
        self.assertEqual(s2.embed.base_url, 'http://10.0.0.5:8080')
        self.assertEqual(s2.embed.model, 'bge-m3')
        self.assertEqual(s2.vector_backend, 'lancedb')
        self.assertEqual(s2.target_chars, 777)
        self.assertEqual(s2.embed_context_tokens, 4096)
        self.assertEqual(s2.search_min_score, 0.35)
        self.assertEqual(s2.attr_mode, 'fulltext')
        self.assertFalse(s2.attributes[0].enabled)
        self.assertTrue(s2.attributes[1].enabled)
        # descriptions preserved from defaults for known names
        self.assertEqual(s2.attributes[1].description, utils.DEFAULT_ATTRIBUTES[1].description)

    def test_custom_attribute_preserved(self):
        get, set_ = self._prefs()
        s = utils.load_settings(get)
        s.attributes.append(utils.AttrField('my_field', 'ss_myfield', 'tags', 'custom desc', True))
        utils.save_settings(set_, s)
        s2 = utils.load_settings(get)
        names = [a.name for a in s2.attributes]
        self.assertIn('my_field', names)
        mine = next(a for a in s2.attributes if a.name == 'my_field')
        self.assertEqual(mine.type, 'tags')
        self.assertEqual(mine.description, 'custom desc')
        self.assertTrue(mine.enabled)

    def test_corrupt_json_falls_back(self):
        get, set_ = self._prefs()
        set_(utils.PREF_KEY, '{not json')
        s = utils.load_settings(get)
        self.assertEqual(s.embed.model, 'nomic-embed-text')

    def test_enabled_attributes(self):
        s = utils.Settings()
        s.attributes[0].enabled = False
        en = s.enabled_attributes()
        self.assertNotIn(s.attributes[0].name, [a.name for a in en])
        self.assertEqual(len(en), len(s.attributes) - 1)


class TestLanceInstall(unittest.TestCase):
    def test_pip_command(self):
        cmd = utils.pip_install_command()
        self.assertEqual(cmd[:4], [sys.executable, '-m', 'pip', 'install'])
        self.assertEqual(cmd[-1], 'lancedb')

    def _fake_popen(self, results):
        calls = []

        class FakeProc:
            def __init__(self, rc, lines):
                self._rc = rc
                self.stdout = iter(lines)

            def wait(self):
                return self._rc

        def popen(cmd, **kw):
            calls.append(list(cmd))
            rc, lines = results.pop(0)
            return FakeProc(rc, lines)

        return popen, calls

    def test_retry_with_user_then_success(self):
        popen, calls = self._fake_popen([(1, ['error: permission denied']), (0, ['Successfully installed lancedb'])])
        seen = []
        ok, msg = utils.install_lancedb(progress=seen.append, _popen=popen)
        self.assertTrue(ok)
        self.assertEqual(len(calls), 2)
        self.assertNotIn('--user', calls[0])
        self.assertIn('--user', calls[1])
        self.assertTrue(any('Successfully installed' in l for l in seen))

    def test_both_attempts_fail(self):
        popen, calls = self._fake_popen([(1, ['first error']), (2, ['still boom'])])
        ok, msg = utils.install_lancedb(_popen=popen)
        self.assertFalse(ok)
        self.assertEqual(len(calls), 2)
        self.assertIn('still boom', msg)

    def test_lancedb_status_shape(self):
        installed, info = utils.lancedb_status()
        self.assertIsInstance(installed, bool)
        self.assertIsInstance(info, str)


if __name__ == '__main__':
    unittest.main()
