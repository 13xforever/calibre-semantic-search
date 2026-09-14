import os as _os
import shutil
import sys
import sys as _sys
import tempfile
import unittest

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

    def test_exposed_roundtrip(self):
        get, set_ = self._prefs()
        s = utils.load_settings(get)
        s.attributes[0].exposed = False
        utils.save_settings(set_, s)
        s2 = utils.load_settings(get)
        self.assertFalse(s2.attributes[0].exposed)
        self.assertTrue(s2.attributes[1].exposed)

    def test_legacy_json_without_exposed_defaults_true(self):
        # prefs written before the 'exposed' flag existed must keep mirroring on
        get, set_ = self._prefs()
        import json as _json

        data = {'attributes': [{'name': 'pov', 'label': 'ss_pov', 'type': 'text', 'description': 'x', 'enabled': True}]}
        set_(utils.PREF_KEY, _json.dumps(data))
        s = utils.load_settings(get)
        pov = next(a for a in s.attributes if a.name == 'pov')
        self.assertTrue(pov.exposed)

    def test_exposed_attributes(self):
        s = utils.Settings()
        s.attributes[0].enabled = False
        s.attributes[1].exposed = False
        names = [a.name for a in s.exposed_attributes()]
        self.assertNotIn(s.attributes[0].name, names)  # disabled
        self.assertNotIn(s.attributes[1].name, names)  # not exposed
        self.assertEqual(len(names), len(s.attributes) - 2)


class _WheelBuilder:
    """Craft a minimal but valid .whl so install/uninstall can run fully offline."""

    @staticmethod
    def make(path, dist, version, pkg_files):
        import zipfile as _zf

        entries = {}
        for rel, content in pkg_files.items():
            entries[rel] = content.encode() if isinstance(content, str) else content
        di = f'{dist}-{version}.dist-info'
        entries[f'{di}/METADATA'] = f'Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n'.encode()
        record = [f'{rel},,' for rel in sorted(entries)] + [f'{di}/RECORD,,']
        entries[f'{di}/RECORD'] = ('\n'.join(record) + '\n').encode()
        with _zf.ZipFile(path, 'w', _zf.ZIP_DEFLATED) as zf:
            for arc, data in entries.items():
                zf.writestr(arc, data)


class _FakeProc:
    """Doubles subprocess.Popen for both the communicate() and stdout-iteration paths."""

    def __init__(self, rc, lines):
        self._rc = rc
        self.stdout = iter(lines)
        self.returncode = rc

    def wait(self):
        return self._rc

    def communicate(self, timeout=None):
        rest = '\n'.join(self.stdout)
        return (rest + '\n' if rest else ''), ''


def _offline_popen():
    """popen fake: passes the pip --version probe and, for `download`, writes a real
    wheel for the requested dist into the -d dir. Returns (popen, calls)."""
    calls = []

    def popen(cmd, **kw):
        cmd = list(cmd)
        calls.append(cmd)
        if '--version' in cmd:
            return _FakeProc(0, ['pip 24.0 from /x/lib (python 3.14)'])
        spec = next(a for a in cmd if '==' in a and not a.startswith('-'))
        dist, ver = spec.split('==', 1)
        outdir = cmd[cmd.index('-d') + 1]
        _WheelBuilder.make(_os.path.join(outdir, f'{dist}-{ver}-py3-none-any.whl'), dist, ver, {f'{dist}/__init__.py': f'pkg = "{dist}"\n'})
        return _FakeProc(0, [f'Saved {dist}.whl', 'Successfully downloaded 1 package'])

    return popen, calls


class TestExternalRoot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='ss-deptest-')

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_root_name_is_version_keyed(self):
        root = utils.external_deps_root(_base=self._tmp)
        tag = f'{sys.version_info.major}{sys.version_info.minor}'
        self.assertEqual(_os.path.basename(root), f'semantic-search-libs-py{tag}')
        self.assertTrue(root.startswith(self._tmp))

    def test_dep_in_root_false_when_missing(self):
        self.assertFalse(utils.dep_in_root('numpy', _base=self._tmp))


class TestExternalInstall(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='ss-deptest-')

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_install_lancedb_offline(self):
        popen, calls = _offline_popen()
        ok, msg = utils.install_dep('lancedb', _popen=popen, _base=self._tmp)
        self.assertTrue(ok, msg)
        root = utils.external_deps_root(_base=self._tmp)
        self.assertTrue(utils.dep_in_root('lancedb', _base=self._tmp))
        self.assertTrue(_os.path.isfile(_os.path.join(root, 'lancedb', '__init__.py')))
        dl = next(c for c in calls if 'download' in c)
        self.assertIn('lancedb==0.38.0', dl)
        self.assertIn('--only-binary=:all:', dl)
        self.assertIn('-c', dl)  # a constraints file pins the transitive numpy

    def test_extract_skips_already_present_dists(self):
        # numpy is already installed (and, on Windows, its .pyd would be mapped/locked)
        popen, _calls = _offline_popen()
        ok, msg = utils.install_dep('numpy', _popen=popen, _base=self._tmp)
        self.assertTrue(ok, msg)
        root = utils.external_deps_root(_base=self._tmp)

        # sentinel: re-extracting numpy would clobber this
        np_init = _os.path.join(root, 'numpy', '__init__.py')
        with open(np_init, 'w', encoding='utf-8') as fh:
            fh.write('SENTINEL\n')

        wheels = tempfile.mkdtemp(prefix='ss-wheels-')
        self.addCleanup(shutil.rmtree, wheels, ignore_errors=True)
        _WheelBuilder.make(_os.path.join(wheels, 'numpy-2.5.3-py3-none-any.whl'), 'numpy', '2.5.3', {'numpy/__init__.py': 'pkg = "numpy"\n'})
        _WheelBuilder.make(_os.path.join(wheels, 'lancedb-0.38.0-py3-none-any.whl'), 'lancedb', '0.38.0', {'lancedb/__init__.py': 'pkg = "lancedb"\n'})

        n = utils._extract_wheels(wheels, root)
        self.assertEqual(n, 1)  # only lancedb extracted; numpy (already present) skipped
        with open(np_init, encoding='utf-8') as fh:
            self.assertEqual(fh.read(), 'SENTINEL\n')  # not clobbered
        self.assertTrue(_os.path.isfile(_os.path.join(root, 'lancedb', '__init__.py')))

    def test_install_streams_progress(self):
        popen, _calls = _offline_popen()
        seen = []
        ok, _msg = utils.install_dep('zstandard', progress=seen.append, _popen=popen, _base=self._tmp)
        self.assertTrue(ok)
        self.assertTrue(any('Downloading zstandard' in l for l in seen))

    def test_install_requires_system_python(self):
        def no_popen(cmd, **kw):
            return _FakeProc(127, ['py: command not found'])

        ok, msg = utils.install_dep('numpy', _popen=no_popen, _base=self._tmp)
        self.assertFalse(ok)
        self.assertIn('No usable system Python', msg)

    def test_install_unknown_dep(self):
        ok, msg = utils.install_dep('pandas', _base=self._tmp)
        self.assertFalse(ok)
        self.assertIn('unknown dependency', msg)

    def test_install_refused_when_external_deps_disabled(self):
        from unittest import mock

        with mock.patch.object(utils, 'external_deps_disabled', return_value=True):
            ok, msg = utils.install_dep('numpy', _base=self._tmp)
        self.assertFalse(ok)
        self.assertIn('not available on macOS', msg)


class TestExternalUninstall(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='ss-deptest-')
        popen, _calls = _offline_popen()
        for dep in ('numpy', 'lancedb'):
            ok, msg = utils.install_dep(dep, _popen=popen, _base=self._tmp)
            self.assertTrue(ok, msg)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_uninstall_lancedb_keeps_numpy(self):
        root = utils.external_deps_root(_base=self._tmp)
        ok, msg = utils.uninstall_dep('lancedb', _base=self._tmp)
        self.assertTrue(ok, msg)
        utils.process_pending_removals(_base=self._tmp)  # no-op on POSIX; completes the Windows deferral
        self.assertFalse(utils.dep_in_root('lancedb', _base=self._tmp))
        self.assertTrue(utils.dep_in_root('numpy', _base=self._tmp))  # coupling: numpy survives
        self.assertTrue(_os.path.isfile(_os.path.join(root, 'numpy', '__init__.py')))
        self.assertFalse(_os.path.exists(_os.path.join(root, 'lancedb')))  # no empty dir left

    def test_uninstall_last_dep_wipes_root(self):
        root = utils.external_deps_root(_base=self._tmp)
        utils.uninstall_dep('lancedb', _base=self._tmp)
        ok, msg = utils.uninstall_dep('numpy', _base=self._tmp)
        self.assertTrue(ok, msg)
        utils.process_pending_removals(_base=self._tmp)  # no-op on POSIX; completes the Windows deferral
        self.assertFalse(_os.path.exists(root))  # empty folder removed

    def test_uninstall_absent_is_noop(self):
        ok, msg = utils.uninstall_dep('zstandard', _base=self._tmp)
        self.assertIn('nothing to remove', msg)

    def test_uninstall_refused_when_external_deps_disabled(self):
        from unittest import mock

        with mock.patch.object(utils, 'external_deps_disabled', return_value=True):
            ok, msg = utils.uninstall_dep('numpy', _base=self._tmp)
        self.assertFalse(ok)
        self.assertIn('not available on macOS', msg)

    @unittest.skipUnless(_os.name == 'nt', 'deferred uninstall is Windows-specific')
    def test_uninstall_defers_to_restart_on_windows(self):
        tmp = tempfile.mkdtemp(prefix='ss-deptest-')
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        popen, _calls = _offline_popen()
        ok, msg = utils.install_dep('numpy', _popen=popen, _base=tmp)
        self.assertTrue(ok, msg)
        root = utils.external_deps_root(_base=tmp)

        ok, msg = utils.uninstall_dep('numpy', _base=tmp)
        self.assertTrue(ok, msg)
        self.assertIn('restart', msg)  # deferred to next start, not removed in-session
        self.assertFalse(utils.dep_in_root('numpy', _base=tmp))  # dist-info dropped now -> UI flips
        self.assertTrue(_os.path.isfile(_os.path.join(root, 'numpy', '__init__.py')))  # files remain
        self.assertTrue(_os.path.isfile(_os.path.join(root, '.pending_removal.json')))

        # next calibre start: nothing is imported yet, so the deferred files can go
        utils.process_pending_removals(_base=tmp)
        self.assertFalse(_os.path.exists(root))  # last dep fully removed -> root wiped


class TestDepStatus(unittest.TestCase):
    def test_status_shape(self):
        for fn in (utils.numpy_status, utils.zstandard_status, utils.lancedb_status):
            installed, info = fn()
            self.assertIsInstance(installed, bool)
            self.assertIsInstance(info, str)

    def test_known_deps_importable_in_test_env(self):
        # the dev venv has all three pinned, so each must report present with a version
        for dep in ('numpy', 'zstandard', 'lancedb'):
            installed, info = utils._dep_status(dep)
            self.assertTrue(installed, f'{dep} should be importable in the test env')
            self.assertNotEqual(info, 'unknown')


if __name__ == '__main__':
    unittest.main()
