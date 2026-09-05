'''Dev gate for the semantic_search plugin: compile, test, build.

Usage:
    python dev.py            # compile + test + build, stop on first failure
    python dev.py compile    # byte-compile src/ and tests/
    python dev.py test       # run the unittest suite (tests/)
    python dev.py build      # write semantic_search.zip mirroring src/
'''

import os
import py_compile
import sys
import unittest
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
TESTS = os.path.join(ROOT, 'tests')
ZIP_PATH = os.path.join(ROOT, 'semantic_search.zip')


def _py_files(folder):
    out = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for fn in sorted(files):
            if fn.endswith('.py'):
                out.append(os.path.join(root, fn))
    return out


def cmd_compile():
    files = _py_files(SRC) + _py_files(TESTS)
    for path in files:
        py_compile.compile(path, doraise=True)
    print(f'compiled {len(files)} files')
    return True


def cmd_test():
    loader = unittest.TestLoader()
    suite = loader.discover(TESTS, top_level_dir=TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def _src_entries():
    entries = []
    for root, dirs, files in os.walk(SRC):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for fn in sorted(files):
            path = os.path.join(root, fn)
            arcname = os.path.relpath(path, SRC).replace(os.sep, '/')
            entries.append((arcname, path))
    return entries


def cmd_build():
    entries = _src_entries()
    arcnames = {arcname for arcname, _ in entries}
    for required in ('__init__.py', 'plugin-import-name-semantic_search.txt'):
        if required not in arcnames:
            print(f'refusing to build: {required} must sit at the src/ top level (calibre loader requirement)')
            return False
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in entries:
            zf.write(path, arcname)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        actual = set(zf.namelist())
    expected = {arcname for arcname, _ in entries}
    if actual != expected:
        print(f'ZIP mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}')
        return False
    print(f'wrote {os.path.basename(ZIP_PATH)} with {len(entries)} files (mirrors src/)')
    return True


def main(argv):
    steps = {'compile': cmd_compile, 'test': cmd_test, 'build': cmd_build}
    if not argv:
        plan = list(steps.items())
    elif len(argv) == 1 and argv[0] in steps:
        plan = [(argv[0], steps[argv[0]])]
    else:
        print(__doc__)
        return 2
    for name, fn in plan:
        print(f'== {name} ==')
        if not fn():
            print(f'FAILED: {name}')
            return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
