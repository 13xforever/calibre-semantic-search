'''Dev gate for the semantic_search plugin: compile, lint, test, build.

Usage:
  python dev.py            run compile + lint + test + build, stop on first failure
  python dev.py setup      create/refresh the project-local .venv and install the
                           dev dependencies (requirements-dev.txt) into it
  python dev.py compile    byte-compile src/ and tests/
  python dev.py lint       ruff over src/, tests/, dev.py (pyflakes + isort rules)
  python dev.py test       run the unittest suite
  python dev.py build      write semantic_search.zip mirroring src/
'''

import os
import py_compile
import re
import subprocess
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


def cmd_lint():
    proc = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--output-format=concise', SRC, TESTS, os.path.join(ROOT, 'dev.py')]
    )
    return proc.returncode == 0


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


GATE = {'compile': cmd_compile, 'lint': cmd_lint, 'test': cmd_test, 'build': cmd_build}


VENV_DIR = os.path.join(ROOT, '.venv')


def _venv_python():
    if os.name == 'nt':
        return os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(VENV_DIR, 'bin', 'python')


def _pick_interpreter():
    '''Closest installed Python to what calibre ships (3.14), never newer: the highest in 3.12..3.14.'''
    seen = {}
    try:
        out = subprocess.run(['py', '-0p'], capture_output=True, text=True).stdout
    except OSError:
        out = ''
    for line in out.splitlines():
        # py -0p lines look like "-V:3.14[-64] *  C:\...\python.exe": version, an
        # optional [arch] marker, an optional default-marker '*', then the path
        m = re.match(r'\s*-V:(\d+\.\d+)(?:\[[^\]]*\])?\s*(?:\*)?\s*(\S.*)$', line)
        if not m:
            continue
        ver = tuple(int(x) for x in m.group(1).split('.'))
        seen.setdefault(ver, m.group(2).strip())
    in_range = {v: p for v, p in seen.items() if (3, 12) <= v <= (3, 14)}
    if in_range:
        return in_range[max(in_range)]
    if (3, 12) <= sys.version_info[:2] <= (3, 14):
        return sys.executable
    print(f'no suitable Python found via the py launcher: need an installed '
          f'interpreter between 3.12 and 3.14 (calibre ships 3.14); '
          f'found {sorted(seen) or "none"}')
    return None


def _dev_requirements():
    """Parsed requirements-dev.txt as requirement strings (comments/blank lines removed)."""
    reqs = []
    with open(os.path.join(ROOT, 'requirements-dev.txt'), encoding='utf-8') as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if line:
                reqs.append(line)
    return reqs


def _req_satisfied(python, req):
    """True when the venv already provides `req` (exact version match when pinned)."""
    if '==' in req:
        name, want = [x.strip() for x in req.split('==', 1)]
        code = "import importlib.metadata as m; raise SystemExit(0 if m.version(%r)==%r else 1)" % (name, want)
        return subprocess.run([python, '-c', code], stdout=subprocess.DEVNULL).returncode == 0
    if req == 'ruff':
        return subprocess.run([python, '-m', 'ruff', '--version'], stdout=subprocess.DEVNULL).returncode == 0
    mod = req.replace('-', '_')
    return subprocess.run([python, '-c', f'import {mod}'], stdout=subprocess.DEVNULL).returncode == 0


def _missing_dev_deps(python):
    """Requirement strings from requirements-dev.txt that the venv does not yet satisfy."""
    return [req for req in _dev_requirements() if not _req_satisfied(python, req)]


def cmd_setup():
    '''Create/refresh the project-local .venv and install dev dependencies into it.'''
    venv_py = _venv_python()
    if os.path.exists(venv_py):
        print(f'reusing existing {os.path.basename(VENV_DIR)}')
    else:
        interpreter = _pick_interpreter()
        if interpreter is None:
            return False
        print(f'creating {os.path.basename(VENV_DIR)} from {interpreter}')
        proc = subprocess.run([interpreter, '-m', 'venv', VENV_DIR])
        if proc.returncode != 0 or not os.path.exists(venv_py):
            print('venv creation failed')
            return False
    missing = _missing_dev_deps(venv_py)
    if not missing:
        print('all dev dependencies present in venv')
    else:
        print(f'installing into venv: {", ".join(missing)}')
        proc = subprocess.run([venv_py, '-m', 'pip', 'install',
                               '--disable-pip-version-check', *missing])
        if proc.returncode != 0 or _missing_dev_deps(venv_py):
            print('dependency install failed (network problem?) - re-run when online')
            return False
    version = subprocess.run([venv_py, '--version'], capture_output=True, text=True).stdout.strip()
    print(f'environment ready: {os.path.basename(VENV_DIR)} ({version})')
    print(f'run the gate with: {os.path.relpath(venv_py, ROOT)} dev.py')
    return True


def main(argv):
    steps = dict(GATE)
    steps['setup'] = cmd_setup
    if not argv:
        plan = list(GATE.items())
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
