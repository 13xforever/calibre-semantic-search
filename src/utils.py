'''Shared helpers: settings model and preferences access.'''

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass, field, fields
from typing import Any

PREF_KEY = 'semantic_search_settings'

DEFAULT_FORMAT_PRIORITY = ['EPUB', 'AZW3', 'MOBI', 'FB2', 'DOCX', 'HTML', 'TXT', 'PDF']


@dataclass
class EmbedSettings:
    base_url: str = 'http://localhost:11434'
    model: str = 'nomic-embed-text'
    api_key: str = ''
    batch_size: int = 64
    concurrency: int = 4
    timeout: int = 300


@dataclass
class AttrField:
    name: str
    label: str
    type: str  # 'text' or 'tags'
    description: str
    enabled: bool = True
    exposed: bool = True  # mirror into a calibre custom column (False = stored internally only)

    def clone(self) -> 'AttrField':
        return AttrField(**asdict(self))


DEFAULT_ATTRIBUTES = (
    AttrField(
        'main_character_gender',
        'ss_gender',
        'text',
        "Gender of the main character(s)/POV character(s), e.g. 'female', 'male', 'non-binary', or 'multiple' if there are several protagonists.",
    ),
    AttrField(
        'protagonist_orientation',
        'ss_orientation',
        'tags',
        "The romantic/sexual orientation of the main character(s) as depicted in the book, e.g. 'heterosexual', 'gay', 'lesbian', 'bisexual', 'pansexual', 'aromantic'. Only include what is actually shown or stated.",
    ),
    AttrField(
        'romance_elements',
        'ss_romance',
        'tags',
        "Romance plot elements and dynamics, e.g. 'slow burn', 'enemies to lovers', 'love triangle', 'forced proximity', 'forbidden romance', 'second chance'.",
    ),
    AttrField('tropes', 'ss_tropes', 'tags', "Notable tropes, e.g. 'chosen one', 'found family', 'reincarnation', 'island survival'."),
    AttrField('themes', 'ss_themes', 'tags', "Major themes, e.g. 'identity', 'grief', 'power and corruption', 'coming of age'."),
    AttrField('pov', 'ss_pov', 'text', "Narrative point of view, e.g. 'first person', 'third person limited', 'omniscient', 'unreliable narrator'."),
    AttrField(
        'content_warnings',
        'ss_warnings',
        'tags',
        "Content warnings for sensitive material, e.g. 'violence', 'self-harm', 'abuse', 'suicide'. Only include if actually present.",
    ),
    AttrField(
        'blurb',
        'ss_blurb',
        'text',
        "A spoiler-free marketing blurb in the style of a publisher's back cover, written in the language of the book: 1-3 short paragraphs covering the premise, the protagonist(s) and the central conflict or stakes. Never reveal plot twists, outcomes, deaths or how it ends.",
        enabled=True,
        exposed=False,
    ),
)


@dataclass
class Settings:
    embed: EmbedSettings = field(default_factory=EmbedSettings)
    vector_backend: str = 'sqlite'  # default backend for NEW libraries (a library's actual choice is stored in its store meta)
    format_priority: list[str] = field(default_factory=lambda: list(DEFAULT_FORMAT_PRIORITY))
    target_chars: int = 1000
    overlap_chars: int = 150
    embed_context_tokens: int = 8192  # embedding model max input (tokens); caps the chunk size
    max_chunks_per_book: int = 0  # 0 = unlimited
    search_min_score: float = 0.2  # only show search results with score >= this (0..1)
    attr_mode: str = 'sampled'  # sampled | fulltext (map-reduce)
    attr_context_tokens: int = 8192  # model context window (tokens); sample/group sizes derive from this
    auto_extract_attributes: bool = True  # run LLM attribute extraction automatically after indexing
    attributes: list[AttrField] = field(default_factory=lambda: [f.clone() for f in DEFAULT_ATTRIBUTES])

    def enabled_attributes(self) -> list[AttrField]:
        return [a for a in self.attributes if a.enabled]

    def exposed_attributes(self) -> list[AttrField]:
        """Fields that get a calibre custom column: extracted AND mirrored."""
        return [a for a in self.attributes if a.enabled and a.exposed]


def _settings_from_dict(data: dict[str, Any]) -> Settings:
    ans = Settings()
    if not isinstance(data, dict):
        return ans
    embed = data.get('embed')
    if isinstance(embed, dict):
        ans.embed = EmbedSettings(**{k: v for k, v in embed.items() if k in {f.name for f in fields(EmbedSettings)}})
    for key in ('vector_backend', 'format_priority', 'target_chars', 'overlap_chars', 'embed_context_tokens', 'max_chunks_per_book', 'search_min_score', 'attr_mode', 'attr_context_tokens', 'auto_extract_attributes'):
        if key in data:
            setattr(ans, key, data[key])
    if ans.vector_backend == 'auto':
        # legacy global value; per-library backend now lives in the store meta, so the
        # global is only a default for new libraries and must be an explicit choice
        ans.vector_backend = 'sqlite'
    attrs = data.get('attributes')
    if isinstance(attrs, list):
        merged = []
        for item in attrs:
            if not isinstance(item, dict) or not item.get('name'):
                continue
            base = next((a.clone() for a in ans.attributes if a.name == item['name']), None)
            if base is None:
                base = AttrField(name=item['name'], label=item.get('label', 'ss_' + item['name']), type='text', description='')
            for f in fields(AttrField):
                if f.name in item and item[f.name] is not None:
                    setattr(base, f.name, item[f.name])
            merged.append(base)
        if merged:
            # default fields added after this blob was saved (e.g. 'blurb') are
            # appended so existing users gain them; saved values always win above
            have = {a.name for a in merged}
            ans.attributes = merged + [a.clone() for a in ans.attributes if a.name not in have]
    return ans


def load_settings(prefs) -> Settings:
    """Load settings. prefs is a mapping (e.g. gprefs) or a get_pref(key, default) callable."""
    if callable(prefs):
        raw = prefs(PREF_KEY, '') or ''
    else:
        raw = prefs.get(PREF_KEY, '') or ''
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    return _settings_from_dict(data)


def save_settings(prefs, settings: Settings) -> None:
    blob = json.dumps(asdict(settings))
    if callable(prefs):
        prefs(PREF_KEY, blob)
    else:
        prefs[PREF_KEY] = blob


# -- optional dependencies -------------------------------------------------------
# numpy / zstandard / lancedb are NOT installed into calibre's own Python. They are
# downloaded (via a system Python) into a private, version-keyed library folder that
# the plugin appends to sys.path at load time and imports directly. This leaves
# calibre's bundled interpreter untouched and needs no restart or admin rights.

DEP_PINS = {
    'numpy': '2.5.3',
    'zstandard': '0.25.0',
    'lancedb': '0.38.0',
}


def _py_tag() -> str:
    """Interpreter tag for the folder name and pip cross-download (e.g. '314')."""
    return f'{sys.version_info.major}{sys.version_info.minor}'


def _calibre_data_dir() -> str:
    """Calibre's per-user config dir (writable, survives updates). Never raises."""
    try:
        from calibre.utils import config_dir

        return os.path.abspath(config_dir())
    except Exception:
        pass
    home = os.path.expanduser('~')
    if os.name == 'nt':
        appdata = os.environ.get('APPDATA')
        if appdata:
            return os.path.join(appdata, 'calibre')
    elif sys.platform == 'darwin':
        return os.path.join(home, 'Library', 'Preferences', 'calibre')
    else:
        xdg = os.environ.get('XDG_CONFIG_HOME')
        if xdg:
            return os.path.join(xdg, 'calibre')
        return os.path.join(home, '.config', 'calibre')
    return os.path.join(home, 'calibre')


def external_deps_root(_base=None) -> str:
    """Flat folder holding every downloaded package for this calibre Python version."""
    base = _calibre_data_dir() if _base is None else _base
    return os.path.join(base, f'semantic-search-libs-py{_py_tag()}')


def external_deps_disabled() -> bool:
    """True where the plugin offers only the basic setup (no external dependency
    installs/uninstalls) -- currently macOS."""
    return sys.platform == 'darwin'


def bootstrap_external_deps() -> None:
    """Append the external library folder to sys.path (end) so calibre can import it.

    Called once at plugin load, before store/gui are imported. No-op when the folder
    does not exist yet or where external deps are disabled. Appending at the end keeps
    calibre's own bundled packages (urllib3, packaging, ...) ahead of any transitive
    copies we download."""
    if external_deps_disabled():
        return
    root = external_deps_root()
    if os.path.isdir(root) and root not in sys.path:
        sys.path.append(root)


def _find_dist_dir(root: str, dep: str):
    """The <dep>-<ver>.dist-info directory inside `root`, or None."""
    if not os.path.isdir(root):
        return None
    prefix = dep + '-'
    found = None
    for entry in os.listdir(root):
        if entry.startswith(prefix) and entry.endswith('.dist-info') and os.path.isdir(os.path.join(root, entry)):
            found = os.path.join(root, entry)
    return found


def dep_in_root(dep: str, _base=None) -> bool:
    """True when `dep`'s files are present in the external library folder."""
    return _find_dist_dir(external_deps_root(_base), dep) is not None


def _maybe_wipe_root(root: str) -> None:
    """Delete the whole folder once no dependency remains (no *.dist-info left)."""
    if (os.path.isdir(root) and not any(e.endswith('.dist-info') for e in os.listdir(root)) and not _read_pending(root)):
        shutil.rmtree(root, ignore_errors=True)


def _prune_empty_dirs(root: str) -> None:
    """Remove directories left empty after a dist's files are deleted (never `root`)."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


def _dep_status(dep: str) -> tuple[bool, str]:
    """(importable_now, version_or_reason). Reflects what the plugin can use right now."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(dep)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
    if spec is None:
        return False, f"the '{dep}' package is not installed"
    try:
        from importlib.metadata import version

        ver = version(dep)
    except Exception:
        ver = 'unknown'
    return True, ver


def numpy_status() -> tuple[bool, str]:
    return _dep_status('numpy')


def zstandard_status() -> tuple[bool, str]:
    return _dep_status('zstandard')


def lancedb_status() -> tuple[bool, str]:
    return _dep_status('lancedb')


def _target_tags():
    """(implementation, python_version, platform) wheel tags for THIS interpreter."""
    impl = 'cp'
    pyver = _py_tag()
    if os.name == 'nt':
        plat = 'win_amd64'
    elif sys.platform == 'darwin':
        plat = 'macosx_11_0_arm64' if platform.machine().lower() in ('arm64', 'aarch64') else 'macosx_10_9_x86_64'
    else:
        plat = 'manylinux2014_x86_64'
    return impl, pyver, plat


def _popen_hidden():
    """Extra Popen kwargs so child processes don't flash a console window (Windows GUI app)."""
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {}


def _run_captured(cmd, _popen=None, timeout=60):
    """Run cmd and collect (returncode, [non-empty lines]). Inject _popen for tests."""
    popen = _popen or subprocess.Popen
    try:
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', **_popen_hidden())
    except Exception as e:
        return 127, [f'{type(e).__name__}: {e}']
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate()
        except Exception:
            pass
        return 124, [f'timed out after {timeout}s']
    lines = [ln.rstrip() for ln in (out or '').splitlines() if ln.strip()]
    return proc.returncode, lines


def _find_system_python(_popen=None):
    """A command prefix (list[str]) for a system Python that has pip, or None.

    Used only to fetch packages; the plugin never runs code through it."""
    candidates = (['py', '-3'], ['python'], ['python3']) if os.name == 'nt' else (['python3'], ['python'])
    for prefix in candidates:
        rc, lines = _run_captured(prefix + ['-m', 'pip', '--version'], _popen, timeout=30)
        if rc == 0 and lines:
            return prefix
    return None


def _pip_download_command(dep: str, pin: str, constraints_path: str, wheels_dir: str, pyexe) -> list[str]:
    impl, pyver, plat = _target_tags()
    return [
        *pyexe, '-m', 'pip', 'download', f'{dep}=={pin}',
        '--only-binary=:all:',
        '--implementation', impl,
        '--python-version', pyver,
        '--platform', plat,
        '-c', constraints_path,
        '-d', wheels_dir,
    ]


def _pip_download(dep, pin, constraints_path, wheels_dir, pyexe, _popen=None):
    """Run `pip download` for one top-level dep (+ its closure) into wheels_dir.

    Raw pip output is kept only for the error tail; it is never streamed to the UI."""
    cmd = _pip_download_command(dep, pin, constraints_path, wheels_dir, pyexe)
    popen = _popen or subprocess.Popen
    try:
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', **_popen_hidden())
    except Exception as e:
        return False, f'{type(e).__name__}: {e}\ncommand: {" ".join(cmd)}'
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            if len(tail) > 12:
                tail.pop(0)
    rc = proc.wait()
    if rc == 0:
        return True, ''
    err = '\n'.join(tail) or f'pip exited with code {rc} and produced no output'
    return False, f'{err}\ncommand: {" ".join(cmd)}'


def _wheel_dist_name(whl: str):
    """The distribution name a wheel installs (from its .dist-info directory), or None."""
    try:
        with zipfile.ZipFile(whl) as zf:
            for name in zf.namelist():
                top = name.split('/', 1)[0]
                if top.endswith('.dist-info'):
                    return top.rsplit('-', 1)[0]
    except Exception:
        return None
    return None


def _extract_wheels(wheels_dir: str, root: str) -> int:
    """Unpack each .whl in wheels_dir flat into root, skipping distributions already
    present there (re-extracting a loaded package would fail on Windows). Returns the
    number of packages actually extracted."""
    os.makedirs(root, exist_ok=True)
    extracted = 0
    for whl in sorted(glob.glob(os.path.join(wheels_dir, '*.whl'))):
        dist = _wheel_dist_name(whl)
        if dist is not None and _find_dist_dir(root, dist) is not None:
            continue  # already installed (and possibly mapped into this process)
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(root)
        extracted += 1
    return extracted


def install_dep(dep: str, progress=None, _popen=None, _base=None) -> tuple[bool, str]:
    """Download `dep` (pinned) + its closure into the external library folder."""
    if dep not in DEP_PINS:
        return False, f"unknown dependency '{dep}'"
    if external_deps_disabled():
        return False, 'External dependencies are not available on macOS.'
    pin = DEP_PINS[dep]
    pyexe = _find_system_python(_popen)
    if pyexe is None:
        return False, (
            'No usable system Python with pip was found. Install a 64-bit CPython '
            '(e.g. from python.org) so that "py" (Windows) or "python3"/"python" is on PATH.'
        )
    root = external_deps_root(_base)
    wheels_dir = tempfile.mkdtemp(prefix='ss-deps-')
    try:
        constraints_path = os.path.join(wheels_dir, 'constraints.txt')
        with open(constraints_path, 'w', encoding='utf-8') as fh:
            for name, p in DEP_PINS.items():
                fh.write(f'{name}=={p}\n')
        if progress is not None:
            progress(f'Downloading {dep} and its dependencies...')
        ok, err = _pip_download(dep, pin, constraints_path, wheels_dir, pyexe, _popen)
        if not ok:
            return False, f'Failed to download {dep}:\n{err}'
        if progress is not None:
            progress('Extracting packages into the library folder...')
        if not glob.glob(os.path.join(wheels_dir, '*.whl')):
            return False, f'pip reported success but produced no wheels in {wheels_dir}'
        n = _extract_wheels(wheels_dir, root)
        if n == 0:
            return True, f'{dep} is already installed in the external library folder; nothing to do.'
        return True, f'{dep} installed into the external library folder ({n} package(s) added).'
    finally:
        shutil.rmtree(wheels_dir, ignore_errors=True)


def _pending_path(root: str) -> str:
    return os.path.join(root, '.pending_removal.json')


def _read_pending(root: str) -> list[str]:
    p = _pending_path(root)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return []
    return [x for x in data if isinstance(x, str)]


def _write_pending(root: str, items: list[str]) -> None:
    try:
        with open(_pending_path(root), 'w', encoding='utf-8') as fh:
            json.dump(sorted(set(items)), fh)
    except OSError:
        pass


def process_pending_removals(_base=None) -> None:
    """Delete files an uninstall had to defer because they were locked in-session.

    Runs at plugin load, before any external package is imported (so nothing is
    mapped into the process yet and every deferred path can be removed)."""
    root = external_deps_root(_base)
    pending = _read_pending(root)
    if not pending:
        return
    remaining = []
    for path in pending:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.lexists(path):
                os.remove(path)
        except OSError:
            remaining.append(path)  # still locked; retried on the next start
    _write_pending(root, remaining)
    if not remaining:
        _prune_empty_dirs(root)
        _maybe_wipe_root(root)


def uninstall_dep(dep: str, progress=None, _base=None) -> tuple[bool, str]:
    """Remove `dep`'s own files (per its RECORD) from the external library folder.

    On Windows every package ships native libraries (.pyd/.dll) that are mapped into
    this process once loaded and cannot be deleted in-session, so the whole package is
    deferred to a pending-removal manifest cleaned at the next start; only the (unlocked)
    dist-info is dropped now, which is what flips the UI. On POSIX every file is removed
    immediately, since open/mapped files can be unlinked."""
    if dep not in DEP_PINS:
        return False, f"unknown dependency '{dep}'"
    if external_deps_disabled():
        return False, 'External dependencies are not available on macOS.'
    root = external_deps_root(_base)
    dist_dir = _find_dist_dir(root, dep)
    if dist_dir is None:
        return True, f'{dep} is not installed in the external library folder; nothing to remove.'
    record = os.path.join(dist_dir, 'RECORD')
    entries: list[str] = []
    if os.path.isfile(record):
        with open(record, encoding='utf-8') as fh:
            entries = [ln.strip() for ln in fh]
    dist_base = os.path.basename(dist_dir)

    # Drop the (unlocked) dist-info first so dep_in_root flips and the UI updates; if
    # even that fails there is nothing safe to do.
    try:
        shutil.rmtree(dist_dir)
    except OSError as e:
        return False, f'could not remove {dep} metadata ({dist_base}): {e}'

    if os.name == 'nt':
        # Native libraries are locked in-session: defer the package's files to next start.
        pending = []
        for raw in entries:
            if not raw:
                continue
            rel = urllib.parse.unquote(raw.split(',')[0])
            if rel == dist_base or rel.startswith(dist_base + '/'):
                continue  # dist-info already removed above
            target = os.path.join(root, rel)
            if os.path.lexists(target):
                pending.append(target)
        _write_pending(root, _read_pending(root) + pending)
        if progress is not None:
            progress(f'{dep} will be fully removed when calibre restarts')
        return True, (f'{dep} marked for removal; its native libraries are in use by the running '
                      'session, so its files will be deleted when calibre restarts.')

    # POSIX: open/mapped files can be unlinked, so remove everything now.
    removed = 0
    failed: list[str] = []
    for raw in entries:
        if not raw:
            continue
        rel = urllib.parse.unquote(raw.split(',')[0])
        if rel == dist_base or rel.startswith(dist_base + '/'):
            continue  # dist-info already removed above
        target = os.path.join(root, rel)
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.lexists(target):
                os.remove(target)
                removed += 1
        except OSError as e:
            failed.append(f'{rel}: {e}')
    if progress is not None:
        progress(f'removed {dep} ({removed} files)')
    if failed:
        return False, f'{dep} mostly removed but some files could not be deleted:\n' + '\n'.join(failed[:10])
    _prune_empty_dirs(root)
    _maybe_wipe_root(root)
    return True, f'{dep} removed from the external library folder.'
