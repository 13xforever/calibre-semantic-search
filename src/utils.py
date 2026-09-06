'''Shared helpers: settings model and preferences access.'''

import json
import subprocess
import sys
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
)


@dataclass
class Settings:
    embed: EmbedSettings = field(default_factory=EmbedSettings)
    vector_backend: str = 'auto'  # auto | sqlite | lancedb
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
            ans.attributes = merged
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


# -- lancedb installation --------------------------------------------------------


def lancedb_status() -> tuple[bool, str]:
    """Return (installed, version_or_error_message)."""
    try:
        import lancedb

        return True, getattr(lancedb, '__version__', 'unknown')
    except ImportError as e:
        return False, str(e)


def pip_install_command(package: str = 'lancedb', extra_args=()) -> list[str]:
    return [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *extra_args, package]


def _interpreter_note() -> str:
    """Read-only, in-process diagnostics about this interpreter and pip.

    Appended to install-failure messages so a silent or odd pip failure is
    self-explanatory (e.g. 'pip not importable' points at a bundled Python that
    ships without pip). Uses find_spec/metadata only -- never imports or runs pip.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec('pip')
    except Exception as e:
        return f'interpreter: {sys.executable}\npip check failed: {type(e).__name__}: {e}'
    if spec is None:
        return f'interpreter: {sys.executable}\npip is NOT importable in this Python (not installed)'
    try:
        from importlib.metadata import version as _pkg_version

        pip_ver = _pkg_version('pip')
    except Exception:
        pip_ver = 'unknown version'
    origin = getattr(spec, 'origin', None) or 'unknown location'
    return f'interpreter: {sys.executable}\npip {pip_ver} present at {origin}'


def _pip_install(package: str, progress=None, _popen=None) -> tuple[bool, str]:
    """Shared pip installer. Tries a normal install first, then falls back to
    --user (for permission errors). progress(line) receives pip output lines as
    they arrive. Returns (ok, message)."""
    popen = _popen or subprocess.Popen
    attempts = (pip_install_command(package), pip_install_command(package, ('--user',)))
    last_err = 'no install attempt was made'
    for cmd in attempts:
        # errors='replace': calibre's bundled Python may emit bytes the locale
        # codec can't decode; never let that mask the real pip error
        try:
            proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}\ncommand: {" ".join(cmd)}'
            continue
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if progress is not None and line:
                progress(line)
            if line:
                tail.append(line)
                if len(tail) > 12:
                    tail.pop(0)
        rc = proc.wait()
        if rc == 0:
            return True, f'{package} installed.'
        if tail:
            last_err = '\n'.join(tail)
        else:
            # pip gave no output at all (seen with embedded/calibre Pythons); show the
            # exact command so the error can be reproduced by running it manually
            last_err = f'pip exited with code {rc} and produced no output\ncommand: {" ".join(cmd)}'
    return False, f'{last_err}\n\n{_interpreter_note()}'


def install_lancedb(progress=None, _popen=None) -> tuple[bool, str]:
    """Install lancedb into calibre's own Python via pip (see _pip_install)."""
    return _pip_install('lancedb', progress, _popen)


def zstandard_status() -> tuple[bool, str]:
    """Return (installed, version_or_error_message)."""
    try:
        import zstandard

        return True, getattr(zstandard, '__version__', 'unknown')
    except ImportError as e:
        return False, str(e)


def install_zstandard(progress=None, _popen=None) -> tuple[bool, str]:
    """Install zstandard into calibre's own Python via pip (see _pip_install)."""
    return _pip_install('zstandard', progress, _popen)


def numpy_status() -> tuple[bool, str]:
    """Return (installed, version_or_error_message)."""
    try:
        import numpy

        return True, getattr(numpy, '__version__', 'unknown')
    except ImportError as e:
        return False, str(e)


def install_numpy(progress=None, _popen=None) -> tuple[bool, str]:
    """Install numpy into calibre's own Python via pip (see _pip_install)."""
    return _pip_install('numpy', progress, _popen)
