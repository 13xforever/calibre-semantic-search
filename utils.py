'''Shared helpers: settings model and preferences access.'''

import json
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
    max_chunks_per_book: int = 0  # 0 = unlimited
    attr_mode: str = 'sampled'  # sampled | fulltext (map-reduce)
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
    for key in ('vector_backend', 'format_priority', 'target_chars', 'overlap_chars', 'max_chunks_per_book', 'attr_mode'):
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
