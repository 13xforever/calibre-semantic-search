'''LLM-based attribute extraction into calibre custom columns.

The attribute schema is user-configurable (see utils.AttrField). Extraction runs
on a worker thread; the LLM provider is injected so tests can fake it.
'''

from __future__ import annotations

from typing import Annotated, Any, Optional

DEFAULT_CONTEXT_TOKENS = 8192
OVERHEAD_TOKENS = 1024  # reserved for prompt + field schema + output + safety margin
MIN_TEXT_CHARS = 2000  # floor so a tiny context limit still yields a usable sample

# Script-aware token estimation (kept in sync with chunker.py; this module is
# loaded standalone in tests and cannot import from the plugin package).
CHARS_PER_TOKEN = 3.5            # Latin text (and fallback)
NONLATIN_CHARS_PER_TOKEN = 2.5   # Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, ...
DENSE_TOKENS_PER_CHAR = 1.2      # CJK ideographs, kana, hangul: roughly one token per char

_DENSE_RANGES = (
    (0x3000, 0x30FF),   # CJK punctuation + Japanese kana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility
    (0xFF00, 0xFFEF),   # fullwidth forms
    (0x20000, 0x2EBEF),  # CJK extension B+
)
_NONLATIN_RANGES = (
    (0x0370, 0x03FF),   # Greek
    (0x0400, 0x052F),   # Cyrillic
    (0x0530, 0x058F),   # Armenian
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic supplement
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E7F),   # Thai
    (0x10A0, 0x10FF),   # Georgian
    (0xFB50, 0xFDFF),   # Arabic presentation forms A
    (0xFE70, 0xFEFF),   # Arabic presentation forms B
)


def estimate_tokens(text: str) -> int:
    """Script-aware token estimate for `text` (conservative; see constants above)."""
    dense = nonlatin = 0
    for ch in text:
        cp = ord(ch)
        if cp < 0x370:
            continue
        for lo, hi in _DENSE_RANGES:
            if lo <= cp <= hi:
                dense += 1
                break
        else:
            for lo, hi in _NONLATIN_RANGES:
                if lo <= cp <= hi:
                    nonlatin += 1
                    break
    latin = len(text) - dense - nonlatin
    total = dense * DENSE_TOKENS_PER_CHAR + nonlatin / NONLATIN_CHARS_PER_TOKEN + latin / CHARS_PER_TOKEN
    return max(1, int(total + 0.5))


def text_budget_chars(context_tokens: int) -> int:
    """Book-text character budget per LLM call, derived from the model's context limit."""
    return max(MIN_TEXT_CHARS, int((context_tokens - OVERHEAD_TOKENS) * CHARS_PER_TOKEN))


def column_key(label: str) -> str:
    return '#' + label


def human_name(name: str) -> str:
    return name.replace('_', ' ').title()


def ensure_columns(new_api, settings) -> dict[str, str]:
    """Ensure a custom column exists for every enabled attribute.

    Returns {field.name: '#label'}. Multi-value ('tags') fields use datatype
    'text' with is_multiple=True (the same mechanism calibre uses for #tags).
    """
    out = {}
    existing = new_api.backend.custom_column_label_map
    for f in settings.enabled_attributes():
        if f.label not in existing:
            new_api.create_custom_column(f.label, human_name(f.name), 'text', f.type == 'tags')
            existing = new_api.backend.custom_column_label_map
        out[f.name] = column_key(f.label)
    return out


def build_schema_class(fields):
    """Build a dynamic structured-output schema class for the given fields.

    Calibre's structured-output parser instantiates the class via ``cls(**parsed_json)``
    (see calibre.ai.structured.instantiate) and introspects its annotations/defaults, so
    it must be a dataclass: that gives us both the keyword-accepting __init__ and the
    field metadata calibre reads for types and defaults.
    """
    import dataclasses

    anns: dict[str, Any] = {}
    ns: dict[str, Any] = {'__doc__': 'Attributes extracted from a book.'}
    for f in fields:
        desc = f.description or f.name
        if f.type == 'tags':
            anns[f.name] = Annotated[Optional[list[str]], desc]
        else:
            anns[f.name] = Annotated[Optional[str], desc]
        ns[f.name] = None
    ns['__annotations__'] = anns
    try:
        from calibre.ai.structured import Doc

        ns['doc'] = Doc('Extract the following attributes about a book. Use only information actually present in the text; leave fields null when not determinable.')
    except ImportError:
        pass
    return dataclasses.dataclass(type('BookAttributes', (), ns))


def _chunks_for_book(store, book_id: int):
    return store.book_chunks_text(book_id)


def sample_text(chunks: list[str], max_chars: int | None = None, max_tokens: int | None = None) -> str:
    """Evenly sample chunks across the book, capped by a char or token budget.

    The token budget (max_tokens) is script-aware and preferred for LLM prompts:
    it keeps foreign-language text inside the model's context window.
    """
    if not chunks:
        return ''
    if max_tokens is None:
        if max_chars is None:
            max_chars = text_budget_chars(DEFAULT_CONTEXT_TOKENS)
        total = sum(len(c) for c in chunks)
        if total <= max_chars:
            return '\n\n'.join(chunks)
        # pick evenly spaced chunks until budget is spent
        step = max(1, len(chunks) * total // max_chars)
        picked = [chunks[i] for i in range(0, len(chunks), step)]
        out, used = [], 0
        for c in picked:
            if used + len(c) > max_chars and out:
                break
            out.append(c)
            used += len(c)
        return '\n\n'.join(out)
    toks = [estimate_tokens(c) for c in chunks]
    total_tok = sum(toks)
    if total_tok <= max_tokens:
        return '\n\n'.join(chunks)
    step = max(1, len(chunks) * total_tok // max_tokens)
    out, used = [], 0
    for i in range(0, len(chunks), step):
        if used + toks[i] > max_tokens and out:
            break
        out.append(chunks[i])
        used += toks[i]
    return '\n\n'.join(out)


def _split_for_map(chunks: list[str], group_chars: int | None = None, group_tokens: int | None = None):
    if group_tokens is not None:
        toks = [estimate_tokens(c) for c in chunks]
        groups, cur, used = [], [], 0
        for c, t in zip(chunks, toks):
            if cur and used + t > group_tokens:
                groups.append('\n\n'.join(cur))
                cur, used = [], 0
            cur.append(c)
            used += t
        if cur:
            groups.append('\n\n'.join(cur))
        return groups
    if group_chars is None:
        group_chars = text_budget_chars(DEFAULT_CONTEXT_TOKENS)
    groups, cur, used = [], [], 0
    for c in chunks:
        if cur and used + len(c) > group_chars:
            groups.append('\n\n'.join(cur))
            cur, used = [], 0
        cur.append(c)
        used += len(c)
    if cur:
        groups.append('\n\n'.join(cur))
    return groups


def _normalize_value(value: Any, field_type: str):
    if value is None:
        return [] if field_type == 'tags' else ''
    if isinstance(value, (list, tuple)):
        if field_type == 'tags':
            return [str(x).strip() for x in value if str(x).strip()]
        return ', '.join(str(x) for x in value if str(x).strip())
    s = str(value).strip()
    if not s or s.lower() in {'null', 'none', 'n/a', 'unknown'}:
        return [] if field_type == 'tags' else ''
    if field_type == 'tags':
        parts = [p.strip(' .') for p in s.split(',')]
        return [p for p in parts if p]
    return s


def _prompt_for(text: str, fields) -> str:
    lines = []
    for f in fields:
        kind = 'a comma-separated list of tags' if f.type == 'tags' else 'a short phrase or sentence'
        lines.append(f"- {f.name} ({kind}): {f.description}")
    return (
        'Read the book text below and extract its attributes.\n\n'
        + '\n'.join(lines)
        + '\n\nReturn only the structured data. If an attribute cannot be determined from the text, leave it null.'
        + '\n\n--- BOOK TEXT ---\n'
        + text
    )


def extract_book_attributes(book_id: int, new_api, store, settings, llm=None, progress_cb=None):
    """Extract attributes for one book and write them to custom columns.

    Returns the raw values dict. Raises on LLM errors (caller decides whether to retry).
    """
    fields = settings.enabled_attributes()
    if not fields:
        return {}
    if llm is None:
        from calibre.ai import AICapabilities
        from calibre.ai.prefs import plugin_for_purpose

        llm = plugin_for_purpose(AICapabilities.text_to_text)
        if llm is None:
            raise RuntimeError('no text-to-text AI provider configured (Preferences > Plugins > AI Provider)')
    schema = build_schema_class(fields)
    chunks = _chunks_for_book(store, book_id)
    if not chunks:
        return {}
    max_tok = settings.attr_context_tokens - OVERHEAD_TOKENS

    values: dict[str, Any] = {}
    if settings.attr_mode == 'fulltext':
        groups = _split_for_map(chunks, group_tokens=max_tok)
        partials = []
        for i, g in enumerate(groups):
            if progress_cb:
                progress_cb(f'map {i + 1}/{len(groups)}')
            res = llm.generate_structured_output(_prompt_for(g, fields), schema, 'You are extracting book attributes from a portion of a book. Only report what is present in this portion.')
            if res.exception is not None:
                raise RuntimeError(f'attribute extraction failed: {res.error_details or res.exception}')
            partials.append(res.data)
        # reduce: merge partials (later non-null wins for text; union for tags)
        merged: dict[str, Any] = {}
        for p in partials:
            if p is None:
                continue
            for f in fields:
                v = getattr(p, f.name, None)
                nv = _normalize_value(v, f.type)
                if not nv:
                    continue
                cur = merged.get(f.name)
                if f.type == 'tags':
                    merged[f.name] = list(dict.fromkeys((cur or []) + nv))
                else:
                    merged[f.name] = nv  # keep last non-empty
        values = merged
    else:
        text = sample_text(chunks, max_tokens=max_tok)
        res = llm.generate_structured_output(_prompt_for(text, fields), schema, 'You are extracting book attributes. Use only information actually present in the text.')
        if res.exception is not None:
            raise RuntimeError(f'attribute extraction failed: {res.error_details or res.exception}')
        data = res.data
        for f in fields:
            values[f.name] = _normalize_value(getattr(data, f.name, None) if data is not None else None, f.type)

    # Guarantee every enabled field has a key so the book counts as complete
    # (fulltext map-reduce only keeps non-empty values; fill the rest with defaults).
    for f in fields:
        values.setdefault(f.name, [] if f.type == 'tags' else '')

    # Persist to the plugin store first: this is the source of truth and is
    # thread-safe (our own SQLite), so the book counts as done even if the
    # custom-column mirror below fails.
    store.set_attrs(book_id, values)

    # Best-effort mirror into calibre custom columns so the attributes are also
    # browsable/filterable in calibre's main view. A failure here must not fail
    # the book — attrs_raw already holds the data.
    try:
        colmap = ensure_columns(new_api, settings)
        updates: dict[str, dict[int, Any]] = {}
        for f in fields:
            key = colmap[f.name]
            val = values.get(f.name)
            if f.type == 'tags':
                val = val or []
            else:
                val = val or ''
            updates.setdefault(key, {})[book_id] = val
        for key, mapping in updates.items():
            new_api.set_field(key, mapping)
    except Exception:
        pass
    return values


def pending_attribute_books(store, settings) -> list[int]:
    """Books that are indexed but lack stored attributes (or schema changed)."""
    enabled = [f.name for f in settings.enabled_attributes()]
    out = []
    for b in store.indexed_books():
        if b['n_chunks'] == 0:
            continue
        data = store.get_attrs(b['id'])
        stored = set(data.keys())
        if stored != set(enabled):
            out.append(b['id'])
    return out
