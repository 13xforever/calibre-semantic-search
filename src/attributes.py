'''LLM-based attribute extraction into calibre custom columns.

The attribute schema is user-configurable (see utils.AttrField). Extraction runs
on a worker thread; the LLM provider is injected so tests can fake it.
'''

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Annotated, Any, Optional

from .chunker import CHARS_PER_TOKEN, estimate_tokens
from .utils import parse_template_kwargs

DEFAULT_CONTEXT_TOKENS = 8192
OVERHEAD_TOKENS = 1024  # tokens held back for the model's structured output + wrapper; the per-schema prompt size is measured and subtracted separately (see _text_token_budget)
EST_SAFETY_FACTOR = 1.13  # headroom for estimate_tokens undershooting the model's real tokenizer
MIN_TEXT_CHARS = 2000  # floor so a tiny context limit still yields a usable sample


def text_budget_chars(context_tokens: int) -> int:
    """Book-text character budget per LLM call, derived from the model's context limit."""
    return max(MIN_TEXT_CHARS, int((context_tokens - OVERHEAD_TOKENS) * CHARS_PER_TOKEN))


def _text_token_budget(context_tokens: int, fields) -> int:
    """Book-text token budget per LLM call for a schema and model window.

    Holds back a safety-scaled slice of the window, then subtracts the actual
    prompt size for THIS schema (which grows with user-added fields/descriptions)
    and the output/wrapper reserve. estimate_tokens can undershoot the model's
    real tokenizer, so EST_SAFETY_FACTOR keeps realized messages inside the window."""
    prefix_est = estimate_tokens(_prompt_for('', fields))
    return max(1, int(context_tokens / EST_SAFETY_FACTOR) - prefix_est - OVERHEAD_TOKENS)


def column_key(label: str) -> str:
    return '#' + label


def human_name(name: str) -> str:
    return name.replace('_', ' ').title()


def column_title(f) -> str:
    """The calibre column's display title for a field: the configured title, or a
    derived one (underscores to spaces, title-cased) when none is set."""
    t = (f.title or '').strip()
    return t or human_name(f.name)


def _column_spec(f) -> tuple[str, bool]:
    """Calibre column (datatype, is_multiple) for an attribute field."""
    if f.type == 'tags':
        return 'text', True  # the same mechanism calibre uses for #tags
    if f.type == 'text':
        return 'comments', False  # plain multi-line like #description; no tag-browser category
    return 'text', False  # category: single-value normalized text


def _value_for_type(value, field_type):
    """Shape a stored value for the target field type (list for tags, string otherwise)."""
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
    else:
        s = str(value).strip()
        parts = [s] if s else []
    if field_type == 'tags':
        return parts or None
    return ', '.join(parts) or None


def _convert_column(api, f, datatype, is_multiple):
    """Recreate a column with a different calibre type, preserving its values."""
    old = {}
    try:
        for bid in api.all_ids():
            v = api.get_custom(bid, label=f.label, index_is_id=True)
            if not v:
                continue
            cv = _value_for_type(v, f.type)
            if cv:
                old[bid] = cv
    except Exception as e:
        print(f'semantic search: could not read column {f.label} for conversion: {e!r}')
    try:
        api.delete_custom_column(label=f.label)
        api.create_custom_column(f.label, column_title(f), datatype, is_multiple)
        if old:
            api.set_field(column_key(f.label), old)
    except Exception as e:
        print(f'semantic search: could not convert column {f.label} to {datatype}: {e!r}')


def ensure_columns(new_api, settings) -> dict[str, str]:
    """Ensure a custom column of the right type and title exists for every enabled attribute.

    Returns {field.name: '#label'}. 'tags' fields use datatype 'text' with
    is_multiple=True (the same mechanism calibre uses for #tags); 'category'
    fields use single-value 'text'; 'text' fields use datatype 'comments'
    (plain multi-line, like #description — no tag-browser category). A column
    whose stored type no longer matches the field is converted in place with
    its values copied across; a column whose display title differs from the
    configured one is renamed in place (no data loss).
    """
    out = {}
    existing = new_api.backend.custom_column_label_map
    for f in settings.enabled_attributes():
        datatype, is_multiple = _column_spec(f)
        desired = column_title(f)
        meta = existing.get(f.label)
        if meta is None:
            new_api.create_custom_column(f.label, desired, datatype, is_multiple)
            existing = new_api.backend.custom_column_label_map
        elif meta['datatype'] != datatype or bool(meta['is_multiple']) != is_multiple:
            _convert_column(new_api, f, datatype, is_multiple)
            existing = new_api.backend.custom_column_label_map
        # enforce the configured title in place (covers the no-type-change case and
        # any manual rename); a no-op once the stored name already matches
        meta = existing.get(f.label)
        if meta is not None and meta.get('name') != desired:
            new_api.set_custom_column_metadata(meta['num'], name=desired)
            existing = new_api.backend.custom_column_label_map
        out[f.name] = column_key(f.label)
    return out


def sync_attribute_columns(api, store, settings) -> None:
    """Make the library's custom columns match the attribute schema.

    Enabled fields get their column created (or converted to the right type) if
    needed and backfilled from the plugin store (attrs_raw is the source of
    truth, so no LLM calls are made). Columns of disabled fields are deleted —
    their values remain in the store and are restored when the field is
    re-enabled. Best-effort: a per-operation failure is logged, not raised.
    Idempotent — a no-op once the columns already match the settings.
    """
    existing = api.backend.custom_column_label_map
    for f in settings.attributes:
        if f.enabled or f.label not in existing:
            continue
        try:
            api.delete_custom_column(label=f.label)
        except Exception as e:
            print(f'semantic search: could not delete column {f.label}: {e!r}')
        else:
            existing = api.backend.custom_column_label_map
    enabled = settings.enabled_attributes()
    if not enabled:
        return
    colmap = ensure_columns(api, settings)
    stored = store.all_attrs()
    for f in enabled:
        mapping = {}
        for bid, values in stored.items():
            v = values.get(f.name)
            if f.type == 'tags':
                v = [str(x).strip() for x in (v or []) if str(x).strip()]
                if not v:
                    continue
            else:
                v = '' if v is None else str(v).strip()
                if not v:
                    continue
            mapping[bid] = v
        if mapping:
            try:
                api.set_field(colmap[f.name], mapping)
            except Exception as e:
                print(f'semantic search: could not backfill column {f.label}: {e!r}')


def build_schema_class(fields, doc=None):
    """Build a dynamic structured-output schema class for the given fields.

    Calibre's structured-output parser instantiates the class via ``cls(**parsed_json)``
    (see calibre.ai.structured.instantiate) and introspects the annotations/defaults, so
    it must be a dataclass: that gives us both the keyword-accepting __init__ and the
    field metadata calibre reads for types and defaults.
    """
    import dataclasses

    if doc is None:
        doc = 'Extract the following attributes about a book. Use only information actually present in the text; leave fields null when not determinable.'
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

        ns['doc'] = Doc(doc)
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
        total = sum(toks)
        # ceil(total / budget): Python's // floors, so -(-a // b) is the integer ceil idiom.
        n_groups = max(1, -(-total // max(1, group_tokens)))
        target = total / n_groups  # even share; closing at it keeps groups balanced
        groups, cur, used = [], [], 0
        for c, t in zip(chunks, toks):
            # close early to keep groups even-sized, and never let a group exceed the budget
            if cur and (used >= target or used + t > group_tokens):
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


_TAG_WS_RE = re.compile(r'\s+')
_TAG_TRAIL_PAREN_RE = re.compile(r'\s*\([^()]*\)\s*$')
_TAG_SEP_RE = re.compile(r'\s*(?:[/&+]|\band\b)\s*')


def _tag_key(tag: str) -> str:
    """Dedup key for a tag: case- and whitespace-insensitive, separator variants
    ('/' vs '&' vs '+' vs 'and') collapsed, trailing parentheticals ignored."""
    s = _TAG_WS_RE.sub(' ', tag.strip().lower())
    prev = None
    while prev != s:
        prev = s
        s = _TAG_TRAIL_PAREN_RE.sub('', s)
    s = re.sub(r'\|+', '|', _TAG_SEP_RE.sub('|', s)).strip('|')
    return s or tag.strip().lower()


def normalize_tags(tags) -> list[str]:
    """Deduplicate near-identical tags, keeping first-seen order.

    Tags that differ only in case, whitespace, separator style, or a trailing
    parenthetical are treated as the same tag; the shortest spelling wins.
    """
    best: dict[str, str] = {}
    order: list[str] = []
    for t in tags:
        s = str(t).strip()
        if not s:
            continue
        k = _tag_key(s)
        if k not in best:
            best[k] = s
            order.append(k)
        elif len(s) < len(best[k]):
            best[k] = s
    return [best[k] for k in order]


def _language_note(f) -> str:
    """Per-field prompt note forcing the value language ('' = no instruction)."""
    if not f.language:
        return ''
    if f.language == 'book':
        return ' Write all values in the original language of the book text.'
    return f' Write all values in {f.language}.'


def _prompt_for(text: str, fields) -> str:
    lines = []
    for f in fields:
        if f.type == 'tags':
            lines.append(f"- {f.name} (a comma-separated list of tags): {f.description}{_language_note(f)}")
        else:
            # no length hint: the description carries the format (fields may be prose)
            lines.append(f"- {f.name}: {f.description}{_language_note(f)}")
    return (
        'Read the book text below and extract its attributes.\n\n'
        + '\n'.join(lines)
        + '\n\nReturn only the structured data. If an attribute cannot be determined from the text, leave it null.'
        + '\n\n--- BOOK TEXT ---\n'
        + text
    )


def _distinct_values(values):
    """Dedupe case-insensitively, keeping first-seen spelling and order."""
    seen = set()
    out = []
    for v in values:
        k = v.casefold()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _reduce_prompt(pending, max_tok):
    """Render the reduce prompt; over the token budget, drop middle portions
    (first and last of each field kept) until it fits."""
    keep = {f.name: list(vals) for f, vals in pending}

    def render():
        lines = [
            'The partial values below were extracted in order from consecutive portions of the same book. Merge them into one final value for each field. '
            'Where portions disagree about a fact, prefer the earliest value unless a later one clearly corrects it. '
            'For prose fields, write one cohesive final version; do not concatenate or quote the partials.'
        ]
        for f, _ in pending:
            # the note keeps merged prose in the field's target language (partials already are)
            lines.append(f"- {f.name}: {f.description}{_language_note(f)}")
            for i, v in enumerate(keep[f.name], 1):
                lines.append(f'  portion {i}: {v}')
        return '\n'.join(lines) + '\n\nReturn only the structured data.'

    while estimate_tokens(render()) > max_tok:
        targets = [n for n in keep if len(keep[n]) > 2]
        if not targets:
            break
        n = max(targets, key=lambda n: len(keep[n]))
        keep[n].pop(len(keep[n]) // 2)
    return render()


def _merge_fulltext_partials(fields, partials, llm, max_tok, stage_cb=None):
    """Merge per-group partials into final values.

    Tags union without an LLM call. Text fields with a single distinct value use it
    directly; when groups disagree, one reduce call over the ordered partials
    produces the final value (a null reduce result falls back to the first partial).
    `stage_cb('merging')` is emitted right before that reduce call so callers can
    surface the extra API wait as its own progress stage.
    """
    by_field: dict[str, list] = {}
    for p in partials:
        if p is None:
            continue
        for f in fields:
            nv = _normalize_value(getattr(p, f.name, None), f.type)
            if nv:
                by_field.setdefault(f.name, []).append(nv)
    values: dict[str, Any] = {}
    pending = []
    for f in fields:
        vals = by_field.get(f.name) or []
        if f.type == 'tags':
            merged = []
            for sub in vals:
                for v in sub:
                    if v not in merged:
                        merged.append(v)
            values[f.name] = merged
        else:
            distinct = _distinct_values(vals)
            if len(distinct) <= 1:
                values[f.name] = distinct[0] if distinct else ''
            else:
                pending.append((f, distinct))
    if pending:
        if stage_cb:
            stage_cb('merging')
        schema = build_schema_class([f for f, _ in pending], doc='Merge the partial attribute values into one final value per field.')
        res = llm.generate_structured_output(_reduce_prompt(pending, max_tok), schema, 'You are merging partial book attribute extractions into final values. Use only information present in the partials.')
        if res.exception is not None:
            raise RuntimeError(f'attribute extraction failed: {res.error_details or res.exception}')
        for f, distinct in pending:
            r = _normalize_value(getattr(res.data, f.name, None) if res.data is not None else None, 'text')
            values[f.name] = r or distinct[0]
    return values


@contextmanager
def _with_template_kwargs(llm, raw: str):
    """Yield llm; while active, merge the configured template kwargs into every
    outgoing chat request of an OpenAI-protocol provider.

    calibre's backends build the request body internally with no hook for extra
    fields, so the provider's live module is temporarily wrapped at its single
    request funnel (chat_request). The injection is scoped to this thread — other
    features may use the same provider concurrently — and any setup problem
    degrades to a plain no-op: extraction must never break because of this.
    """
    kwargs = parse_template_kwargs(raw)
    if not kwargs or getattr(llm, 'name', '') not in ('OpenAI compatible', 'LMStudio'):
        yield
        return
    mod = orig = patched = None
    try:
        # builtin_live_module (not a fresh import): calibre.live may hand the
        # provider a live-updated copy of the backend, a distinct module object.
        mod = llm.builtin_live_module
        orig = getattr(mod, 'chat_request', None)
        if mod is not None and callable(orig):
            tid = threading.get_ident()

            def patched(data, *args, **kw):
                if threading.get_ident() == tid and isinstance(data, dict):
                    merged = data.get('chat_template_kwargs')
                    merged = dict(merged) if isinstance(merged, dict) else {}
                    merged.update(kwargs)
                    data['chat_template_kwargs'] = merged
                return orig(data, *args, **kw)

            mod.chat_request = patched
    except Exception:
        pass  # degrades to no-op; the request goes out unmodified
    try:
        yield
    finally:
        # restore only when we are still the topmost wrapper
        if patched is not None and mod is not None and getattr(mod, 'chat_request', None) is patched:
            try:
                mod.chat_request = orig
            except Exception:
                pass


def extract_book_attributes(book_id: int, new_api, store, settings, llm=None, progress_cb=None, stage_cb=None):
    """Extract attributes for one book and write them to custom columns.

    Returns the raw values dict. Raises on LLM errors (caller decides whether to retry).
    In fulltext mode `progress_cb(done, total)` is called before each map call so the
    caller can surface per-book sub-progress; sampled mode is a single call and emits none.
    `stage_cb(name)` reports stage transitions: 'merging' is emitted right before the
    extra reduce LLM call (only when groups disagree on a text field).
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
    max_tok = _text_token_budget(settings.attr_context_tokens, fields)

    values: dict[str, Any] = {}
    with _with_template_kwargs(llm, settings.attr_template_kwargs):
        if settings.attr_mode == 'fulltext':
            groups = _split_for_map(chunks, group_tokens=max_tok)
            partials = []
            for i, g in enumerate(groups):
                if progress_cb:
                    progress_cb(i + 1, len(groups))
                res = llm.generate_structured_output(_prompt_for(g, fields), schema, 'You are extracting book attributes from a portion of a book. Only report what is present in this portion.')
                if res.exception is not None:
                    raise RuntimeError(f'attribute extraction failed: {res.error_details or res.exception}')
                partials.append(res.data)
            values = _merge_fulltext_partials(fields, partials, llm, max_tok, stage_cb=stage_cb)
        else:
            text = sample_text(chunks, max_tokens=max_tok)
            res = llm.generate_structured_output(_prompt_for(text, fields), schema, 'You are extracting book attributes. Use only information actually present in the text.')
            if res.exception is not None:
                raise RuntimeError(f'attribute extraction failed: {res.error_details or res.exception}')
            data = res.data
            for f in fields:
                values[f.name] = _normalize_value(getattr(data, f.name, None) if data is not None else None, f.type)

    # The map-reduce union (and even a single LLM call) can yield near-duplicate
    # tags that differ only in spelling; collapse them before persisting.
    for f in fields:
        if f.type == 'tags':
            values[f.name] = normalize_tags(values.get(f.name) or [])

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
            if f.name not in colmap:
                continue  # defensive: enabled fields always have a column
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
    enabled = frozenset(f.name for f in settings.enabled_attributes())
    fields = store.attrs_fields()  # one query for every book's stored field names
    out = []
    for b in store.indexed_books():
        if b['n_chunks'] == 0:
            continue
        if fields.get(b['id'], frozenset()) != enabled:
            out.append(b['id'])
    return out
