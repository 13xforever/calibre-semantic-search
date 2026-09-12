'''Chunk book text into paragraph groups with chapter context.

No calibre or Qt imports, so it is unit-testable standalone. The only
non-stdlib dependency is lxml, which calibre bundles. requirements-dev.txt
pins the exact lxml version calibre ships, but the PyPI wheel compiles a
different libxml2 into its binary than calibre does — so the dev gate runs
tests/test_chunker.py twice: under the venv's lxml and, when calibre-debug
is available, under calibre's Python (see parse_page for why both matter).
'''

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree
from lxml import html as lhtml

HEADING_RE = re.compile(r'^h([1-6])$', re.I)
BLOCK_TAGS = {'p', 'div', 'blockquote', 'li', 'pre', 'table', 'section', 'article'}
_XMLNS_DECL_RE = re.compile(r'\s+xmlns(?::[A-Za-z_][\w.-]*)?="[^"]*"')


def parse_page(html: str):
    """Parse a spine page into an element tree.

    OEB pages are well-formed XML (they come from ``etree.tostring`` of a parsed
    tree), so when the HTML parser rejects one we re-parse it in XML mode with
    namespace declarations stripped. That fallback is required because calibre's
    bundled libxml2 (2.15.x) raises 'internal error' on non-BMP characters (e.g.
    emoji) when parsing unicode input — in both the HTML and the XML parser, and
    even in recover mode; the same document parses fine as raw UTF-8 bytes, so
    the fallback encodes before parsing.
    """
    try:
        return lhtml.fromstring(html)
    except etree.XMLSyntaxError:
        return etree.fromstring(_XMLNS_DECL_RE.sub('', html).encode('utf-8'))


@dataclass
class Chunk:
    chunk_no: int
    text: str
    chapter_path: list[str] = field(default_factory=list)
    para_start: int = 0  # paragraph index range within the book's flat paragraph list
    para_end: int = 0
    char_offset: int = 0  # offset of chunk start in the book's full plain text

    def chapter_label(self) -> str:
        return ' > '.join(self.chapter_path)


# Chars-per-token by script class. Deliberately conservative: over-estimating
# tokens keeps chunks inside the model's context window for foreign-language
# text, at the cost of a few extra chunks. The non-Latin value is calibrated
# against real BPE tokenizers (llama.cpp embeddings on Russian: ~1.4 chars per
# token) — BPE handles Cyrillic far worse than Latin, not just a little.
CHARS_PER_TOKEN = 3.5            # Latin text (and fallback)
NONLATIN_CHARS_PER_TOKEN = 1.5   # Cyrillic, Greek, Arabic, Hebrew, Devanagari, Thai, ...
DENSE_TOKENS_PER_CHAR = 1.2      # CJK ideographs, kana, hangul: roughly one token per char
CONTEXT_OVERHEAD_TOKENS = 64     # reserved for the model's own wrapper tokens

# Codepoint ranges for scripts that tokenize at ~1 token per character.
_DENSE_RANGES = (
    (0x3000, 0x30FF),   # CJK punctuation + Japanese kana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility
    (0xFF00, 0xFFEF),   # fullwidth forms
    (0x20000, 0x2EBEF),  # CJK extension B+
)
# Other non-Latin space-separated scripts with fewer chars per token than Latin.
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


def _script_counts(text: str) -> tuple[int, int]:
    """Count (dense, non-latin) chars in `text` per the ranges above."""
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
    return dense, nonlatin


def estimate_tokens(text: str) -> int:
    """Script-aware token estimate for `text` (conservative; see constants above)."""
    dense, nonlatin = _script_counts(text)
    latin = len(text) - dense - nonlatin
    total = dense * DENSE_TOKENS_PER_CHAR + nonlatin / NONLATIN_CHARS_PER_TOKEN + latin / CHARS_PER_TOKEN
    return max(1, int(total + 0.5))


def _split_to_budget(text: str, max_tokens: float, max_chars: int) -> list[str]:
    """Split `text` into pieces within both a token and a char budget.

    Greedy word packing on whitespace (word boundaries kept); an unbroken run
    longer than either whole budget is hard-cut at a length that fits even the
    densest script. Text is preserved exactly (''.join(pieces) == text).
    """
    if len(text) <= max_chars and estimate_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    cur = ''
    cur_tok = 0.0
    hard_cut = max(1, min(int(max_tokens / DENSE_TOKENS_PER_CHAR), max_chars))
    for w in re.findall(r'\S+\s*', text):
        d, nl = _script_counts(w)
        w_tok = d * DENSE_TOKENS_PER_CHAR + nl / NONLATIN_CHARS_PER_TOKEN + (len(w) - d - nl) / CHARS_PER_TOKEN
        if cur and (len(cur) + len(w) > max_chars or cur_tok + w_tok > max_tokens):
            pieces.append(cur)
            cur, cur_tok = '', 0.0
        if not cur and (len(w) > max_chars or w_tok > max_tokens):
            for j in range(0, len(w), hard_cut):
                pieces.append(w[j:j + hard_cut])
            continue
        cur += w
        cur_tok += w_tok
    if cur.strip():
        pieces.append(cur)
    return pieces or [text]


MIN_CHUNK_CHARS = 200


def max_chunk_chars(context_tokens: int) -> int:
    """Largest chunk size (chars, assuming Latin text) within the model's input limit.

    Non-Latin scripts are additionally protected by the per-chunk token cap in
    group_paragraphs (max_tokens), since their chars-per-token is lower.
    """
    return max(MIN_CHUNK_CHARS, int((context_tokens - CONTEXT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN))


def group_paragraphs(paragraphs: list[str], chapter_paths: list[list[str]], target_chars: int, overlap_chars: int, max_tokens: int | None = None) -> list[Chunk]:
    """Group flat paragraph lists into chunks of ~target_chars with char overlap.

    paragraphs[i] and chapter_paths[i] must align. Returns chunks whose text is
    the joined paragraphs (plus a leading overlap tail from the previous chunk).
    When max_tokens is given, a chunk is also closed once its script-aware token
    estimate (estimate_tokens) exceeds it, so dense scripts such as CJK cannot
    overflow the embedding model's context even when the char target allows more.
    A single paragraph longer than the cap (whole chapters between blank lines,
    or HTML blocks without <p> tags) is first split at word boundaries into
    pieces that fit the cap, so no chunk can exceed it.
    """
    chunks: list[Chunk] = []
    n = len(paragraphs)
    if not n:
        return chunks
    if target_chars <= 0:
        target_chars = 1000
    if overlap_chars < 0:
        overlap_chars = 0
    if overlap_chars >= target_chars // 2:
        overlap_chars = target_chars // 5

    if max_tokens is not None:
        # Piece budgets: the token one reserves room for the overlap tail
        # (worst case: all dense chars) that prefixes the chunk a piece opens;
        # the char one keeps pieces small enough that grouping can still pack
        # them into ~target_chars chunks.
        budget = max(1.0, float(max_tokens) - overlap_chars * DENSE_TOKENS_PER_CHAR)
        expanded: list[str] = []
        expanded_paths: list[list[str]] = []
        for para, path in zip(paragraphs, chapter_paths):
            p = para.strip()
            if not p:
                continue
            for piece in _split_to_budget(p, budget, target_chars):
                expanded.append(piece)
                expanded_paths.append(path)
        paragraphs, chapter_paths = expanded, expanded_paths

    para_toks = [estimate_tokens(p) for p in paragraphs] if max_tokens is not None else None

    cur_parts: list[str] = []
    cur_len = 0
    cur_tok = 0.0
    start_idx: int | None = None
    end_idx: int | None = None
    offset_at_start = 0
    total_offset = 0
    prev_tail = ''
    chunk_no = 0

    for i, para in enumerate(paragraphs):
        p = para.strip()
        if not p:
            continue
        add_len = len(p) + 2
        add_tok = (para_toks[i] + 2 / CHARS_PER_TOKEN) if para_toks is not None else 0.0
        over_tokens = para_toks is not None and cur_tok + add_tok > max_tokens
        if cur_parts and start_idx is not None and (cur_len + add_len > target_chars or over_tokens):
            text = '\n\n'.join(cur_parts).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_no=chunk_no,
                        text=text,
                        chapter_path=list(chapter_paths[start_idx]) if start_idx < len(chapter_paths) else [],
                        para_start=start_idx,
                        para_end=end_idx if end_idx is not None else start_idx,
                        char_offset=offset_at_start,
                    )
                )
                chunk_no += 1
            prev_tail = text[-overlap_chars:] if overlap_chars else ''
            cur_parts = []
            cur_len = 0
            cur_tok = float(estimate_tokens(prev_tail)) if (para_toks is not None and prev_tail) else 0.0
            start_idx = end_idx = i
            offset_at_start = total_offset
            if prev_tail:
                cur_parts.append(prev_tail)
                cur_len += len(prev_tail) + 2
        else:
            if start_idx is None:
                start_idx = end_idx = i
                offset_at_start = total_offset
                if para_toks is not None and prev_tail:
                    cur_tok = float(estimate_tokens(prev_tail))
                if prev_tail:
                    cur_parts.append(prev_tail)
                    cur_len += len(prev_tail) + 2
            else:
                end_idx = i
        cur_parts.append(p)
        cur_len += add_len
        cur_tok += add_tok
        total_offset += add_len

    if cur_parts:
        text = '\n\n'.join(cur_parts).strip()
        if text and start_idx is not None:
            chunks.append(
                Chunk(
                    chunk_no=chunk_no,
                    text=text,
                    chapter_path=list(chapter_paths[start_idx]) if start_idx < len(chapter_paths) else [],
                    para_start=start_idx,
                    para_end=end_idx if end_idx is not None else start_idx,
                    char_offset=offset_at_start,
                )
            )
    return chunks


def split_plain_text(text: str, target_chars: int, overlap_chars: int, max_tokens: int | None = None) -> list[Chunk]:
    """Chunk a flat plain-text string (e.g. PDF extraction) on blank lines."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    paths = [[] for _ in paragraphs]
    return group_paragraphs(paragraphs, paths, target_chars, overlap_chars, max_tokens)


_SKIP_TAGS = {'script', 'style', 'head'}
_HEADING_TAGS = {f'h{i}' for i in range(1, 7)}


def _has_block_desc(el):
    """True if `el`'s subtree contains a text-bearing block-level or heading element.

    Empty blocks (e.g. <div class="mbp_pagebreak"/>) carry no content, so they do
    not count: a subtree of inline runs plus empty markers is still an inline run.
    """
    for d in el.iter():
        if d is not el and isinstance(d.tag, str) and d.tag.lower() in BLOCK_TAGS | _HEADING_TAGS:
            if any(ch.strip() for ch in d.itertext()):
                return True
    return False


def _inline_segments(el):
    """Text of an inline-only subtree as segments split on <br/> elements."""
    segs = []
    cur = []

    def go(node):
        if not isinstance(node.tag, str):
            return
        t = node.tag.lower()
        if t == 'br':
            if ''.join(cur).strip():
                segs.append(''.join(cur))
            cur.clear()
            return
        if t in ('script', 'style'):
            return
        if node.text:
            cur.append(node.text)
        for ch in node:
            go(ch)
            if ch.tail:
                cur.append(ch.tail)

    go(el)
    if ''.join(cur).strip():
        segs.append(''.join(cur))
    return segs


def html_to_units(html: str):
    """Walk an HTML page and yield (kind, payload) units.

    kind is 'heading' (payload: (level, text)) or 'para' (payload: text).

    Text lives in block elements (<p>, <div>, ...) in well-formed OEB pages, but
    some conversions (old Mobipocket files in particular) leave it as inline runs
    — e.g. wrapped in a non-block tag like <widger> or sitting directly under
    <body>, separated by <br/>. Such runs are collected too, split on <br/>.
    """
    root = parse_page(html)
    if root.tag == 'html':
        body = root.find('.//body')
        root = body if body is not None else root
    out = []

    def text_of(el):
        # itertext() works on both html and plain etree elements (text_content()
        # is missing from the latter in some lxml builds)
        return ''.join(el.itertext())

    def walk(el):
        tag = el.tag.lower() if isinstance(el.tag, str) else ''
        if tag in _SKIP_TAGS:
            return
        if tag == 'br':
            # in mixed content, text after a bare <br/> is an inline run of its own
            s = re.sub(r'\s+', ' ', el.tail or '').strip()
            if s:
                out.append(('para', s))
            return
        m = HEADING_RE.match(tag)
        if m:
            text = text_of(el).strip()
            if text:
                out.append(('heading', (int(m.group(1)), text)))
            for child in el:
                walk(child)
            return
        if tag in BLOCK_TAGS:
            text = text_of(el).strip()
            # don't double count nested blocks: only treat as a paragraph unit
            # if the subtree holds no other block-level element of interest
            if text and not _has_block_desc(el):
                out.append(('para', re.sub(r'\s+', ' ', text)))
                return
        elif not _has_block_desc(el):
            segs = _inline_segments(el)
            if el.tail and el.tail.strip():
                segs.append(el.tail)
            for seg in segs:
                s = re.sub(r'\s+', ' ', seg).strip()
                if s:
                    out.append(('para', s))
            return
        for child in el:
            walk(child)

    walk(root)
    return out


def page_to_paragraphs(html: str):
    """Convert one spine page's HTML into (paragraphs, chapter_paths).

    Headings maintain a stack; the path recorded for each paragraph is the
    current heading stack. Returns ([para, ...], [path, ...]).
    """
    paragraphs: list[str] = []
    paths: list[list[str]] = []
    stack: list[tuple[int, str]] = []
    for kind, payload in html_to_units(html):
        if kind == 'heading':
            level, text = payload
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            paragraphs.append(payload)
            paths.append([t for _, t in stack])
    return paragraphs, paths


def chunks_from_pages(pages: list[str], target_chars: int, overlap_chars: int, max_tokens: int | None = None) -> list[Chunk]:
    """Chunk a whole book given its spine pages as HTML strings."""
    all_paras: list[str] = []
    all_paths: list[list[str]] = []
    for page in pages:
        if not page:
            continue
        paras, paths = page_to_paragraphs(page)
        all_paras.extend(paras)
        all_paths.extend(paths)
    return group_paragraphs(all_paras, all_paths, target_chars, overlap_chars, max_tokens)
