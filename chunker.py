'''Chunk book text into paragraph groups with chapter context.

Pure Python: no calibre or Qt imports, so it is unit-testable standalone.
The HTML walking helpers here are written against lxml (bundled with calibre)
but degrade gracefully to a regex-free plain-text path when given str input.
'''

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r'^h([1-6])$', re.I)
BLOCK_TAGS = {'p', 'div', 'blockquote', 'li', 'pre', 'table', 'section', 'article'}


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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


CHARS_PER_TOKEN = 3.5  # conservative chars-per-token for English text
MIN_CHUNK_CHARS = 200


def max_chunk_chars(context_tokens: int) -> int:
    """Largest chunk size (chars) that fits within the embedding model's input limit."""
    return max(MIN_CHUNK_CHARS, int((context_tokens - 64) * CHARS_PER_TOKEN))


def group_paragraphs(paragraphs: list[str], chapter_paths: list[list[str]], target_chars: int, overlap_chars: int) -> list[Chunk]:
    """Group flat paragraph lists into chunks of ~target_chars with char overlap.

    paragraphs[i] and chapter_paths[i] must align. Returns chunks whose text is
    the joined paragraphs (plus a leading overlap tail from the previous chunk).
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

    cur_parts: list[str] = []
    cur_len = 0
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
        if cur_parts and start_idx is not None and cur_len + len(p) + 2 > target_chars:
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
            start_idx = end_idx = i
            offset_at_start = total_offset
            if prev_tail:
                cur_parts.append(prev_tail)
                cur_len += len(prev_tail) + 2
        else:
            if start_idx is None:
                start_idx = end_idx = i
                offset_at_start = total_offset
                if prev_tail:
                    cur_parts.append(prev_tail)
                    cur_len += len(prev_tail) + 2
            else:
                end_idx = i
        cur_parts.append(p)
        cur_len += len(p) + 2
        total_offset += len(p) + 2

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


def split_plain_text(text: str, target_chars: int, overlap_chars: int) -> list[Chunk]:
    """Chunk a flat plain-text string (e.g. PDF extraction) on blank lines."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    paths = [[] for _ in paragraphs]
    return group_paragraphs(paragraphs, paths, target_chars, overlap_chars)


def html_to_units(html: str):
    """Walk an HTML page and yield (kind, payload) units.

    kind is 'heading' (payload: (level, text)) or 'para' (payload: text).
    Uses lxml when available; falls back to a lightweight regex walk otherwise
    (good enough for tests and for simple HTML).
    """
    try:
        from lxml import html as lhtml

        root = lhtml.fromstring(html)
        if root.tag == 'html':
            body = root.find('.//body')
            root = body if body is not None else root
        out = []

        def walk(el):
            tag = el.tag.lower() if isinstance(el.tag, str) else ''
            m = HEADING_RE.match(tag)
            if m:
                text = (el.text_content() or '').strip()
                if text:
                    out.append(('heading', (int(m.group(1)), text)))
                for child in el:
                    walk(child)
                return
            if tag in BLOCK_TAGS:
                text = (el.text_content() or '').strip()
                # don't double count nested blocks: only treat as a paragraph
                # unit if it has no block-level children of interest
                has_block_child = any(
                    isinstance(c.tag, str) and c.tag.lower() in BLOCK_TAGS | {t for t in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')}
                    for c in el
                )
                if text and not has_block_child:
                    out.append(('para', re.sub(r'\s+', ' ', text)))
            for child in el:
                walk(child)

        walk(root)
        return out
    except ImportError:
        return _regex_html_units(html)


def _regex_html_units(html: str):
    """Very small fallback parser for when lxml is unavailable (tests)."""
    out = []
    pos = 0
    tag_re = re.compile(r'<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*>', re.S)
    heading_re = re.compile(r'^h([1-6])$', re.I)
    in_heading = 0
    heading_buf = ''
    para_buf = []
    skip_depth = 0

    def flush_para():
        text = re.sub(r'\s+', ' ', ' '.join(para_buf)).strip()
        if text:
            out.append(('para', text))
        para_buf.clear()

    for m in tag_re.finditer(html):
        closing, tag = m.group(1) == '/', m.group(2).lower()
        between = html[pos : m.start()]
        pos = m.end()
        if tag in ('script', 'style'):
            if not closing:
                skip_depth += 1
            elif skip_depth:
                skip_depth -= 1
            continue
        if skip_depth:
            continue
        hm = heading_re.match(tag)
        if hm and not closing:
            flush_para()
            in_heading = int(hm.group(1))
            heading_buf = ''
        elif hm and closing and in_heading == int(hm.group(1)):
            text = re.sub(r'\s+', ' ', heading_buf).strip()
            if text:
                out.append(('heading', (in_heading, text)))
            in_heading = 0
        elif in_heading:
            heading_buf += between
        else:
            if tag in BLOCK_TAGS and closing:
                flush_para()
            else:
                para_buf.append(between)
    flush_para()
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


def chunks_from_pages(pages: list[str], target_chars: int, overlap_chars: int) -> list[Chunk]:
    """Chunk a whole book given its spine pages as HTML strings."""
    all_paras: list[str] = []
    all_paths: list[list[str]] = []
    for page in pages:
        if not page:
            continue
        paras, paths = page_to_paragraphs(page)
        all_paras.extend(paras)
        all_paths.extend(paths)
    return group_paragraphs(all_paras, all_paths, target_chars, overlap_chars)
