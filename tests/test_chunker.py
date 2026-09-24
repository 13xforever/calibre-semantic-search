import os as _os
import sys as _sys
import unittest
from unittest import mock

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

chunker = load('chunker')


class TestMaxChunkChars(unittest.TestCase):
    def test_larger_context_gives_larger_cap(self):
        self.assertGreater(chunker.max_chunk_chars(8192), chunker.max_chunk_chars(512))

    def test_derivation(self):
        self.assertEqual(chunker.max_chunk_chars(8192), int((8192 - 64) * chunker.CHARS_PER_TOKEN))

    def test_floor(self):
        self.assertGreaterEqual(chunker.max_chunk_chars(0), chunker.MIN_CHUNK_CHARS)


class TestGroupParagraphs(unittest.TestCase):
    def test_single_chunk(self):
        paras = ['hello world', 'second para']
        chunks = chunker.group_paragraphs(paras, [[], []], 1000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn('hello world', chunks[0].text)
        self.assertIn('second para', chunks[0].text)

    def test_multiple_chunks(self):
        paras = [f'paragraph number {i} ' + 'x' * 50 for i in range(20)]
        paths = [[f'ch{i // 5}'] for i in range(20)]
        chunks = chunker.group_paragraphs(paras, paths, 300, 0)
        self.assertGreater(len(chunks), 1)
        # every paragraph appears in exactly one chunk (no overlap)
        joined = '\n\n'.join(c.text for c in chunks)
        for i in range(20):
            self.assertIn(f'paragraph number {i} ', joined)

    def test_overlap_carry(self):
        paras = ['A' * 400, 'B' * 400, 'C' * 400]
        chunks = chunker.group_paragraphs(paras, [[], [], []], 500, 100)
        self.assertGreaterEqual(len(chunks), 2)
        # the tail of chunk 1 should appear at the start of chunk 2
        self.assertTrue(chunks[1].text.startswith('A' * 100))

    def test_chapter_path_recorded(self):
        paras = ['p1', 'p2']
        paths = [['Intro'], ['Intro', 'Section 1']]
        chunks = chunker.group_paragraphs(paras, paths, 1000, 0)
        self.assertEqual(chunks[0].chapter_path, ['Intro'])

    def test_empty(self):
        self.assertEqual(chunker.group_paragraphs([], [], 1000, 0), [])

    def test_para_ranges_and_offsets(self):
        paras = [f'para{i}' for i in range(10)]
        chunks = chunker.group_paragraphs(paras, [[] for _ in paras], 15, 0)
        self.assertGreater(len(chunks), 1)
        # offsets strictly increasing, ranges cover the book
        offs = [c.char_offset for c in chunks]
        self.assertEqual(offs, sorted(offs))
        self.assertEqual(chunks[0].para_start, 0)
        self.assertEqual(chunks[-1].para_end, 9)


class TestEstimateTokens(unittest.TestCase):
    def test_latin(self):
        self.assertEqual(chunker.estimate_tokens('x' * 350), 100)

    def test_cyrillic_is_denser_than_latin(self):
        self.assertGreater(chunker.estimate_tokens('а' * 350), chunker.estimate_tokens('x' * 350))
        self.assertEqual(chunker.estimate_tokens('а' * 1000), 667)

    def test_cjk_is_denser_than_cyrillic(self):
        self.assertGreater(chunker.estimate_tokens('中' * 100), chunker.estimate_tokens('x' * 100))
        self.assertEqual(chunker.estimate_tokens('中' * 1000), 1200)

    def test_mixed_scripts(self):
        # 350 latin (100 tokens) + 1000 cyrillic (667 tokens)
        self.assertEqual(chunker.estimate_tokens('x' * 350 + 'а' * 1000), 767)


class TestTokenCap(unittest.TestCase):
    def test_cyrillic_chunks_respect_token_cap(self):
        paras = ['б' * 500 for _ in range(12)]  # 500 cyrillic chars = 333 tokens each
        chunks = chunker.group_paragraphs(paras, [[] for _ in paras], 100000, 0, max_tokens=450)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 450 + 2)


class TestOversizedParagraph(unittest.TestCase):
    """A single extracted 'paragraph' longer than the model's context (whole
    chapters between blank lines, HTML blocks without <p>) must be split, not
    sent as one request."""

    def _big_para(self):
        return ('тестовое слово для проверки разбиения текста на части в русской книге ' * 130).strip()

    def _big_cjk_para(self):
        # CJK text has no word boundaries: the whole paragraph is one unbreakable run,
        # so it must go through the hard-cut path, not word packing
        return '这是一段用于测试超长段落切分行为的中文文本，包含标点符号。' * 175

    def _big_mixed_para(self):
        # Cyrillic and Latin words interleaved (names, loanwords) — mixed per-char costs
        return ('тестовое слово levinson english word ' * 300).strip()

    def test_giant_paragraph_is_split_and_capped(self):
        para = self._big_para()
        self.assertGreater(chunker.estimate_tokens(para), 4032)
        chunks = chunker.group_paragraphs([para], [['Ch 1']], 1000, 150, max_tokens=4032)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 4032)

    def test_giant_paragraph_chunks_stay_near_target_chars(self):
        para = self._big_para()
        chunks = chunker.group_paragraphs([para], [['Ch 1']], 1000, 150, max_tokens=4032)
        for c in chunks:
            # target + overlap tail (+ join separators)
            self.assertLessEqual(len(c.text), 1000 + 150 + 2)

    def test_unbreakable_run_is_hard_cut(self):
        chunks = chunker.group_paragraphs(['а' * 9500], [[]], 1000, 150, max_tokens=4032)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 4032)

    def test_chapter_path_preserved_across_pieces(self):
        para = self._big_para()
        chunks = chunker.group_paragraphs([para], [['Ch 1']], 1000, 150, max_tokens=4032)
        self.assertTrue(all(c.chapter_path == ['Ch 1'] for c in chunks))

    def test_no_words_lost_end_to_end(self):
        # chunk boundaries may drop the separating space (pieces are stripped and
        # joined with blank lines), but every word must survive, in order
        para = self._big_para()
        chunks = chunker.group_paragraphs([para], [[]], 1000, 0, max_tokens=4032)
        joined = ' '.join(c.text for c in chunks)
        self.assertEqual(' '.join(joined.split()), ' '.join(para.split()))

    def test_splitter_preserves_text_exactly(self):
        para = self._big_para()
        pieces = chunker._split_to_budget(para, 4032 - 150 * chunker.DENSE_TOKENS_PER_CHAR, 1000)
        self.assertEqual(''.join(pieces), para)

    def test_giant_cjk_paragraph_is_split_and_capped(self):
        # a single CJK paragraph has no word boundaries -> hard-cut path
        para = self._big_cjk_para()
        self.assertGreater(chunker.estimate_tokens(para), 4032)
        chunks = chunker.group_paragraphs([para], [['第一章']], 1000, 150, max_tokens=4032)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 4032)
            self.assertEqual(c.chapter_path, ['第一章'])

    def test_giant_cjk_paragraph_preserved(self):
        # hard cuts must cover the run exactly: no character lost or duplicated
        para = self._big_cjk_para()
        chunks = chunker.group_paragraphs([para], [[]], 1000, 0, max_tokens=4032)
        self.assertEqual(''.join(c.text for c in chunks), para)

    def test_giant_mixed_script_paragraph_is_split_and_capped(self):
        para = self._big_mixed_para()
        self.assertGreater(chunker.estimate_tokens(para), 4032)
        chunks = chunker.group_paragraphs([para], [['Ch 1']], 1000, 150, max_tokens=4032)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 4032)

    def test_cjk_chunks_respect_token_cap(self):
        paras = ['中' * 800 for _ in range(6)]  # 800 CJK chars = 1160 tokens each
        chunks = chunker.group_paragraphs(paras, [[] for _ in paras], 100000, 0, max_tokens=2000)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(chunker.estimate_tokens(c.text), 2000 + 2)

    def test_english_unaffected_when_cap_not_binding(self):
        paras = [f'paragraph number {i} ' + 'x' * 50 for i in range(20)]
        paths = [[f'ch{i // 5}'] for i in range(20)]
        a = chunker.group_paragraphs(paras, paths, 300, 0)
        b = chunker.group_paragraphs(paras, paths, 300, 0, max_tokens=100000)
        self.assertEqual([c.text for c in a], [c.text for c in b])

    def test_plain_text_passthrough(self):
        text = ('中' * 500 + '\n\n' + '中' * 500)
        chunks = chunker.split_plain_text(text, 100000, 0, max_tokens=900)
        self.assertGreater(len(chunks), 1)


class TestPageToParagraphs(unittest.TestCase):
    def test_headings_build_path(self):
        html = '<html><body><h1>Chapter One</h1><p>First para.</p><h2>Scene</h2><p>Second para.</p></body></html>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(len(paras), 2)
        self.assertEqual(paths[0], ['Chapter One'])
        self.assertEqual(paths[1], ['Chapter One', 'Scene'])

    def test_heading_stack_pop(self):
        html = '<body><h1>A</h1><p>p1</p><h2>B</h2><p>p2</p><h3>C</h3><p>p3</p><h2>D</h2><p>p4</p></body>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paths[0], ['A'])
        self.assertEqual(paths[1], ['A', 'B'])
        self.assertEqual(paths[2], ['A', 'B', 'C'])
        self.assertEqual(paths[3], ['A', 'D'])

    def test_skips_script_style(self):
        html = '<body><script>var x=1;</script><style>.a{}</style><p>Visible text</p></body>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['Visible text'])


class TestParsePageEncoding(unittest.TestCase):
    """Pages without a charset declaration must not be decoded as windows-1252
    when the lenient str-parse fails: calibre's libxml2 raises 'internal error'
    on certain non-ASCII characters, and calibre strips <meta charset> from its
    parsed trees, so bare UTF-8 bytes would otherwise be misread (CJK text
    becomes cp1252 mojibake). Well-formed pages are recovered via strict XML;
    the final byte fallback carries an explicit UTF-8 declaration."""

    PAGE = (
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">'
        '<head><title>t</title></head>'
        '<body><p>\u300c\u8a71\u3053\u3046\u300d\u306b\u9ad8\u901f\u3067\u8fd4\u3057\u305f\u3002'
        '<ruby>\u50b7<rt>\u3046</rt></ruby>\u3000\u96fb\u78c1\u5c04\u51fa\u6a5f\u578b \u2019\U0001F600</p></body></html>'
    )

    def _assert_clean(self, root):
        text = ''.join(root.itertext())
        self.assertIn('\u300c\u8a71\u3053\u3046\u300d', text)
        self.assertIn('\u96fb\u78c1\u5c04\u51fa\u6a5f\u578b', text)
        # cp1252 mojibake markers: 'ã' and C1 control characters
        self.assertNotIn('\u00e3', text)
        self.assertFalse(any(0x80 <= ord(c) <= 0x9f for c in text))

    def test_well_formed_cjk_page_without_charset(self):
        root = chunker.parse_page(self.PAGE)
        self._assert_clean(root)

    def test_str_parse_failure_still_yields_clean_text(self):
        # Simulate the 'internal error' calibre's libxml2 raises on str input:
        # only the str call fails; byte calls behave as they really do.
        real_fromstring = chunker.lhtml.fromstring

        def flaky(source):
            if isinstance(source, str):
                raise chunker.etree.XMLSyntaxError('internal error', None, 0, 0)
            return real_fromstring(source)

        with mock.patch.object(chunker.lhtml, 'fromstring', flaky):
            root = chunker.parse_page(self.PAGE)
        self._assert_clean(root)

    def test_bytes_fallback_declares_utf8(self):
        # The prolog is what makes the final fallback decode as UTF-8 in every
        # libxml2 build; without it, undeclared bytes default to windows-1252.
        raw = self.PAGE.encode('utf-8')
        root = chunker.lhtml.fromstring(b'<?xml version="1.0" encoding="utf-8"?>' + raw)
        self._assert_clean(root)


class TestInlineRuns(unittest.TestCase):
    """Books whose text is not wrapped in block elements: old Mobipocket files
    leave it as inline runs (separated by <br/>) inside non-block wrappers like
    <widger>, or directly under <body>. Such runs must still be collected."""

    def test_widger_wrapped_br_flow(self):
        html = (
            '<html><body><p><img src="cover.png"/></p><div class="mbp_pagebreak"/>'
            '<br/><widger>line one<br/>line two<br/><span>mid</span> tail'
            '<br/><br/>final line</widger></body></html>'
        )
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['line one', 'line two', 'mid tail', 'final line'])

    def test_inline_run_directly_under_body(self):
        html = '<body><widger>alpha<br/>beta</widger></body>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['alpha', 'beta'])

    def test_br_tail_flow_with_empty_marker(self):
        # real Mobipocket shape: line text sits in <br> tails, with empty
        # page-break divs sprinkled in; the empty marker must not break collection
        html = (
            '<body><widger>'
            '<br/><br/>PART 1<br/><span>1874</span><br/>'
            '<div class="mbp_pagebreak"/>'
            'line after marker<br/>final line'
            '</widger></body>'
        )
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['PART 1', '1874', 'line after marker', 'final line'])

    def test_no_double_count_with_nested_blocks(self):
        html = '<body><div><span>inner</span></div><p>block</p></body>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['inner', 'block'])

    def test_script_style_not_collected_in_inline_run(self):
        html = '<body><widger>a<script>var x=1;</script>b</widger></body>'
        paras, paths = chunker.page_to_paragraphs(html)
        self.assertEqual(paras, ['ab'])

    def test_widger_book_chunks_end_to_end(self):
        # shape of the real-world case: one huge page, all text in a <widger>
        lines = [f'paragraph {i} ' + 'word ' * 20 for i in range(50)]
        html = '<html><body><p><img src="c.png"/></p><widger>' + '<br/>'.join(lines) + '</widger></body></html>'
        chunks = chunker.chunks_from_pages([html], 1000, 150)
        self.assertGreater(len(chunks), 1)
        joined = ' '.join(c.text for c in chunks)
        for i in (0, 25, 49):
            self.assertIn(f'paragraph {i}', joined)


class TestChunksFromPages(unittest.TestCase):
    def test_two_pages(self):
        pages = [
            '<body><h1>Ch 1</h1><p>alpha</p></body>',
            '<body><h1>Ch 2</h1><p>beta</p></body>',
        ]
        chunks = chunker.chunks_from_pages(pages, 1000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn('alpha', chunks[0].text)
        self.assertIn('beta', chunks[0].text)

    def test_plain_text_split(self):
        text = ('para one\n\n' + 'x' * 300 + '\n\npara three')
        chunks = chunker.split_plain_text(text, 200, 0)
        self.assertGreaterEqual(len(chunks), 2)


class TestHtmlParserFallback(unittest.TestCase):
    """calibre's bundled libxml2 (2.15.x) HTML parser rejects non-BMP characters
    ('internal error') even in recover mode; OEB pages must still be chunked via
    the XML-mode fallback in parse_page."""

    PAGE = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter 7</title></head>'
        '<body><div id="book-inner">'
        '<p><span xmlns="http://www.w3.org/1999/xhtml" class="koboSpan" id="kobo.25.1">On the bridge \U0001F60A</span></p>'
        '<p>Second paragraph.</p>'
        '</div></body></html>'
    )

    def test_emoji_page_chunks(self):
        chunks = chunker.chunks_from_pages([self.PAGE], 1000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn('On the bridge', chunks[0].text)
        self.assertIn('Second paragraph.', chunks[0].text)

    def test_fallback_matches_html_parser(self):
        from lxml import etree as _etree

        real = chunker.lhtml.fromstring
        try:
            chunker.lhtml.fromstring = lambda html, *a, **kw: (_ for _ in ()).throw(
                _etree.XMLSyntaxError('internal error', None, 15, 126)
            )
            fallback = chunker.chunks_from_pages([self.PAGE], 1000, 0)
        finally:
            chunker.lhtml.fromstring = real
        normal = chunker.chunks_from_pages([self.PAGE], 1000, 0)
        self.assertEqual([c.text for c in fallback], [c.text for c in normal])
        self.assertTrue(fallback)

    def test_fallback_keeps_headings_and_paths(self):
        from lxml import etree as _etree

        page = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<h1>Chapter One</h1><p>First \U0001F600 para.</p>'
            '</body></html>'
        )
        real = chunker.lhtml.fromstring
        try:
            chunker.lhtml.fromstring = lambda html, *a, **kw: (_ for _ in ()).throw(
                _etree.XMLSyntaxError('internal error', None, 1, 1)
            )
            paras, paths = chunker.page_to_paragraphs(page)
        finally:
            chunker.lhtml.fromstring = real
        self.assertEqual(paras, ['First \U0001F600 para.'])
        self.assertEqual(paths, [['Chapter One']])


class TestNonBmpContent(unittest.TestCase):
    """Non-BMP content must survive chunking under both libxml2 builds the dev
    gate uses: the venv wheel's (2.11.x, where the HTML parser handles it
    directly) and calibre's bundled one (2.15.x, which needs the bytes fallback
    in parse_page)."""

    # 👩🏻‍👩🏼‍👧🏼‍👧🏻 — ZWJ family with skin-tone modifiers
    FAMILY = '\U0001F469\U0001F3FB\u200D\U0001F469\U0001F3FC\u200D\U0001F467\U0001F3FC\u200D\U0001F467\U0001F3FB'
    # CJK Unified Ideographs extensions A, B, C, D, E
    CJK_EXT = '\u3400\U00020000\U0002A700\U0002B740\U0002B820'

    def test_zwj_family_emoji_page(self):
        page = f'<body><p>Home {self.FAMILY} home</p></body>'
        chunks = chunker.chunks_from_pages([page], 1000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn(self.FAMILY, chunks[0].text)

    def test_cjk_extension_ideographs_page(self):
        page = f'<body><p>{self.CJK_EXT} test</p></body>'
        chunks = chunker.chunks_from_pages([page], 1000, 0)
        self.assertEqual(len(chunks), 1)
        for ch in self.CJK_EXT:
            self.assertIn(ch, chunks[0].text)

    def test_fallback_preserves_zwj_and_cjk_extensions(self):
        from lxml import etree as _etree

        page = f'<body><p>{self.FAMILY} {self.CJK_EXT}</p></body>'
        real = chunker.lhtml.fromstring
        try:
            chunker.lhtml.fromstring = lambda html, *a, **kw: (_ for _ in ()).throw(
                _etree.XMLSyntaxError('internal error', None, 1, 1)
            )
            chunks = chunker.chunks_from_pages([page], 1000, 0)
        finally:
            chunker.lhtml.fromstring = real
        self.assertEqual(len(chunks), 1)
        self.assertIn(self.FAMILY, chunks[0].text)
        for ch in self.CJK_EXT:
            self.assertIn(ch, chunks[0].text)


class TestNamespacedFallback(unittest.TestCase):
    """Pages with declared namespace prefixes (e.g. epub:type) must survive the
    parse_page fallback chain: calibre's libxml2 2.15.x rejects str input that
    contains certain non-ASCII chars (U+2019, emoji), and the re-parse must keep
    the namespace declarations — the old fallback stripped them, turning a valid
    page into an 'undeclared prefix' error."""

    PAGE = (
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>t</title></head>'
        '<body><section epub:type="bodymatter chapter">'
        '<p>I think they needed to red herring that Abel is Zuma, because otherwise it\u2019s really obvious.</p>'
        '</section></body></html>'
    )

    def test_curly_quote_page_chunks(self):
        # real trigger on libxml2 2.15.x (U+2019 in str input); direct parse elsewhere
        chunks = chunker.chunks_from_pages([self.PAGE], 1000, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn('it\u2019s really obvious', chunks[0].text)

    def test_forced_fallback_keeps_namespace_declarations(self):
        from lxml import etree as _etree

        real = chunker.lhtml.fromstring
        try:
            chunker.lhtml.fromstring = lambda html, *a, **kw: (_ for _ in ()).throw(
                _etree.XMLSyntaxError('internal error', None, 1, 1)
            )
            chunks = chunker.chunks_from_pages([self.PAGE], 1000, 0)
        finally:
            chunker.lhtml.fromstring = real
        self.assertEqual(len(chunks), 1)
        self.assertIn('it\u2019s really obvious', chunks[0].text)


if __name__ == '__main__':
    unittest.main()
