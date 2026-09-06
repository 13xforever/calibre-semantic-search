import os as _os
import sys as _sys
import unittest

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
        pieces = chunker._split_to_budget(para, 4032 - 150 * 1.2, 1000)
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
        paras = ['中' * 800 for _ in range(6)]  # 800 CJK chars = 960 tokens each
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


if __name__ == '__main__':
    unittest.main()
