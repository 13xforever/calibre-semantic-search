import unittest

import os as _os, sys as _sys
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
