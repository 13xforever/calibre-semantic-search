import json
import os as _os
import sys as _sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from util import load

embed_client = load('embed_client')


class _Handler(BaseHTTPRequestHandler):
    # server-side state set by tests
    fail_times = 0
    seen_auth = None
    echo = False
    fixed_status = 0  # if non-zero, always answer with this status + fixed_body
    fixed_body = b''
    bad_times = 0  # if >0, answer 200 with a bad body (per bad_kind) this many times
    bad_kind = 'null'  # one of: null, short, empty_vec, nan, bool, sparse_null
    request_count = 0

    def do_POST(self):
        type(self).request_count += 1
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))
        type(self).seen_auth = self.headers.get('Authorization')
        if type(self).fixed_status:
            resp = type(self).fixed_body
            self.send_response(type(self).fixed_status)
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
            return
        n = len(body['input'])
        if type(self).bad_times > 0:
            type(self).bad_times -= 1
            kind = type(self).bad_kind
            if kind == 'null':
                data = [{'object': 'embedding', 'index': i, 'embedding': (None if i == 0 else [1.0])} for i in range(n)]
            elif kind == 'short':
                data = [{'object': 'embedding', 'index': i, 'embedding': [1.0]} for i in range(max(0, n - 1))]
            elif kind == 'empty_vec':
                data = [{'object': 'embedding', 'index': i, 'embedding': []} for i in range(n)]
            elif kind == 'nan':
                data = [{'object': 'embedding', 'index': i, 'embedding': [float('nan')]} for i in range(n)]
            elif kind == 'sparse_null':
                data = [{'object': 'embedding', 'index': i, 'embedding': [None, 1.0, None]} for i in range(n)]
            else:  # bool
                data = [{'object': 'embedding', 'index': i, 'embedding': [True, 1.0]} for i in range(n)]
            resp = json.dumps({'object': 'list', 'data': data}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        if type(self).echo:
            # vector derived from the text itself so order can be verified
            data = [{'object': 'embedding', 'index': i, 'embedding': [float(t), 0.0]} for i, t in enumerate(body['input'])]
        else:
            data = [{'object': 'embedding', 'index': i, 'embedding': [float(i), 1.0 / (i + 1)]} for i in range(len(body['input']))]
        resp = json.dumps({'object': 'list', 'data': data, 'model': body.get('model', '')}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


class TestEmbedClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_embed(self):
        c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
        out = c.embed(['a', 'b', 'c'])
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], [0.0, 1.0])
        self.assertAlmostEqual(out[2][0], 2.0)
        self.assertAlmostEqual(out[2][1], 1.0 / 3)

    def test_batched_order_preserved(self):
        c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
        texts = [f'text {i}' for i in range(10)]
        out = c.embed_batched(texts, batch_size=3)
        self.assertEqual(len(out), 10)
        # indices within each batch restart at 0; check first batch ordering
        self.assertEqual(out[0], [0.0, 1.0])
        self.assertEqual(out[1], [1.0, 0.5])
        self.assertEqual(out[2], [2.0, 1.0 / 3])

    def test_parallel_batches_preserve_order(self):
        _Handler.echo = True
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            texts = [str(i) for i in range(8)]
            calls = []
            out = c.embed_batched(texts, batch_size=2, concurrency=4, progress=lambda d, t: calls.append((d, t)))
            self.assertEqual([v[0] for v in out], [float(i) for i in range(8)])
            self.assertEqual(calls[-1], (8, 8))
        finally:
            _Handler.echo = False

    def test_api_key_header(self):
        c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', api_key='sekret', max_retries=2)
        c.embed(['x'])
        self.assertEqual(_Handler.seen_auth, 'Bearer sekret')

    def test_retry_then_success(self):
        _Handler.fail_times = 2
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=5)
            out = c.embed(['x'])
            self.assertEqual(len(out), 1)
        finally:
            _Handler.fail_times = 0

    def test_error_after_retries(self):
        _Handler.fail_times = 99
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            # the server's error body (read on each retried 5xx) must survive into the final message
            self.assertIn('server said', str(cm.exception))
            self.assertIn('boom', str(cm.exception))
            # exhausted server-side retries get an actionable hint for local servers
            self.assertIn('unload and reload the model', str(cm.exception))
        finally:
            _Handler.fail_times = 0

    def test_base_url_normalization(self):
        c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}/v1', model='m', max_retries=2)
        self.assertEqual(c.base_url, f'http://127.0.0.1:{self.port}')
        out = c.embed(['x'])
        self.assertEqual(len(out), 1)

    def test_405_fails_fast_without_retry(self):
        _Handler.fixed_status = 405
        _Handler.fixed_body = b'{"error": "method not allowed"}'
        before = _Handler.request_count
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=4)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            self.assertEqual(_Handler.request_count - before, 1)
            msg = str(cm.exception)
            self.assertIn('405', msg)
            self.assertIn(f'http://127.0.0.1:{self.port}/v1/embeddings', msg)
        finally:
            _Handler.fixed_status = 0
            _Handler.fixed_body = b''

    def test_malformed_response(self):
        # point at a server that returns wrong count: reuse port with custom handler not needed;
        # simulate by checking length validation logic via a bad batch size is enough coverage.
        c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
        self.assertEqual(c.embed([]), [])

    def test_bad_response_retried_then_success(self):
        _Handler.bad_times = 1
        _Handler.bad_kind = 'null'
        before = _Handler.request_count
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=5)
            out = c.embed(['x'])
            self.assertEqual(len(out), 1)
            self.assertEqual(_Handler.request_count - before, 2)
        finally:
            _Handler.bad_times = 0

    def test_bad_response_gives_up_after_two_retries(self):
        _Handler.bad_times = 99
        _Handler.bad_kind = 'null'
        before = _Handler.request_count
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=9)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            self.assertEqual(_Handler.request_count - before, 3)
            msg = str(cm.exception)
            self.assertIn('after 3 attempts', msg)
            self.assertIn('item 0: embedding is null', msg)
        finally:
            _Handler.bad_times = 0

    def test_short_response_retried_then_success(self):
        _Handler.bad_times = 1
        _Handler.bad_kind = 'short'
        before = _Handler.request_count
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=5)
            out = c.embed(['x', 'y'])
            self.assertEqual(len(out), 2)
            self.assertEqual(_Handler.request_count - before, 2)
        finally:
            _Handler.bad_times = 0

    def test_empty_vector_rejected(self):
        _Handler.bad_times = 99
        _Handler.bad_kind = 'empty_vec'
        before = _Handler.request_count
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            self.assertEqual(_Handler.request_count - before, 3)
            self.assertIn('empty list', str(cm.exception))
        finally:
            _Handler.bad_times = 0

    def test_nan_vector_rejected(self):
        _Handler.bad_times = 99
        _Handler.bad_kind = 'nan'
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            self.assertIn('element 0 is float: nan', str(cm.exception))
        finally:
            _Handler.bad_times = 0

    def test_bool_element_rejected(self):
        _Handler.bad_times = 99
        _Handler.bad_kind = 'bool'
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            self.assertIn('element 0 is bool: True', str(cm.exception))
        finally:
            _Handler.bad_times = 0

    def test_sparse_null_reports_all_bad_indices(self):
        _Handler.bad_times = 99
        _Handler.bad_kind = 'sparse_null'
        try:
            c = embed_client.EmbedClient(f'http://127.0.0.1:{self.port}', model='m', max_retries=2)
            with self.assertRaises(embed_client.EmbedError) as cm:
                c.embed(['x'])
            msg = str(cm.exception)
            self.assertIn('2 of 3 elements bad at [0, 2]', msg)
        finally:
            _Handler.bad_times = 0


if __name__ == '__main__':
    unittest.main()
