'''OpenAI-compatible /v1/embeddings client (urllib only, no external deps).

Note: some local llama.cpp-based servers can answer HTTP 200 with unusable
vectors — full-length embeddings of JSON nulls (non-finite values serialized
as null). Observed with the f16 Qwen3-Embedding-8B GGUF served by unsloth
Studio's bundled llama-server (its --kv-unified / slot-reuse configuration);
a Q4_K_M build of the same model did not exhibit it. The defect is server-side,
so we validate every vector and retry bad batches (BAD_RESPONSE_RETRIES)
instead of trusting the 200; a batch that keeps failing raises an error that
suggests trying another model/quantization or embedding server.
'''

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request


class EmbedError(RuntimeError):
    pass


class _BadResponse(EmbedError):
    """A 200 response whose body is unusable; retryable at the embed() level."""


class EmbedClient:
    def __init__(self, base_url: str, model: str, api_key: str = '', timeout: int = 300, max_retries: int = 4):
        base = (base_url or '').strip().rstrip('/')
        if base.lower().endswith('/v1'):
            # users often paste the OpenAI-SDK style base (which includes /v1);
            # we always append /v1/embeddings ourselves, so drop it to avoid /v1/v1/...
            base = base[:-3].rstrip('/')
        self.base_url = base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _post(self, path: str, payload: dict) -> tuple[dict, bytes]:
        url = self.base_url + path
        body = json.dumps(payload).encode('utf-8')
        last_err = None
        last_detail = ''
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=self._headers(), method='POST')
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if not raw.strip():
                        raise _BadResponse(f'empty response from {url}')
                    try:
                        return json.loads(raw.decode('utf-8')), raw
                    except json.JSONDecodeError as e:
                        raise _BadResponse(f'invalid JSON in response from {url} ({e}); head: {raw[:200]!r}')
            except urllib.error.HTTPError as e:
                detail = ''
                try:
                    detail = e.read().decode('utf-8', 'replace').strip()
                except Exception:
                    pass
                last_err = e
                if detail:
                    last_detail = detail
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                msg = f'embeddings request failed: HTTP {e.code} for {url}'
                if detail:
                    msg += f'; server said: {detail[:300]}'
                raise EmbedError(msg)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        msg = f'embeddings request failed after {self.max_retries} attempts to {url}: {last_err}'
        if last_detail:
            msg += f'; server said: {last_detail[:300]}'
        raise EmbedError(msg)

    # A 200 with a bad body (null/short/malformed vectors) is usually transient
    # server state; retry this many times, without delay, before failing the batch.
    BAD_RESPONSE_RETRIES = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input, same order."""
        if not texts:
            return []
        payload = {'model': self.model, 'input': texts}
        last_err: _BadResponse | None = None
        for _ in range(1 + self.BAD_RESPONSE_RETRIES):
            try:
                data, raw = self._post('/v1/embeddings', payload)
                return self._parse(data, raw, len(texts))
            except _BadResponse as e:
                last_err = e
        raise EmbedError(
            f'embeddings response invalid after {1 + self.BAD_RESPONSE_RETRIES} attempts '
            f'(server keeps returning unusable vectors; try another model/quantization or embedding server): {last_err}')

    def _parse(self, data, raw: bytes, expected: int) -> list[list[float]]:
        items = data.get('data') if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != expected:
            got = len(items) if isinstance(items, list) else type(items).__name__
            raise _BadResponse(f'expected {expected} embeddings, got {got}; response head: {raw[:200]!r}')
        out = []
        for i, it in enumerate(items):
            v = it.get('embedding') if isinstance(it, dict) else None
            problem = self._vector_problem(i, it, v)
            if problem is not None:
                raise _BadResponse(f'malformed embedding vector in response: {problem}')
            out.append([float(x) for x in v])
        return out

    @staticmethod
    def _vector_problem(i: int, it, v):
        """Short description of why `v` is not a usable vector, or None when it is."""
        if not isinstance(it, dict):
            return f'item {i} is not an object: {repr(it)[:120]}'
        if v is None:
            return f'item {i}: embedding is null'
        if not isinstance(v, list):
            return f'item {i}: embedding is {type(v).__name__}, not a list: {repr(v)[:120]}'
        if not v:
            return f'item {i}: embedding is an empty list'
        bad = [(j, x) for j, x in enumerate(v)
               if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)]
        if bad:
            idxs = ', '.join(str(j) for j, _ in bad[:10])
            extra = f' (+{len(bad) - 10} more)' if len(bad) > 10 else ''
            j0, x0 = bad[0]
            return (f'item {i}: {len(bad)} of {len(v)} elements bad at [{idxs}{extra}]; '
                    f'first: element {j0} is {type(x0).__name__}: {x0!r}')
        return None

    def embed_batched(self, texts: list[str], batch_size: int = 64, progress=None, concurrency: int = 1) -> list[list[float]]:
        """Embed with batching; progress(done, total) is optional.

        Batches are sent in order-preserving fashion; with concurrency > 1 they
        run in parallel on a thread pool and results are reassembled in input order.
        """
        if not texts:
            return []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        total = len(texts)
        out: list[list[float]] = [[] for _ in batches]
        state = {'done': 0}
        lock = threading.Lock()

        def work(idx: int, batch: list[str]):
            vecs = self.embed(batch)
            out[idx] = vecs
            with lock:
                state['done'] += len(batch)
                d = state['done']
            if progress is not None:
                progress(min(d, total), total)

        if concurrency > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(batches)))) as ex:
                futs = [ex.submit(work, i, b) for i, b in enumerate(batches)]
                for fu in futs:
                    fu.result()  # re-raise the first batch error, if any
        else:
            for i, b in enumerate(batches):
                work(i, b)
        return [v for chunk in out for v in chunk]
