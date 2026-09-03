'''OpenAI-compatible /v1/embeddings client (urllib only, no external deps).'''

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class EmbedError(RuntimeError):
    pass


class EmbedClient:
    def __init__(self, base_url: str, model: str, api_key: str = '', timeout: int = 300, max_retries: int = 4):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload).encode('utf-8')
        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=body, headers=self._headers(), method='POST')
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
        raise EmbedError(f'embeddings request failed after {self.max_retries} attempts: {last_err}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per input, same order."""
        if not texts:
            return []
        data = self._post('/v1/embeddings', {'model': self.model, 'input': texts})
        items = data.get('data') or []
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbedError(f'expected {len(texts)} embeddings, got {len(items) if isinstance(items, list) else type(items).__name__}')
        out = []
        for it in items:
            v = it.get('embedding')
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise EmbedError('malformed embedding vector in response')
            out.append([float(x) for x in v])
        return out

    def embed_batched(self, texts: list[str], batch_size: int = 64, progress=None) -> list[list[float]]:
        """Embed with batching; progress(done, total) is optional."""
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out.extend(self.embed(batch))
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return out
