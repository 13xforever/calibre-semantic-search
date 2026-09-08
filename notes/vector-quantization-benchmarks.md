# Vector storage: quantization, sizes & search timings (benchmark findings)

Session benchmark of f16/int8 quantization across the **sqlite** and **lancedb** backends, on a
real sample library, to decide (a) which precision to store at and (b) what index to build in the
lancedb finalize stage. All numbers below are **measured** unless explicitly marked *projected*.

- Target embedding dim: **D = 4096** (model `qwen3_embedding_8b`), vectors **unit-normalized**.
- lancedb pinned at **0.38.0** (`DEP_PINS` in `src/utils.py`).
- All byte counts are binary units (1 GiB = 1024^3 B).

## Sample dataset

| property | value |
|---|---|
| source file | `E:\Temp\calibre\semantic-search-sample_v2.db` (read-only) |
| books | 805 total, 798 with chunks |
| chunks | **719,067** |
| dim / model | 4096 / `qwen3_embedding_8b` |
| scale | ≈ 5% of the full set → full set ≈ **~20× ≈ 14.4M chunks** |

## Raw data composition (f32 baseline)

| component | size | notes |
|---|---|---|
| vectors raw f32 | 11,235.4 MiB ≈ 10.97 GiB | 719,067 × 4096 × 4 B |
| text (zstd + dict) | 289.0 MiB | plain ≈ 637 MB, avg ~886 B/chunk |
| chapter_path | 7.72 MiB | |
| **sqlite file total** | **11.65 GiB** | `master.db` |

- f32 vectors are only **~7% zstd-compressible** (ratio 0.925) — float storage is already near-incompressible.

## Quantization fidelity (measured on the sample)

| scheme | relative L2 error (mean) | max cosine Δ | top-10 overlap vs exact |
|---|---|---|---|
| f16 | 0.00021 | ~2e-5 | effectively exact |
| int8 scalar-quant (per-vector scale) | 1.48% | 0.00148 | **99.8%** (with corrected scoring) |

- int8 per-vector scale spread: min 0.0757, p50 0.0993, max 0.1706 (~2.25× range).
- Raw int8 dot/L2 distance is **not** proportional to cosine; the corrected (scale-adjusted) scoring
  recovers 99.8% top-10 overlap. Uncorrected raw-dot overlap was only 51.6%.

## Storage sizes (measured, per backend)

| variant | on-disk size | notes |
|---|---|---|
| sqlite f32 (`master.db`) | **11.65 GiB** | baseline |
| sqlite f16 | **5.87 GiB** | −49.6% |
| sqlite i8 (scale + codes) | **3.13 GiB** | −73% |
| lancedb f32, built (798 fragments) | 11.35 GiB | one `add()` per book |
| lancedb f32, compacted (1 fragment) | 11.33 GiB | after `optimize()` |
| lancedb f16 | **5.87 GiB** | same as sqlite f16 |
| lancedb i8 | **3.13 GiB** | same as sqlite i8 |

Rebuilt in the f16/i8 index round (vector-only schema, compacted to 1 fragment): lance f32 = 10.97 GiB,
lance f16 = **5.49 GiB**, lance i8 (int8 codes + f32 scale) = **2.75 GiB**. Compact layout varies a few %
between builds; use these as steady-state figures.

Key points:

- **lancedb stores the f32 vector column essentially uncompressed**, so at f32 it saves *no* space vs
  sqlite (11.35 vs 11.65 GiB). Quantization savings are identical per backend (f16 −50%, i8 −73%).
- **Compaction without version-cleanup doubles disk** (old fragments retained alongside the merged one):
  f16 → 11.68 GiB, i8 → 6.20 GiB, f32 → 22.68 GiB. With the default 7-day retention, old versions
  persist for 7 days before pruning. Use `optimize(cleanup_older_than=...)` to reclaim immediately.
- **Pruning needs a second `optimize()` call** (verified in 0.38): one call compacts but does *not* prune
  the files its own compaction just orphaned; a follow-up `optimize(cleanup_older_than=timedelta(0))`
  removes them (probe: 10 adds → 2× disk after 1st call, 1× after 2nd). `delete_unverified=True` was not
  needed in a single-process flow but clears failed-transaction leftovers. Index replacement orphans the old
  index files the same way — settle with two `optimize()` calls after any `create_index(replace=True)`.

## Flat (no-index) search latency

Warm cache, median of 3 runs × 10 real queries, plugin defaults (`limit=20`, `min_score=0`).

| backend / precision | latency | notes |
|---|---|---|
| sqlite f32 | ~19.96 s | (first measurement ~18.9 s) |
| sqlite f16 | **~11.66 s** | ~42% faster than f32 |
| sqlite i8 | ~11.77 s | **no speed win over f16** (i4 matmul not BLAS-tuned + per-row overhead) |
| lancedb f32 flat | 3.16 s (one rebuild: 5.78 s — machine-state variance) | native columnar scan |
| lancedb f16 flat | **2.24 s** | native halffloat search works in 0.38 |
| lancedb i8 | 8.01 s | columnar read + corrected CPU scoring; no native ANN path |

Takeaway: on sqlite, **f16 is the better trade** (−50% disk *and* ~42% faster scan, simpler code);
i8 only wins on disk with no speed gain. On lancedb flat, f16 is fastest and i8 is slowest of the three.

## Index experiments (lancedb f32 table)

### Round 1 — API-default partition count (√n ≈ 848), not doc-recommended

| index | build | disk Δ | latency | top-10 recall |
|---|---|---|---|---|
| IVF_RQ, 1-bit, default nprobes | 18 s | +365 MiB | 9–227 ms | **0/10** |
| IVF_RQ, 8-bit, nprobes=848 | — | — | 235 ms | **0/10** |
| IVF_SQ, nprobes=1 | — | — | 33 ms | 4.4/10 |
| IVF_SQ, nprobes=64 | — | — | 60 ms | 10/10 |
| HNSW_SQ (m=20, efc=300) | 320 s | +2969 MiB | 1127 ms | 10/10 |

### Round 2 — doc-recommended partition counts (the correct comparison)

Docs' `num_partitions` guidance: **HNSW-backed = `n // 1,048,576`** (→ **1** for the sample),
**IVF_RQ/PQ/SQ = `n // 4096`** (→ 175 for the sample); start `ef_construction=150`, `ef` ≈ 1.5×k.

| index (doc config) | build | disk Δ | latency | top-10 recall |
|---|---|---|---|---|
| **IVF_HNSW_SQ, cosine, 1 part, efc=150** | ~168–251 s | +2.9 GiB | **~9 ms** | **10/10** (one build 9.9/10 on a borderline item) |
| IVF_HNSW_SQ, dot, 1 part, efc=150 | ~251 s | +2.9 GiB | ~13 ms | 10/10 |
| IVF_SQ, cosine, 175 parts, nprobes auto/32/64 | ~35 s | small | 16–31 ms | 9.7–9.9/10 |
| IVF_RQ, 175 parts, all nprobes × refine_factor | ~20 s | +365 MiB | 38–229 ms | **0/10 — broken** |

### Conclusions

- **IVF_RQ is broken in lancedb 0.38**: 0/10 recall with the doc-recommended partition count too, at
  every nprobes (auto/64/all) and refine_factor (none/3). The query's own chunk never appears in top-60.
  Not a config issue — it's a version bug. **Avoid IVF_RQ on 0.38** (worth an upstream report).
- **IVF_HNSW_SQ with docs' partitioning is the answer**: ~9 ms vs ~5.8 s flat (≈650× faster) with exact
  top-10 recall, and it's the docs' own recommendation for unfiltered search ("best recall/latency
  trade-off"). Our searches use no `where()`, so the HNSW filtered-latency-variance warning doesn't apply.
- **`dot` vs `cosine`**: docs say `dot` is best for normalized vectors, but measured 13 ms vs 9 ms — no win.
  Stick with cosine (also matches existing score math `score = 1 - distance`).

### Round 3 — indexes on quantized storage (f16 / int8 tables)

Flat baselines re-measured this round: lance f16 flat **1.67 s** (10/10), lance i8 flat corrected
**7.58 s** (9.9/10) — consistent with round 1 within machine-state variance.

| table | index (doc config) | build | disk Δ | latency | top-10 recall |
|---|---|---|---|---|---|
| lance_i8 | **not possible** | — | — | flat only: 7.58 s | 9.9/10 |
| lance_f16 | IVF_HNSW_SQ, cosine, 1 part, efc=150 | ~168 s | **+2.90 GiB** | **~55 ms** | 10/10 (rf=1; also 10/10 w/o rf on rebuild) |
| lance_f16 | IVF_SQ, cosine, 175 parts | ~330–360 s | +2.60 GiB | 60–75 ms (nprobes auto/32/64) | 9.8–10/10 |

- **int8 storage cannot be indexed at all**: `create_index` on the `FixedSizeList(4096 x Int8)` codes
  column fails with `Schema Error: A IVF HNSW SQ index cannot be created on the field 'codes' which has
  data type FixedSizeList(4096 x Int8)`. lancedb vector indexes require float columns, so int8 storage in
  lancedb means flat corrected-scan only (7.6 s ceiling). **i8 quantization is only useful on the sqlite
  backend**, where we control the scan ourselves.
- **Halffloat tables carry a ~5–6× search overhead vs float32 tables in 0.38, regardless of index type**
  (interleaved A/B in one process, both indexes, two rounds each): f32 HNSW_SQ **8–9 ms** vs f16 HNSW_SQ
  **55–56 ms**; f32 IVF_SQ 16–31 ms vs f16 IVF_SQ 60–75 ms. Recall is equally good (10/10 with
  `refine_factor(1)`), and build times/disk are identical (SQ codes are int8 either way). Mechanism not
  identified in 0.38 — treat as an empirical penalty of storing the column as halffloat.
- Index disk cost is independent of input precision: HNSW_SQ ≈ **+2.9 GiB**, IVF_SQ ≈ **+2.6–2.75 GiB**
  per 719k×4096 (≈ +58 / +52 GiB projected at full scale).
- `refine_factor(1)` again fixed the borderline mis-ranking (9.9→10.0 on both f32 and f16 HNSW_SQ) at
  ~0–1 ms cost — confirmed as part of the recommended search path.

## lancedb 0.38 API notes (measured / verified)

- New index API: `table.create_index("vector", config=HnswSq(...))` works **without** the separate
  `pylance`/`lance` package. Only `cleanup_old_versions()` requires pylance — use
  `optimize(cleanup_older_than=timedelta(...))` instead (pylance is not in our deps).
- Config classes live in `lancedb.index`: `IvfFlat`, `IvfSq`, `IvfPq`, `IvfRq`, `HnswSq`, …
  (0.38 exposes `HnswSq`; newer docs list `IvfHnswSq`).
- Index types: `IVF_FLAT, IVF_SQ, IVF_PQ, IVF_RQ, IVF_HNSW_SQ, IVF_HNSW_PQ, IVF_HNSW_FLAT`. There is no
  bare `HNSW_SQ` — "HNSW-SQ" means `IVF_HNSW_SQ`.
- Only **one** vector index per column (`replace=True` by default).
- `optimize()` = compaction + version pruning (default **7-day** retention) + **incremental index update**
  (rows added after the index build fold into the existing index — no full rebuild on later finalize runs).
- A bare `optimize()` doubles disk until cleanup actually prunes.
- **`ef` must be ≥ the query's internal `limit`** (we fetch limit×3 = 60 candidates); explicit `ef=30`
  raises "ef must be greater than or equal to k". Leaving `ef` unset (auto-tuned) is fine and recommended.
- `nprobes` is auto-tuned by default; `.nprobes(n)` fixes both min and max.
- `refine_factor(1)` makes `_distance` exact over the returned candidates at ~0 ms cost — worth using since
  `store.py` filters on `min_score` derived from that distance (quantized-index distances are otherwise approximate).
- `approx_mode` does **not** exist in 0.38 (newer feature).
- `bypass_vector_index()` gives exact ground truth / recall measurement.
- `fast_search()` skips unindexed rows; the default search includes a fallback flat scan of unindexed rows.
- Column projection is only on the **query builder** (`.select([...])`); `LanceTable` itself has no
  `.select()` — full-table reads go through `to_arrow()` (all columns).
- `create_index` rejects non-float vector columns outright (`Schema Error: ... data type
  FixedSizeList(4096 x Int8)`) — see Round 3.
- IVF_RQ requires dim divisible by 8 (4096 ✓). `num_bits=1` is classic RaBitQ; multi-bit uses a newer on-disk layout.

## Recall diagnostics

- `bypass_vector_index()` matches numpy exact top-10 at 10/10.
- The occasional "miss" (e.g. one build scoring 9.9/10) is a **borderline item** (true rank 2–10 boundary,
  adjacent chunks with cos ~0.74), i.e. build-to-build variance — not a systematic failure. A given query's
  miss was recovered on a rebuild of the same config.

## Full-set projection (~20×, ~14.4M chunks) — *projected*

- Space scales linearly: f32 ≈ **233 GiB** (both backends), f16 ≈ **117 GiB**, i8 ≈ **62 GiB**.
- IVF_HNSW_SQ index ≈ **+58 GiB**, one-time build ~1 h (then incremental via `optimize()`).
  lancedb total with index: f32 route ≈ **291 GiB**, f16 route ≈ **175 GiB** (index cost is precision-independent).
- Flat scan scales linearly: sqlite f32 ≈ 5.5 min/search (unusable), lancedb f32 flat ≈ 60–120 s (too slow).
- Indexed search expected to stay in the low-ms–tens-of-ms range (per-partition graph search; ~13–14 partitions);
  the halffloat-table overhead (Round 3) would keep the f16 route in the tens-of-ms range.

## Recommendations / decisions

Decision constraint (agreed): **sub-1 s per search is good enough; minimize disk without compromising
search quality too much.** That kills every flat/scan option at scale, so disk = data + index always, and
data precision is the only lever.

- **lancedb (chosen)**: store **f16** + build an **IVF_HNSW_SQ cosine index** in the finalize stage:
  `num_partitions = max(1, n // 1_048_576)`, `ef_construction = 150`. Always index (no row threshold).
  ≈ **175 GiB full-set total** (~117 data + ~58 index), ~55 ms search, 10/10 top-10 recall with
  `refine_factor(1)` — smallest footprint that meets both constraints; f16 storage error is ~2e-5 cosine
  (effectively exact), so no measurable quality loss vs the f32 route.
- **f32 + HNSW_SQ** (≈291 GiB, ~9 ms) only if <10 ms search ever matters — not worth +116 GiB under the
  sub-1s constraint.
- **Caveat**: recall was measured single-partition (sample). Full set ≈ 13–14 partitions — expected to
  hold, untested; `nprobes` is auto-tuned and can be raised if it degrades.
- **sqlite under the same constraints**: no index path exists — search is always a full scan, so every
  precision fails sub-1s at full scale (f32 ~6.5 min, f16/i8 ~4 min projected; sample: 20 / 11.7 / 11.8 s).
  Sub-1s only for small libraries: ≤~35k chunks (f32) or ≤~60k (f16/i8) — at the sample's ~900 chunks/book
  that's ≈ **40 books (f32) / 70 books (f16)**; "noticeably slow" (~2–3 s) around 150–200 books. The switch
  is latency-driven, not disk-driven (sqlite f16 is actually smaller than lancedb+f16+index at small scale).
  Within sqlite, f16 dominates
  (half f32 disk, ~42% faster scan, effectively exact recall); i8 saves ~47% more disk over f16 with no
  speed gain and −0.2% recall. Role: small-library fallback at f16, not a full-scale option.
- **Search path**: fetch `limit*3` candidates, add `.refine_factor(1)`, leave `nprobes`/`ef` unset
  (auto-tuned), and project only the needed columns (`book_id, chunk_no, text, chapter_path`).
- **Avoid IVF_RQ on 0.38** (broken; report upstream).
- **int8 storage is sqlite-only**: lancedb cannot index non-float vector columns, so i8 there means a
  7.6 s flat scan — never worth it vs f16-flat (1.7 s) or any indexed path.
- **sqlite**: if quantizing at all, **f16 is the better trade** (−50% disk, ~42% faster scan, simpler code);
  i8 only wins on disk with no speed gain.
- **Finalize housekeeping**: after migration/index work run `optimize(cleanup_older_than=timedelta(0))`
  **twice** (second call prunes files orphaned by the first — see Storage sizes), then keep a periodic
  single `optimize()` for routine maintenance.
