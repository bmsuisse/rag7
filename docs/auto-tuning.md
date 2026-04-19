# Auto-Tuning `RAGConfig` For Your Corpus

`rag7` exposes 21 retrieval knobs — hybrid-search ratio, fast-path thresholds,
HyDE gating, reranker skip rules, stage toggles, and more. Every corpus lands
on a different optimum. This guide walks through the built-in tuner that finds
that optimum automatically with [Optuna](https://optuna.org).

If you only want the TL;DR, the main [README](../README.md#-tune-it-for-your-data)
has a short tuning section. Read on here for the full story: installation,
testset design, CLI and Python API, every searched parameter, how to ship the
result, and how to diagnose a tuning run that goes wrong.

---

## The 30-second pitch

Retrieval defaults are generic. A hybrid-search ratio of `0.5` is a sensible
guess across corpora — but the moment you know your data is a product catalog
of German SKUs with lots of typos and reorderings, `0.5` is almost certainly
wrong. Maybe your corpus likes `0.75`. Maybe HyDE hurts more than it helps
because descriptions are already dense. Maybe the fast-accept path fires too
aggressively and misses niche items.

You can't know which by reading docs. You have to measure. `RAGTuner` does the
measuring for you:

- You hand it a small **testset** of `(query, expected_ids)` pairs.
- It **samples configs** in parallel using Optuna's TPE sampler.
- It runs each config against your real backend and scores hit@k + latency.
- It returns the **best `RAGConfig`** and saves it as TOML.

One command. Typically 30–60 minutes. Usually a 5–15% lift on hit@5 over
library defaults, with side-benefits on paraphrase consistency.

---

## Before/after: what tuning actually buys you

On our internal German product-catalog testset (three Meilisearch indexes,
~250 queries total including adversarial paraphrases), tuning lifted
**paraphrase consistency** from 0.667 → 0.708 (+4.2pp) without giving up any
hit@5:

| Metric          | Baseline | Tuned   | Δ       |
|-----------------|---------:|--------:|--------:|
| hit@5           | 1.000    | 1.000   | ±0      |
| consistency     | 0.667    | 0.708   | +0.041  |
| stable_top1     | 0.523    | 0.541   | +0.018  |
| **combined**    | **0.734**| **0.755**| **+0.021** |

`combined` here is the scoring function `hit@5 * 0.4 + consistency * 0.35 +
stable_top1 * 0.25` used in `tests/eval_v2/run_tuner.py`. The exact weights
don't matter — what matters is that TPE found a config that makes the agent
more robust to paraphrases (synonyms, word reorderings, typos) without any
regression on the primary hit metric.

The headline isn't +4.2pp. The headline is: **you didn't have to guess**.

---

## Installation

`optuna` is an optional extra — the tuner only imports it lazily so the core
`rag7` package stays dependency-light.

```bash
pip install 'rag7[tune]'
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add 'rag7[tune]'
```

This adds `optuna>=4.8.0` to your environment. Everything else (your backend
client, LLM SDK, embedder) is whatever you already had installed for runtime
retrieval.

---

## Building a testset

The tuner's ceiling is your testset's ceiling. Garbage in, garbage out — a
noisy or unrepresentative testset will find a config that looks good on the
noise and mediocre in production. Spend an hour on the testset before you
spend an hour on tuning.

### Format

Two accepted formats — JSON array or JSONL (one object per line):

```json
[
  {"query": "Makita Akku Bohrhammer 18V", "expected_ids": ["SKU-1065144"], "id_field": "sku"},
  {"query": "Bosch Winkelschleifer 125mm", "expected_ids": ["SKU-1057802"], "id_field": "sku"},
  {"query": "kettle 1.7L stainless", "expected_ids": ["KET-009"], "id_field": "sku"},
  {"query": "Festool Staubsauger M-Klasse", "expected_ids": ["SKU-2041"], "id_field": "sku"},
  {"query": "cordless drill brushless 18v dewalt", "expected_ids": ["DW-BR-18"], "id_field": "sku"}
]
```

Each entry is three pieces:

- `query` — what a user would actually type.
- `expected_ids` — one or more canonical IDs that *must* appear in the top-k
  for this query to count as a hit. Multiple IDs means "any of these is fine".
- `id_field` — the metadata key on retrieved documents that holds the ID. For
  catalogs this is usually `sku`, `id`, or `product_id`. It's per-entry so you
  can mix schemas across indexes.

Load it with `rag7.tuner.load_testset`:

```python
from rag7.tuner import load_testset

hit_cases = load_testset("testset.json")
# [("Makita Akku Bohrhammer 18V", ["SKU-1065144"], "sku"), ...]
```

### Testset quality rules of thumb

- **50 queries minimum**, 200+ is better. Too few and TPE overfits.
- **Cover your distribution**. If 30% of real queries are single-token brand
  lookups and 70% are descriptive phrases, mirror that in the testset.
- **Include adversarial variants**. Typos (`Bohrhamer`), reorderings (`18V
  Akku Bohrhammer Makita`), synonyms (`Schlagbohrer` vs `Bohrhammer`),
  language mixes, truncation, extra spaces. These are where bad configs show
  their weaknesses.
- **Long and short queries both**. A config tuned only on short keyword
  queries will over-weight BM25 and hurt long semantic queries.
- **Verify expected_ids are actually in the index**. A query with a wrong
  expected ID is dead weight that can't be satisfied by any config — the
  tuner will give up on it silently.

The internal eval in `tests/eval_v2/` generates adversarial paraphrases
automatically from a base testset — worth stealing the pattern.

---

## CLI usage

`python -m rag7.tuner` runs the tuner end-to-end against a Meilisearch backend
with Azure OpenAI embeddings. It's the fastest path if your stack matches.

```bash
python -m rag7.tuner \
  --index my_index \
  --hits testset.json \
  --trials 50 \
  --pyproject
```

Flag-by-flag:

| Flag           | Required | Default              | What it does                                                                 |
|----------------|----------|----------------------|------------------------------------------------------------------------------|
| `--index`      | yes      | —                    | Meilisearch index name. Reads `MEILI_URL` and `MEILI_MASTER_KEY` from env.   |
| `--hits`       | yes      | —                    | Path to your JSON/JSONL testset.                                             |
| `--trials`     | no       | `50`                 | Number of Optuna trials. More is better but slower; see budget section.      |
| `--seed`       | no       | `42`                 | TPE random seed. Change to explore different regions of the search space.    |
| `--output`     | no       | `rag7.config.toml`   | File to save the tuned config (dedicated TOML mode).                         |
| `--pyproject`  | no       | off                  | Save into `[tool.rag7]` of `pyproject.toml` instead of a dedicated file.     |

The CLI prints the overrides from defaults at the end — handy for spotting
which knobs the tuner actually cared about:

```
Loaded 180 hit cases from testset.json
[I 2026-04-18 10:22:01,332] Trial 0 finished with value: 0.734 ...
...
Saved tuned config to pyproject.toml [tool.rag7]

Overrides from defaults (7):
  semantic_ratio = 0.78
  fusion = dbsf
  fast_accept_score = None
  rerank_skip_dominance = 0.92
  expert_threshold = None
  enable_hyde = False
  enable_filter_intent = False
```

If your stack isn't Meilisearch + Azure OpenAI, skip the CLI and use the
Python API directly — you get full control over backend and embed function.

---

## Python API usage

### Meilisearch backend

```python
from rag7.backend import MeilisearchBackend
from rag7.tuner import RAGTuner, load_testset
from rag7.utils import _make_azure_embed_fn

tuner = RAGTuner(
    backend_factory=lambda: MeilisearchBackend(index="my_index"),
    embed_fn=_make_azure_embed_fn(),
    hit_cases=load_testset("testset.json"),
    eval_k=5,
    latency_weight=0.15,       # 0 = ignore latency, 1 = latency dominates
    latency_budget_ms=2000.0,  # queries slower than this get penalized
)

best = tuner.tune(n_trials=50, seed=42, show_progress=True)

best.save_pyproject()      # writes [tool.rag7] in ./pyproject.toml
# or: best.save_toml("rag7.config.toml")   # dedicated file
```

### LanceDB backend

The tuner takes a `backend_factory` callable so any backend works — just hand
it a zero-arg lambda that builds a fresh backend instance per trial:

```python
from rag7.backend import LanceDBBackend
from rag7.tuner import RAGTuner, load_testset

def make_embed_fn():
    # bring your own embedder — any callable str -> list[float]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    return lambda text: model.encode(text).tolist()

embed_fn = make_embed_fn()

tuner = RAGTuner(
    backend_factory=lambda: LanceDBBackend(
        table="products",
        db_uri="~/.lancedb",
        embed_fn=embed_fn,
    ),
    embed_fn=embed_fn,
    hit_cases=load_testset("testset.json"),
)

best = tuner.tune(n_trials=100)
best.save_toml("rag7.config.toml")
```

Same pattern holds for `QdrantBackend`, `ChromaDBBackend`,
`AzureAISearchBackend`, `PgVectorBackend`, `DuckDBBackend`, `InMemoryBackend`.
The tuner doesn't care which backend it is, only that it can `search()` and
return docs with the `id_field` in metadata.

### Passing custom LLM and reranker

If your production pipeline uses a reranker or non-default LLMs, wire them in
so tuning matches production:

```python
from langchain_openai import AzureChatOpenAI
from rag7.rerank import CohereReranker

tuner = RAGTuner(
    backend_factory=lambda: MeilisearchBackend(index="my_index"),
    embed_fn=_make_azure_embed_fn(),
    hit_cases=load_testset("testset.json"),
    llm=AzureChatOpenAI(deployment_name="gpt-4o-mini"),  # cheap during tuning
    gen_llm=AzureChatOpenAI(deployment_name="gpt-4o-mini"),
    reranker=CohereReranker(model="rerank-multilingual-v3.0"),
)

best = tuner.tune(n_trials=50)
```

The `extra_kwargs` field on `RAGTuner` is forwarded to `AgenticRAG` — useful
for things like `text_fields=["name", "description"]` that shape retrieval but
aren't on `RAGConfig`.

### Using a warm-start baseline

If you already have a hand-tuned config that works decently, pass it as
`baseline_config` — the tuner enqueues it as trial 0 so the best result is
guaranteed to be at least as good as your starting point:

```python
from rag7.config import RAGConfig

warm_start = RAGConfig(
    semantic_ratio=0.75,
    fusion="dbsf",
    enable_hyde=False,
)

best = tuner.tune(n_trials=50, baseline_config=warm_start)
```

Without a `baseline_config`, trial 0 uses library defaults.

---

## What the tuner explores

`RAGTuner.tune()` searches 17 dimensions with Optuna's TPE sampler. Scalar
knobs get a continuous range; booleans get categorical. The interesting
category is **None-able fields** — parameters where `None` means *disable
this stage entirely*, which is a first-class hypothesis during search.

### Scalar parameters

| Parameter                      | Range        | What it does                                                               |
|--------------------------------|-------------:|----------------------------------------------------------------------------|
| `top_k`                        | 5–20         | Final top-k returned from retrieval.                                       |
| `retrieval_factor`             | 2–8          | Over-fetch multiplier before reranking (actual pull = `top_k * factor`).   |
| `rerank_top_n`                 | 3–10         | Docs kept after reranking.                                                 |
| `rerank_cap_multiplier`        | 1.5–4.0      | Hard cap on docs into reranker (`top_k * m`).                              |
| `semantic_ratio`               | 0.3–0.9      | Hybrid-search BM25 ⇄ vector balance. 0.0 = all BM25, 1.0 = all vector.     |
| `short_query_threshold`        | 3–8          | Words below which a query skips LLM preprocessing.                         |
| `bm25_fallback_semantic_ratio` | 0.7–1.0      | Semantic ratio applied when BM25 score is weak (typo/transliteration fallback). |
| `rerank_skip_gap`              | 0.05–0.3     | Score gap between top-1 and top-5 required to skip the reranker.           |

### Categorical parameters

| Parameter                 | Options             | What it does                                                      |
|---------------------------|---------------------|-------------------------------------------------------------------|
| `fusion`                  | `rrf` / `dbsf`      | Score fusion strategy for BM25 + vector results.                  |
| `short_query_sort_tokens` | `true` / `false`    | Sort tokens for short queries (paraphrase invariance).            |

### None-able fields (disable is a first-class option)

Optuna wraps each of these in a categorical `{name}_enabled` gate. If TPE
samples `enabled=False`, the field becomes `None` and the corresponding stage
is bypassed. This is the killer feature — the tuner can discover that your
corpus works better *without* a given stage:

| Parameter                | Range        | `None` means…                                                    |
|--------------------------|-------------:|------------------------------------------------------------------|
| `hyde_min_words`         | 4–12         | Disable HyDE entirely, regardless of query length.               |
| `bm25_fallback_threshold`| 0.2–0.6      | Never boost semantic on weak BM25 — always use configured ratio. |
| `fast_accept_score`      | 0.5–0.95     | Never take the fast path — always run preprocess + HyDE + filter.|
| `fast_accept_confidence` | 0.6–0.95     | Skip the LLM confirmation call on fast path.                     |
| `rerank_skip_dominance`  | 0.6–0.95     | Always rerank, even on obvious single-hit queries.               |
| `expert_threshold`       | 0.05–0.3     | Never escalate to expert reranker.                               |

Concrete example: if your corpus is a clean product catalog where BM25
already dominates, TPE is likely to converge on
`fast_accept_score=None` — i.e., always run the full pipeline because the
fast-path was skipping genuinely-relevant-but-low-scoring items. You'd never
find that by hand-tuning.

### Stage toggles

Same idea, but at pipeline granularity instead of threshold granularity:

| Toggle                    | Default | Disabling it saves…                                        |
|---------------------------|---------|------------------------------------------------------------|
| `enable_hyde`             | `true`  | One LLM call per query (HyDE generation).                  |
| `enable_filter_intent`    | `true`  | One LLM call per query (filter detection).                 |
| `enable_quality_gate`     | `true`  | One LLM call per query (relevance judge) + retry loop.     |
| `enable_preprocess_llm`   | `true`  | One LLM call per query (query rewrite).                    |

Disabling a stage is a pure latency + cost win **if it doesn't hurt quality**.
The tuner measures quality, so it can tell you honestly whether you need it.

### What's *not* tuned

A handful of `RAGConfig` fields are left at defaults during `tune()`:

- Timeouts (`rerank_timeout_s`, `llm_timeout_s`) — these are safety limits,
  not retrieval parameters.
- Agentic loop knobs (`max_iter`, `n_swarm_queries`) — heavy to evaluate and
  orthogonal to retrieval quality.
- `rerank_chars` — doc truncation; tune separately if your docs are unusual.
- `enable_reasoning` — off by default, experimental.

If you need to tune these, fork `RAGTuner.tune()` or run a custom Optuna loop
— see `tests/eval_v2/run_tuner.py` for a worked example.

---

## Config precedence

`RAGConfig.auto()` looks for config in this order, first win:

1. **Runtime kwarg** — `AgenticRAG(config=RAGConfig(...))` passed explicitly.
2. **`[tool.rag7]` in `pyproject.toml`** — standard Python tooling convention
   (same slot as `ruff`, `black`, `mypy`).
3. **`rag7.config.toml`** — dedicated file with a `[rag7]` table.
4. **`RAG_*` env vars** — `RAG_TOP_K`, `RAG_SEMANTIC_RATIO`, etc. Fields that
   accept `None` can be disabled by setting the env var to the literal string
   `"none"` (case-insensitive).
5. **Library defaults** — baked into `RAGConfig` field defaults.

### Which slot should you use?

- **Team-wide tuning, single shared corpus?** → `pyproject.toml`. Commit it,
  everyone gets the same behavior.
- **Per-deployment tuning on corpus-specific data?** → `rag7.config.toml`. Add
  it to `.gitignore`. Each deployment keeps its own.
- **Container/CI overrides for a single knob?** → `RAG_*` env vars. No file
  changes needed.
- **Ephemeral experiments?** → Runtime kwarg. `config=RAGConfig(top_k=15)`.

Mix-and-match works: you can ship a "good for most cases" config in
`pyproject.toml` and let env vars override one or two knobs per deployment.

---

## Writing TOML by hand (and the `disable` list pattern)

TOML has no `null`. Since several `RAGConfig` fields accept `None` to mean
"disable this stage", we use an explicit `disable = [...]` list that the
loader translates back to `None`:

```toml
[tool.rag7]
top_k = 10
retrieval_factor = 4
rerank_top_n = 5
rerank_cap_multiplier = 2.0
semantic_ratio = 0.78
fusion = "dbsf"

hyde_min_words = 8
short_query_threshold = 6
short_query_sort_tokens = true

bm25_fallback_semantic_ratio = 0.9
rerank_skip_gap = 0.1

expert_top_n = 10

enable_hyde = false
enable_filter_intent = false
enable_reasoning = false
enable_quality_gate = true
enable_preprocess_llm = true

rerank_timeout_s = 30.0
llm_timeout_s = 60.0
max_iter = 3
n_swarm_queries = 4
rerank_chars = 2048

# None-valued fields (disabled stages):
disable = ["bm25_fallback_threshold", "fast_accept_score", "expert_threshold"]
```

Both `RAGConfig.save_toml()` and `RAGConfig.save_pyproject()` emit this exact
format. `RAGConfig.from_toml()` and `from_pyproject()` parse it back and set
the listed fields to `None`.

If you're editing by hand, any field that's missing from the TOML gets its
default value, so you only need to include the fields you want to override.

---

## Tuning budget guidance

TPE is a Bayesian method — it learns from prior trials. **50 trials is the
practical minimum**; fewer and you're basically doing random search plus one
or two lucky interpolations. Recommended:

- **Quick sanity check**: 30 trials, ~30 min. Good for "does tuning even help
  on this corpus?"
- **Real tuning**: 50–100 trials, ~1–2 hours.
- **Overnight / CI**: 200–500 trials, ~3–8 hours. Diminishing returns past
  ~150 but sometimes finds unusual corners.

### Rough cost math

Per trial cost is dominated by LLM calls and retrieval latency:

```
trial_time ≈ (n_testset_queries × per_query_latency) / concurrency
per_query_latency ≈ 1.5–4s depending on stages enabled
concurrency = 10 (hardcoded in RAGTuner)
```

For a 180-query testset at ~2s/query with concurrency 10:

```
trial_time ≈ (180 × 2) / 10 ≈ 36s
50 trials ≈ 30 min
```

Add overhead for Optuna bookkeeping and you're looking at ~40–60s per trial
in practice. 50 trials = ~45 minutes.

### Keep LLM cost down

The tuner hits your LLM 1–4 times per query (HyDE, filter-intent, preprocess,
quality gate, depending on which stages are enabled for that trial's config).
Multiply by 180 queries × 50 trials and the bill climbs fast with a premium
model.

**Use a cheap LLM during tuning, swap to production LLM after:**

```python
from langchain_openai import AzureChatOpenAI

# Tuning: cheap
cheap_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")

tuner = RAGTuner(
    backend_factory=...,
    embed_fn=...,
    hit_cases=...,
    llm=cheap_llm,
    gen_llm=cheap_llm,
)
best = tuner.tune(n_trials=50)
best.save_pyproject()

# Production: premium — same config, better model
from rag7 import AgenticRAG, RAGConfig
expensive_llm = AzureChatOpenAI(deployment_name="gpt-4o")

rag = AgenticRAG(
    index="my_index",
    backend=MeilisearchBackend("my_index"),
    embed_fn=embed_fn,
    config=RAGConfig.auto(),  # picks up the tuned [tool.rag7]
    llm=expensive_llm,
    gen_llm=expensive_llm,
)
```

The config transfers cleanly because retrieval thresholds aren't model-specific.
`gpt-4o-mini` and `gpt-4o` reach roughly the same relevance judgments on
typical product queries — the mini model is just noisier at the margins,
which averages out across 180 queries.

### Latency vs quality trade-off

The objective is `score = hit_rate × (1 - latency_weight) + (1 - penalty) × latency_weight`
where `penalty = max(0, mean_latency/latency_budget_ms - 1)`.

- Default `latency_weight=0.15` — mild nudge toward faster configs.
- `latency_weight=0` — pure quality. Use when you don't care about latency
  (batch jobs, offline eval).
- `latency_weight=0.5+` — latency dominates. Good for interactive chat
  where p95 latency matters more than squeezing the last 1% hit@5.

---

## Interpreting results

A tuning run prints progress per trial. In the custom eval loop in
`tests/eval_v2/run_tuner.py`, each trial sets four user-attrs on the Optuna
trial object: `hit@5`, `consistency`, `stable_top1`, `combined`. The
built-in `RAGTuner.tune()` tracks `hit_rate`, `mean_latency_ms`, and `n_hits`.

Reading the output:

- **`combined`** — the objective Optuna is maximizing. Higher is better. Jumps
  early, plateaus late.
- **`hit@5`** — fraction of queries where any expected ID appears in top-5.
  This is the primary retrieval metric. If this doesn't improve, tuning
  didn't help.
- **`consistency`** — for paraphrase groups, the fraction of variants that
  still hit. Measures robustness to rewording.
- **`stable_top1`** — for paraphrase groups, the fraction where the *same*
  top-1 doc is returned across all variants. Measures ranking stability.
- **`mean_latency_ms`** — end-to-end retrieval latency in ms. If this
  balloons, check whether the tuner turned off a skip-stage that was saving
  you time.

After the run finishes, call `.overrides()` on the best config to see which
knobs actually moved:

```python
best = tuner.tune(n_trials=50)
print(best.overrides())
# {'semantic_ratio': 0.78, 'fusion': 'dbsf', 'fast_accept_score': None, ...}
```

Fields that match the default don't appear — so the output size is a rough
proxy for "how far my corpus is from generic".

---

## Troubleshooting

### "Tuned config is *worse* than baseline"

This can't happen by construction if you're using `RAGTuner.tune()` —
baseline is enqueued as trial 0 and Optuna returns the best trial. If you see
this, one of these is wrong:

- You compared the tuned config against a *different* baseline (maybe your
  hand-tuned one instead of library defaults). Pass it as
  `baseline_config=your_existing`.
- You evaluated the tuned config on a *different* testset than you tuned on.
  That's just overfitting and it means your testset is too small or too
  narrow.
- Your testset is noisy (wrong expected IDs, duplicates, queries with no
  correct answer in the index). TPE will find a config that games the noise.

Fix: audit the testset. For each failing query, run it through production and
verify the expected ID is reachable at all.

### "Tuning plateaus after a few trials"

If `best_value` stops changing after 10–15 trials, TPE has converged
prematurely — usually because the search space is too narrow or the testset
is too small to discriminate configs.

Options:

1. **Expand ranges** by calling Optuna directly instead of `RAGTuner.tune()`.
   See `tests/eval_v2/run_tuner.py` for a custom objective function you can
   modify.
2. **Add testset diversity** — if every query is a short brand lookup, any
   reasonable `semantic_ratio` will score the same.
3. **Change the seed** — `tuner.tune(seed=7)` explores a different region.

### "ImportError: install optuna"

```
ImportError: Install optuna to use RAGTuner: `pip install rag7[tune]`
```

You imported `RAGTuner` without the optional extra. Fix:

```bash
pip install 'rag7[tune]'
```

### "Tuner evaluates every trial with the same latency"

If `mean_latency_ms` is identical across trials, your backend is probably
caching aggressively (Meilisearch, pgvector pgbouncer pools). This doesn't
affect correctness of the hit-rate score, but it means `latency_weight` is
effectively doing nothing. Restart the backend or disable caching during
tuning for honest latency measurements.

### "Best params look weird — `enable_quality_gate=False`?"

Totally valid. The quality gate costs a round-trip LLM call and is only worth
it when retrieval is occasionally wrong enough to need a judge. On a clean
corpus with strong BM25, the quality gate can be pure overhead. Trust the
metric, not your priors.

### "CLI says 'Missing dependency'"

```
Missing dependency: No module named 'meilisearch'
```

The CLI is Meilisearch+Azure-only — a convenience wrapper, not a universal
tool. Install the deps or use the Python API with your preferred backend.

---

## Beyond tuning

Tuning is one lever. A few others worth knowing:

- **[Auto-strategy](auto-strategy.md)** — `auto_strategy=True` samples your
  index at init-time and picks text fields, filter fields, and language
  hints automatically. Orthogonal to `RAGTuner`; you can use both.
- **[Rerankers](reranking.md)** — swap in Cohere, Jina, ColBERT, or RankGPT.
  A better reranker can lift hit@5 more than any amount of threshold tuning.
- **[Backends](backends.md)** — 8 backends, same API. If your current backend
  is slow or hit-rate-limited, try another.
- **[Memory](memory.md)** — persistent conversation context via
  `mem0` / Postgres / langgraph checkpoints, for multi-turn chat setups.
- **[Filtering](filtering.md)** — the LLM can build filter expressions from
  natural-language queries. Tune `enable_filter_intent` to control whether
  it runs.
- **[Tracing](tracing.md)** — OpenTelemetry traces for every retrieval
  stage. Essential if you're debugging why a tuned config isn't helping a
  specific query class.

Tuning gets you to "pretty good" fast. Combining tuning with a better
reranker, correct text fields (via auto-strategy), and the right backend gets
you to "stop fiddling, ship it".
