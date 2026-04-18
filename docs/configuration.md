# Configuration

All parameters can be set via constructor arguments or environment variables.

## Core parameters

| Parameter | Env var | Default | Description |
|-----------|---------|---------|-------------|
| `top_k` | `RAG_TOP_K` | `10` | Final result count returned to the LLM |
| `rerank_top_n` | `RAG_RERANK_TOP_N` | `5` | Candidates passed to the reranker |
| `retrieval_factor` | `RAG_RETRIEVAL_FACTOR` | `4` | Over-retrieval multiplier (`top_k × factor` fetched before reranking) |
| `max_iter` | `RAG_MAX_ITER` | `3` | Maximum retrieve-rewrite cycles |
| `semantic_ratio` | `RAG_SEMANTIC_RATIO` | `0.5` | Hybrid search semantic weight (0 = pure BM25, 1 = pure vector) |
| `fusion` | `RAG_FUSION` | `"rrf"` | Score fusion: `"rrf"` (Reciprocal Rank Fusion) or `"dbsf"` (Distribution-Based) |
| `hyde_min_words` | `RAG_HYDE_MIN_WORDS` | `8` | Minimum query word count to trigger HyDE |
| `verbose` | `RAG_VERBOSE` | `0` | Set to `1` to log pipeline steps |

## Constructor reference

```python
from rag7 import Agent

rag = Agent(
    index="docs",               # collection / index name
    backend=backend,            # SearchBackend instance (default: InMemoryBackend)
    collections=None,           # dict[str, SearchBackend] for multi-collection mode
    collection_descriptions={}, # human-readable descriptions for routing LLM
    llm=None,                   # fast LLM for routing, rewriting, quality gate
    gen_llm=None,               # generation LLM for the final answer
    reranker=None,              # reranker instance or alias string
    top_k=10,
    rerank_top_n=5,
    retrieval_factor=4,
    max_iter=3,
    semantic_ratio=0.5,
    fusion="rrf",
    instructions="",            # extra text appended to the system prompt
    embed_fn=None,              # callable (str) -> list[float]
    boost_fn=None,              # callable (doc_dict) -> float for business-signal boosting
    filter=None,                # always-on Meilisearch-style filter expression
    hyde_min_words=8,
    hyde_style_hint="",
    auto_strategy=True,         # sample docs at init and auto-configure
    group_field="",
    name_field="",
    verbose=False,
)
```

## `init_agent` parameters

`init_agent` is a convenience wrapper that accepts string aliases and builds the backend, LLM, and reranker for you:

```python
from rag7 import init_agent

rag = init_agent(
    index="docs",               # collection name (omit when using collections=)
    collections=None,           # list[str] or dict[str, description]
    model="openai:gpt-5.4",      # "provider:model" string
    gen_model=None,             # separate generation model (defaults to model)
    backend="memory",           # backend alias or SearchBackend instance
    backend_url=None,           # backend server URL
    backend_kwargs={},          # extra kwargs passed to the backend constructor
    reranker=None,              # reranker alias or instance
    reranker_model=None,        # model name for the reranker
    reranker_kwargs={},         # extra kwargs for the reranker constructor
    embed_fn=None,
    auto_strategy=True,
    **agent_kwargs,             # any Agent constructor kwarg
)
```

## LLM cache

rag7 can cache LLM calls (preprocessing, HyDE, quality gate) to avoid redundant API calls during development:

| Env var | Default | Description |
|---------|---------|-------------|
| `RAG7_CACHE` | `0` | Set to `1` to enable |
| `RAG7_CACHE_DIR` | _(none)_ | Path for disk-based JSON cache |
| `RAG7_CACHE_PG_URL` | _(none)_ | PostgreSQL connection string for persistent cache |

Disk cache example:

```bash
RAG7_CACHE=1 RAG7_CACHE_DIR=./.cache python my_app.py
```
