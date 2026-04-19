# Examples

Eighteen ready-to-run patterns covering the most common rag7 use cases.

---

### 1. Knowledge base Q&A (InMemory, no external services)

```python
from rag7 import Agent, InMemoryBackend
import hashlib

def embed(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:8]]

backend = InMemoryBackend(embed_fn=embed)
backend.add_documents([
    {"id": "1", "content": "Hybrid search combines BM25 and vector retrieval."},
    {"id": "2", "content": "RRF fusion ranks results by reciprocal rank."},
])

rag = Agent(index="kb", backend=backend, auto_strategy=False)
state = rag.invoke("What is hybrid search?")
print(state.answer)
```

---

### 2. Retrieve documents without generating an answer

```python
from rag7 import init_agent

rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant",
                 backend_url="http://localhost:6333")

query, docs = rag.retrieve_documents("What is retrieval-augmented generation?")
for doc in docs:
    print(doc.page_content[:100])
```

---

### 3. Multi-turn chat

```python
from rag7 import init_agent, ConversationTurn

rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant",
                 backend_url="http://localhost:6333")

history: list[ConversationTurn] = []

for question in ["What is RAG?", "How does it compare to fine-tuning?"]:
    state = rag.chat(question, history)
    history.append(ConversationTurn(question=question, answer=state.answer))
    print(f"Q: {question}\nA: {state.answer}\n")
```

---

### 4. Always-on filter (e-commerce in-stock)

```python
from rag7 import init_agent

rag = init_agent(
    "products",
    model="openai:gpt-5.4",
    backend="meilisearch",
    filter="in_stock = true",
)

state = rag.invoke("cordless drill under 100 euros")
print(state.answer)
```

---

### 5. Exclude a category from every search

```python
from rag7 import init_agent

rag = init_agent(
    "products",
    model="openai:gpt-5.4",
    backend="meilisearch",
    filter="category != 'discontinued'",
)

state = rag.invoke("Find alternatives to brake cleaner 500ml")
print(state.answer)
```

---

### 6. Async usage (FastAPI / Databricks / Jupyter)

```python
import asyncio
from rag7 import init_agent

async def main():
    rag = init_agent("kb", model="openai:gpt-5.4", backend="qdrant",
                     backend_url="http://localhost:6333")
    state = await rag.ainvoke("What is hybrid search?")
    print(state.answer)

asyncio.run(main())
```

`rag.invoke()` also works from Jupyter / Databricks without `asyncio.run()` — rag7 handles running-loop detection automatically.

---

### 7. Tool-calling agent — dynamic filter discovery

```python
from rag7 import init_agent

rag = init_agent("products", model="openai:gpt-5.4", backend="meilisearch")

# The LLM inspects the index schema, samples field values, and builds
# the filter expression itself — no hardcoded filter needed.
result = rag.invoke_agent("Show me Bosch products under 50 euros in stock")
print(result)
```

---

### 8. Streaming the final answer

```python
import asyncio
from rag7 import init_agent

async def main():
    rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant",
                     backend_url="http://localhost:6333")
    async for chunk in rag.astream("Explain retrieval-augmented generation"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

### 9. Qdrant with metadata filter

```python
from rag7 import Agent
from rag7.backend import QdrantBackend
from qdrant_client.models import FieldCondition, MatchValue, Filter

backend = QdrantBackend(
    "docs",
    url="http://localhost:6333",
    embed_fn=my_embed_fn,
)

rag = Agent(index="docs", backend=backend)

from rag7.backend import SearchRequest
hits = backend.search(SearchRequest(
    query="hybrid search",
    limit=5,
    filter_expr=Filter(must=[FieldCondition(key="category", match=MatchValue(value="tech"))]),
))
```

---

### 10. Custom instructions (tone / domain)

```python
from rag7 import init_agent

rag = init_agent(
    "legal-docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    backend_url="http://localhost:6333",
    instructions=(
        "You are a legal assistant. Always cite the specific clause or section. "
        "Use formal language and note when something requires professional legal advice."
    ),
)

state = rag.invoke("What are the termination conditions in the SLA?")
print(state.answer)
```

---

### 11. Auto-tune for your corpus

Optuna explores ~20 retrieval knobs against a small hand-crafted testset, saves
the best config as TOML for your deployment.

```python
from rag7 import MeilisearchBackend
from rag7.tuner import RAGTuner, load_testset
from rag7.utils import _make_azure_embed_fn

tuner = RAGTuner(
    backend_factory=lambda: MeilisearchBackend("products"),
    embed_fn=_make_azure_embed_fn(),
    hit_cases=load_testset("testset.json"),
    eval_k=5,
)

best = tuner.tune(n_trials=50, patience=8)     # early-stops after 8 no-improve
best.save_toml("rag7.config.toml")              # gitignored — local override
```

Or from the CLI:

```bash
python -m rag7.tuner --index products --hits testset.json --trials 50
```

---

### 12. Load auto-discovered config

`RAGConfig.auto()` walks `rag7.config.toml` → `pyproject.toml [tool.rag7]` → env
vars → defaults. No manual wiring.

```python
from rag7 import AgenticRAG, RAGConfig, MeilisearchBackend

rag = AgenticRAG(
    index="products",
    backend=MeilisearchBackend("products"),
    embed_fn=my_embed_fn,
    config=RAGConfig.auto(),
)
```

---

### 13. Mix LLM tiers — cheap weak, strong generation

Use a small model for high-frequency calls (preprocess, filter-intent) and your
best model only for the final answer.

```python
from rag7 import AgenticRAG, RAGConfig, MeilisearchBackend

cfg = RAGConfig(
    strong_model="azure:gpt-5.4",        # final answer — quality matters
    weak_model="azure:gpt-5.4-mini",     # preprocess / quality / filter-intent
    thinking_model="azure:gpt-5.4-mini", # per-doc reasoning
)

rag = AgenticRAG(
    index="products",
    backend=MeilisearchBackend("products"),
    embed_fn=my_embed_fn,
    config=cfg,
)
```

Resolves through `langchain.chat_models.init_chat_model` — any
`provider:model` string works: `azure:`, `openai:`, `anthropic:`,
`bedrock:`, `ollama:`, etc.

---

### 14. Disable optional stages in TOML

Commit a `[tool.rag7]` block in `pyproject.toml` to disable stages your corpus
doesn't need:

```toml
[tool.rag7]
semantic_ratio = 0.4
fusion = "dbsf"

# Disable these entirely (first-class tuning option):
disable = ["bm25_fallback_threshold", "expert_threshold"]
# Toggle off the LLM-based preprocess (short-keyword product queries):
enable_preprocess_llm = false
```

TOML has no null; `disable = [...]` tells rag7 to set those fields to `None`
on load — distinct from "use default" (just omit the field).

---

### 15. Multi-turn with follow-up context

rag7 rewrites short follow-ups using prior-turn context before retrieval.
Works out of the box — no extra wiring.

```python
from rag7 import init_agent, ConversationTurn

rag = init_agent("products", model="azure:gpt-5.4", backend="meilisearch")

# Turn 1
s1 = rag.chat("Makita Akku Bohrhammer 18V", history=[])
history = [ConversationTurn(question="Makita Akku Bohrhammer 18V", answer=s1.answer)]

# Turn 2 — "und die 36V Version?" gets rewritten to
# "Makita Akku Bohrhammer 36V" before retrieval fires
s2 = rag.chat("und die 36V Version?", history=history)
print(s2.answer)
```

---

### 16. Multilingual filter intent (DE / FR / IT / EN)

Lowercase queries with filter-intent words (`von`, `de`, `di`, `from`, `ohne`,
`sans`, `senza`, `without`, …) trigger LLM filter extraction automatically —
no capitalization required.

```python
from rag7 import init_agent

rag = init_agent("products", model="azure:gpt-5.4", backend="meilisearch")

# All of these trigger filter-intent detection on supplier_name:
queries = [
    "trockenbeton von fixit",          # DE — "from Fixit"
    "ciment de fixit",                 # FR
    "cemento di fixit",                # IT
    "concrete from fixit",             # EN
    "rohre ohne pvc",                  # DE negation — "without PVC"
]
for q in queries:
    _, docs = rag.retrieve_documents(q, top_k=1)
    print(q, "→", docs[0].metadata.get("supplier_name"))
```

---

### 17. Measure and weight latency in your eval

`RAGTuner` tracks per-query wall time and exposes `mean_latency_ms` +
`speed` + `combined_prod` metrics so you can pick a config that's both
accurate *and* fast.

```python
from rag7.tuner import RAGTuner, load_testset

tuner = RAGTuner(
    backend_factory=lambda: my_backend,
    embed_fn=my_embed_fn,
    hit_cases=load_testset("testset.json"),
    latency_weight=0.25,        # quarter weight to latency
    latency_budget_ms=1200,     # queries over this budget get linear penalty
)
best = tuner.tune(n_trials=30)

print(f"hit@{tuner.eval_k}: {best.overrides()}")
```

---

### 18. Inspect the auto-init cache

rag7 caches the LLM's auto-strategy result per schema-fingerprint so repeat
initializations against the same corpus skip the LLM call entirely.

```python
from pathlib import Path
import json

cache_dir = Path.home() / ".cache" / "rag7"
for f in cache_dir.glob("auto_*.json"):
    strategy = json.loads(f.read_text())
    print(f"{f.stem}:")
    print(f"  semantic_ratio: {strategy.get('semantic_ratio')}")
    print(f"  fusion:         {strategy.get('fusion')}")
    print(f"  domain_hint:    {strategy.get('domain_hint','')[:80]}")
```

Delete any file to force a fresh LLM call on next init (useful after schema
changes the fingerprint doesn't catch).
