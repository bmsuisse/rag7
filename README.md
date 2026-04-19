# rag7 🕵️🍸🚗🎯 — Licensed to Retrieve

<div align="center">

**Not just hybrid search. A true autonomous retrieval agent.**  
Shaken, not stirred — plug in any vector store, any LLM, any reranker.  
The mission: find the right documents, neutralise irrelevant noise, and deliver the answer. Every time.

[![PyPI](https://img.shields.io/pypi/v/rag7)](https://pypi.org/project/rag7/)
[![Python](https://img.shields.io/pypi/pyversions/rag7)](https://pypi.org/project/rag7/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/bmsuisse/rag7/actions/workflows/ci.yml/badge.svg)](https://github.com/bmsuisse/rag7/actions/workflows/ci.yml)

</div>

---

```python
from rag7 import init_agent

rag = init_agent("documents", model="openai:gpt-5.4", backend="qdrant")
state = rag.chat("What is the status of operation overlord?")
# Your answer. Shaken, not stirred.
```

---

## 🎯 Scope — Retrieval, Not Ingestion

**rag7 is built for optimal retrieval.** The mission is finding the right
documents at query time — hybrid search, reranking, query rewriting,
autonomous retry loops, and an LLM quality gate.

**Ingestion is out of scope.** rag7 does not chunk, clean, embed-at-scale, or
index your corpus. Use your stack of choice for that —
[Docling](https://github.com/docling-project/docling),
[Unstructured](https://unstructured.io),
[LlamaIndex](https://www.llamaindex.ai) ingestion pipelines, a Databricks job,
a custom script — then point rag7 at the resulting index. Every backend
exposes a minimal `add_documents()` helper for convenience and smoke tests,
but it is not meant to replace a real ingestion pipeline.

Keeping the surface narrow is deliberate: one thing, done well.

---

## 🕵️ The Agent

*"We have a problem. Millions of documents. One question. And the clock is ticking."*

Most retrieval systems send a junior analyst — one query, one pass, done. Fast, cheap, and dangerously incomplete.

**So they sent rag7.** Licensed to retrieve. Never satisfied with *good enough*.

Before every mission, rag7 visits Q's lab: **8 backends** to operate from, **any LLM** as the intelligence source, **precision rerankers** to separate signal from noise, and a **tool-calling agent** that inspects schemas, builds filters on the fly, and adapts to whatever the index throws at it.

In the field, it **plans**, **infiltrates**, and **interrogates** — running parallel searches across BM25 and vector space, fusing the evidence, and cross-examining every result through an LLM quality gate. When the trail goes cold, it rewrites the query and tries again. It doesn't stop until the mission is complete.

Only once the evidence is airtight does it surface the answer. **Cited. Grounded. Delivered.**

> 🍸 *"Shaken, not stirred — and always on target."* 🎯

Not in the name of any crown or government. In the name of **whoever is seeking the truth in their data**.

---

## 🕵️ How It Works

Most RAG libraries are **pipelines** — query in, documents out, done. rag7 is an **agent**.

Like a field operative, it doesn't execute a single search and report back. It thinks, adapts, and keeps going until the mission is complete:

1. 🧠 **Understands the intent** — rewrites your query into precise search keywords, detects whether it's a keyword lookup or semantic question, and adjusts the hybrid search ratio accordingly
2. 🔍 **Searches intelligently** — runs multiple query variants simultaneously across BM25 and vector search, fuses the results, and re-ranks with a dedicated reranker
3. 🧐 **Judges the results** — an LLM quality gate evaluates whether the retrieved documents actually answer the question
4. 🔄 **Adapts autonomously** — if results are off-target, rewrites the query and tries again; if a single approach fails, fans out into a swarm of parallel search strategies
5. ✍️ **Delivers the answer** — only once it's confident the evidence is solid does it generate a cited, grounded response

This is the difference between a search box and a field agent.

---

## ✨ Features

- 🚗 **Fast as an Aston Martin** — fully async pipeline, parallel HyDE + preprocessing, zero blocking calls
- 🎯 **On target, every time** — LLM quality gate rejects weak results and rewrites the query until the evidence is airtight
- 🔬 **Deep research, not shallow search** — multi-query swarm fans out across BM25 and vector space simultaneously, fusing intelligence from every angle
- 🃏 **Always has an ace up its sleeve** — when one approach fails, swarm retrieval deploys parallel strategies as backup
- 🕵️ **True agentic loop** — retrieve → judge → rewrite → retry, fully autonomous, up to `max_iter` rounds
- 🔍 **Hybrid search** — BM25 + vector, fused with RRF or DBSF
- 🧠 **HyDE** — hypothetical document embeddings for better recall on vague queries
- 🛠️ **Tool-calling agent** — `get_index_settings`, `get_filter_values`, `search_hybrid`, `search_bm25`, `rerank_results` — LLM picks tools dynamically
- 🏆 **Multi-reranker** — Cohere, HuggingFace, Jina, ColBERT, RankGPT, embed-anything, or custom
- 🗄️ **8 backends** — Meilisearch, Azure AI Search, ChromaDB, LanceDB, Qdrant, pgvector, DuckDB, InMemory
- 🤖 **Any LLM** — OpenAI, Azure, Anthropic, Ollama, Vertex AI, or any LangChain model
- ⚡ **One-line init** — `init_agent("docs", model="openai:gpt-5.4", backend="qdrant")` — no imports needed
- 💬 **Multi-turn chat** — conversation history with citation-aware answers
- 🎯 **Auto-strategy** — LLM samples your collection and tunes itself automatically
- 🔄 **Async-native** — every operation has a sync and async variant

---

## 📦 Install

```bash
# Recommended — Meilisearch + Cohere reranker + interactive CLI
pip install rag7[recommended]

# Base only — in-memory backend, BM25 keyword search
pip install rag7
```

| Extra | What you get | Command |
|-------|-------------|---------|
| **`recommended`** | Meilisearch + Cohere reranker + Rich CLI | `pip install rag7[recommended]` |
| `cli` | Interactive CLI with guided setup wizard | `pip install rag7[cli]` |
| `all` | Every backend + reranker + CLI | `pip install rag7[all]` |

<details>
<summary>🍸 Bond Edition extras — because every mission needs a code name</summary>

| Extra | Code name | Stack |
|-------|-----------|-------|
| `goldeneye` | GoldenEye | Meilisearch + Cohere + CLI — the classic recommended loadout |
| `skyfall` | Skyfall | Everything. All backends, all rerankers, all CLI — nothing left behind |
| `thunderball` | Thunderball | Qdrant + Cohere + CLI — vector power meets precision reranking |
| `moonraker` | Moonraker | ChromaDB + HuggingFace — fully local, no API keys, off the grid |
| `goldfinger` | Goldfinger | Azure AI Search + Azure OpenAI + Cohere — all gold, all cloud |
| `spectre` | Spectre | pgvector + HuggingFace — open-source shadow ops, no paid APIs |
| `casino-royale` | Casino Royale | ChromaDB + Jina — lightweight first mission |
| `licence-to-kill` | Licence to Kill | embed-anything + ChromaDB — Rust-powered, fully local, zero API keys |

```bash
pip install rag7[goldeneye]      # 🍸 The classic
pip install rag7[skyfall]        # 💥 Everything falls into place
pip install rag7[thunderball]    # ⚡ Vector power + precision
pip install rag7[moonraker]     # 🌙 Fully local, no API keys
pip install rag7[goldfinger]     # ☁️  All Azure, all gold
pip install rag7[spectre]        # 👻 Open-source, no paid APIs
pip install rag7[casino-royale]  # 🎰 Lightweight first mission
pip install rag7[licence-to-kill] # 🦀 Rust-powered, fully local
```

</details>

<details>
<summary>Individual backends &amp; rerankers</summary>

```bash
pip install rag7[meilisearch]     # 🔎 Meilisearch
pip install rag7[azure]           # ☁️  Azure AI Search
pip install rag7[chromadb]        # 🟣 ChromaDB
pip install rag7[lancedb]         # 🏹 LanceDB
pip install rag7[pgvector]        # 🐘 PostgreSQL + pgvector
pip install rag7[qdrant]          # 🟡 Qdrant
pip install rag7[duckdb]          # 🦆 DuckDB
pip install rag7[cohere]          # 🏅 Cohere reranker
pip install rag7[huggingface]     # 🤗 HuggingFace cross-encoder (local)
pip install rag7[jina]            # 🌊 Jina reranker
pip install rag7[rerankers]       # 🎯 rerankers (ColBERT, Flashrank, RankGPT, …)
pip install rag7[embed-anything]  # 🦀 Embed-anything (local Rust-accelerated embeddings + reranking)
```

Mix and match: `pip install rag7[qdrant,cohere,cli]`

</details>

---

## 🚀 Quick Start

### One-liner with `init_agent`

The fastest way to get started — no provider imports, string aliases for everything:

```python
from rag7 import init_agent

# Minimal — in-memory backend, LLM from env vars
rag = init_agent("docs")

# OpenAI + Qdrant + Cohere reranker
rag = init_agent(
    "my-collection",
    model="openai:gpt-5.4",
    backend="qdrant",
    backend_url="http://localhost:6333",
    reranker="cohere",
)

# Anthropic + Azure AI Search (native vectorisation, no client-side embeddings)
rag = init_agent(
    "my-index",
    model="anthropic:claude-sonnet-4-6",
    gen_model="anthropic:claude-opus-4-6",
    backend="azure",
    backend_url="https://my-search.search.windows.net",
    reranker="huggingface",
    auto_strategy=True,
)

# Fully local — Ollama + ChromaDB + HuggingFace cross-encoder
rag = init_agent(
    "docs",
    model="ollama:llama3",
    backend="chroma",
    reranker="huggingface",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
```

### Multi-collection routing

Pass several collections and let the agent decide which to search. The LLM
picks the relevant subset before retrieval, using either the collection names
alone or optional natural-language descriptions.

```python
from rag7 import init_agent

# List form — LLM routes by name only
rag = init_agent(
    collections=["products", "faq", "policies"],
    backend="qdrant",
    backend_url="http://localhost:6333",
    model="openai:gpt-5.4",
)

# Dict form — LLM routes using descriptions (better precision)
rag = init_agent(
    collections={
        "products": "Product catalog: SKUs, prices, specs, availability",
        "faq":      "Customer-facing FAQ, troubleshooting, return policy",
        "policies": "Internal HR/legal/compliance policy documents",
    },
    backend="qdrant",
    backend_url="http://localhost:6333",
    model="openai:gpt-5.4",
)

rag.invoke("What's our return policy?")       # → routes to faq / policies
rag.invoke("Price of SKU 12345?")              # → routes to products
```

Each retrieved document carries its origin in `metadata["_collection"]` so you
can merge, filter, or attribute citations downstream. One backend instance is
built per collection; they share the same backend type and URL.

**Backend aliases**

| Alias | Class | Extra |
|-------|-------|-------|
| `"memory"` / `"in_memory"` | `InMemoryBackend` | _(none)_ |
| `"chroma"` / `"chromadb"` | `ChromaDBBackend` | `rag7[chromadb]` |
| `"qdrant"` | `QdrantBackend` | `rag7[qdrant]` |
| `"lancedb"` / `"lance"` | `LanceDBBackend` | `rag7[lancedb]` |
| `"duckdb"` | `DuckDBBackend` | `rag7[duckdb]` |
| `"pgvector"` / `"pg"` | `PgvectorBackend` | `rag7[pgvector]` |
| `"meilisearch"` | `MeilisearchBackend` | `rag7[meilisearch]` |
| `"azure"` | `AzureAISearchBackend` | `rag7[azure]` |

**Reranker aliases**

| Alias | Class | `reranker_model` | Extra |
|-------|-------|-----------------|-------|
| `"cohere"` | `CohereReranker` | Cohere model name (default: `rerank-v3.5`) | `rag7[cohere]` |
| `"huggingface"` / `"hf"` | `HuggingFaceReranker` | HF model name (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) | `rag7[huggingface]` |
| `"jina"` | `JinaReranker` | Jina model name (default: `jina-reranker-v2-base-multilingual`) | `rag7[jina]` |
| `"llm"` | `LLMReranker` | _(uses the agent's LLM)_ | _(none)_ |
| `"rerankers"` | `RerankersReranker` | Any model from the `rerankers` library | `rag7[rerankers]` |
| `"embed-anything"` | `EmbedAnythingReranker` | ONNX reranker model (default: `jina-reranker-v1-turbo-en`) | `rag7[embed-anything]` |

```python
# Cohere (default model)
rag = init_agent("docs", model="openai:gpt-5.4", reranker="cohere")

# HuggingFace — multilingual model
rag = init_agent("docs", model="openai:gpt-5.4", reranker="huggingface",
                 reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Jina
rag = init_agent("docs", model="openai:gpt-5.4", reranker="jina")  # uses JINA_API_KEY

# ColBERT via rerankers library
rag = init_agent("docs", model="openai:gpt-5.4", reranker="rerankers",
                 reranker_model="colbert-ir/colbertv2.0",
                 reranker_kwargs={"model_type": "colbert"})

# Pass a pre-built reranker instance directly
from rag7 import CohereReranker
rag = init_agent("docs", reranker=CohereReranker(model="rerank-v3.5", api_key="..."))
```

**Model strings:** any `"provider:model-name"` from LangChain's `init_chat_model` — `openai`, `anthropic`, `azure_openai`, `google_vertexai`, `ollama`, `groq`, `mistralai`, and more

### Manual setup

```python
from rag7 import Agent, InMemoryBackend

backend = InMemoryBackend(embed_fn=my_embed_fn)
backend.add_documents([
    {"content": "RAG combines retrieval with generation", "source": "wiki"},
    {"content": "Vector search finds similar embeddings", "source": "docs"},
])

rag = Agent(index="demo", backend=backend)

# Single query → full answer
state = rag.invoke("What is retrieval-augmented generation?")
print(state.answer)

# Retrieve only — documents without LLM answer
query, docs = rag.retrieve_documents("What is retrieval-augmented generation?")
for doc in docs:
    print(doc.page_content)

# Override top-K at call time
query, docs = rag.retrieve_documents("hybrid search", top_k=3)
```

### `Agent.from_model` — model string with explicit backend

```python
from rag7 import Agent, QdrantBackend

rag = Agent.from_model(
    "openai:gpt-5.4-mini",          # fast model for routing & rewriting
    index="docs",
    gen_model="openai:gpt-5.4",     # powerful model for the final answer
    backend=QdrantBackend("docs", url="http://localhost:6333"),
)
```

---

## 💬 Multi-turn Chat

```python
from rag7 import Agent, ConversationTurn

rag = Agent(index="articles")
history: list[ConversationTurn] = []

state = rag.chat("What is hybrid search?", history)
history.append(ConversationTurn(question="What is hybrid search?", answer=state.answer))

state = rag.chat("How does it compare to pure vector search?", history)
print(state.answer)
print(f"Sources: {len(state.documents)}")
```

Async variant:

```python
state = await rag.achat("What is hybrid search?", history)
```

---

## 🏗️ Architecture

rag7 has two operating modes — both fully autonomous:

### Graph mode (`rag.chat` / `rag.invoke`)

The default. A LangGraph state machine that runs the full agentic pipeline:

```
Query
  │
  ├─[HyDE]──────────────────────────────────────────┐
  │  Hypothetical document embedding (parallel)      │
  │                                                  ▼
  ▼                                         [Embed HyDE text]
[Preprocess]                                         │
  Extract keywords + variants                        │
  Detect semantic_ratio + fusion strategy            │
  │                                                  │
  └──────────────────────────────────────────────────┘
                        │
                        ▼
              [Hybrid Search × N queries]
               BM25 + Vector, multi-arm
                        │
                        ▼
               [RRF / DBSF Fusion]
                        │
                        ▼
                    [Rerank]
               Cohere / HF / Jina / embed-anything / LLM
                        │
                        ▼
               [Quality Gate]
               LLM judges relevance
                   │         │
                (good)     (bad)
                   │         │
                   ▼         ▼
              [Generate]  [Rewrite] ──► loop (max_iter)
                   │
                   ▼
        Answer + [n] inline citations
```

### Tool-calling agent mode (`rag.invoke_agent`)

The agent receives a set of tools and reasons step-by-step, calling them in whatever order makes sense for the question. No fixed pipeline — pure field improvisation:

```
Query
  │
  ▼
[LLM Agent]  ◄──────────────────────────────────────┐
  Thinks: "What do I need to answer this?"           │
  │                                                  │
  ├── get_index_settings()                           │
  │   Discover filterable / sortable / boost fields  │
  │                                                  │
  ├── get_filter_values(field)                       │
  │   Sample real stored values for a field          │
  │   → build precise filter expressions             │
  │                                                  │
  ├── search_hybrid(query, filter, sort_fields)      │
  │   BM25 + vector, optional filter + sort boost    │
  │                                                  │
  ├── search_bm25(query, filter)                     │
  │   Fallback pure keyword search                   │
  │                                                  │
  ├── rerank_results(query, hits)                    │
  │   Re-rank with configured reranker               │
  │                                                  │
  └── [needs more info?] ─────────────────────────► │

  [done]
  │
  ▼
Answer  (tool calls explained inline)
```

Use `invoke_agent` when questions involve **dynamic filtering** — the agent inspects the index schema, samples real field values, builds filters on the fly, and decides whether to sort by business signals like popularity or recency.

---

## 📚 Examples

### 1. Knowledge base Q&A (InMemory, no external services)

```python
from rag7 import AgenticRAG, InMemoryBackend
from langchain_openai import ChatOpenAI

docs = [
    {"id": "1", "content": "The Eiffel Tower was built in 1889 for the World's Fair in Paris.", "topic": "history"},
    {"id": "2", "content": "The Louvre is the world's largest art museum, located in Paris.", "topic": "art"},
    {"id": "3", "content": "Photosynthesis converts sunlight and CO2 into glucose and oxygen.", "topic": "science"},
    {"id": "4", "content": "The Python programming language was created by Guido van Rossum in 1991.", "topic": "tech"},
    {"id": "5", "content": "Machine learning is a subset of artificial intelligence.", "topic": "tech"},
]

backend = InMemoryBackend(documents=docs)
llm = ChatOpenAI(model="gpt-5.4-mini")

rag = AgenticRAG(index="kb", backend=backend, llm=llm, gen_llm=llm)

state = rag.invoke("When was the Eiffel Tower built?")
print(state.answer)
# → "The Eiffel Tower was built in 1889 for the World's Fair in Paris. [1]"
print(state.query)        # rewritten query
print(state.iterations)   # how many retrieval rounds it took
```

---

### 2. Retrieve documents without generating an answer

Useful when you want the docs and will handle the answer yourself:

```python
from rag7 import AgenticRAG, InMemoryBackend

rag = AgenticRAG(index="kb", backend=backend)

query, docs = rag.retrieve_documents("machine learning", top_k=3)
print(f"Rewritten query: {query}")
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)  # original fields + _rankingScore
```

---

### 3. Multi-turn chat

```python
from rag7 import AgenticRAG, InMemoryBackend, ConversationTurn

rag = AgenticRAG(index="kb", backend=backend, llm=llm, gen_llm=llm)
history: list[ConversationTurn] = []

q1 = "What is machine learning?"
s1 = rag.chat(q1, history)
history.append(ConversationTurn(question=q1, answer=s1.answer))
print(s1.answer)

q2 = "How does it relate to AI?"   # pronoun resolved from history
s2 = rag.chat(q2, history)
history.append(ConversationTurn(question=q2, answer=s2.answer))
print(s2.answer)
```

---

### 4. Always-on filter (e-commerce: in-stock items only)

```python
from rag7 import AgenticRAG, MeilisearchBackend

backend = MeilisearchBackend(
    "products",
    url="http://localhost:7700",
    api_key="masterKey",
)

# Every search is scoped to in-stock items — no per-call boilerplate
rag = AgenticRAG(
    index="products",
    backend=backend,
    filter="is_in_stock = true",
    llm=llm,
    gen_llm=llm,
)

state = rag.invoke("red running shoes size 42")
for doc in state.documents:
    print(doc.metadata["product_name"], "|", doc.metadata["price"])
```

---

### 5. Filter + own-brand exclusion

```python
# Exclude own-brand articles and search for third-party alternatives
rag = AgenticRAG(
    index="products",
    backend=backend,
    filter="is_own_brand = false",
    llm=llm,
    gen_llm=llm,
)

state = rag.invoke("Find alternatives to our house-brand brake cleaner 500ml")
print(state.answer)
# LLM strips the brand prefix, rewrites to "brake cleaner 500ml",
# filter ensures only third-party results are returned.
```

---

### 6. Async usage (FastAPI / Databricks / Jupyter)

```python
import asyncio
from rag7 import AgenticRAG, InMemoryBackend

rag = AgenticRAG(index="kb", backend=backend, llm=llm, gen_llm=llm)

# Async single query
state = await rag.ainvoke("What is photosynthesis?")
print(state.answer)

# Async batch — runs all queries in parallel
states = await rag.abatch([
    "What is photosynthesis?",
    "Who created Python?",
    "Where is the Louvre?",
])
for s in states:
    print(s.answer)
```

Sync variants work from any context including Databricks/Jupyter (running event loop is handled automatically):

```python
# Safe to call from a notebook cell even with a running event loop
state = rag.invoke("What is photosynthesis?")
states = rag.batch(["question one", "question two"])
```

---

### 7. Tool-calling agent — dynamic filter discovery

When you don't know the filter values upfront, the agent inspects the schema and samples field values itself:

```python
from rag7 import AgenticRAG, MeilisearchBackend

rag = AgenticRAG(
    index="products",
    backend=MeilisearchBackend("products", url="http://localhost:7700"),
    llm=llm,
    gen_llm=llm,
)

# Agent calls get_index_settings() → get_filter_values("brand") →
# search_hybrid(filter="brand = 'Bosch'", sort_fields=["popularity"])
result = rag.invoke_agent("Show me the most popular Bosch power tools")
print(result)
```

---

### 8. Streaming the final answer

```python
async def stream_answer():
    async for chunk in rag.astream("Explain hybrid search in simple terms"):
        print(chunk, end="", flush=True)

asyncio.run(stream_answer())
```

---

### 9. Qdrant — vector search with metadata filter

```python
from rag7 import AgenticRAG, QdrantBackend
from qdrant_client import QdrantClient, models

# Insert docs (done once)
client = QdrantClient("http://localhost:6333")
client.upsert("articles", points=[
    models.PointStruct(id=1, vector=embed("RAG combines retrieval and generation"),
                       payload={"content": "RAG combines retrieval and generation", "year": 2023}),
    models.PointStruct(id=2, vector=embed("Vector databases store high-dimensional embeddings"),
                       payload={"content": "Vector databases store high-dimensional embeddings", "year": 2022}),
])

from qdrant_client.models import FieldCondition, MatchValue

rag = AgenticRAG(
    index="articles",
    backend=QdrantBackend("articles", url="http://localhost:6333", embed_fn=embed),
    llm=llm,
    gen_llm=llm,
)

# Pass native Qdrant filter dict — no string translation needed
state = rag.invoke("what is RAG?")
# Or with explicit filter at retrieve time:
_, docs = rag.retrieve_documents("vector databases")
```

---

### 10. Custom instructions (tone / domain)

```python
rag = AgenticRAG(
    index="legal_docs",
    backend=backend,
    llm=llm,
    gen_llm=llm,
    instructions=(
        "You are a legal assistant. Answer in formal language. "
        "Always cite the article number when referencing a law. "
        "If the context is insufficient, say so explicitly."
    ),
)

state = rag.invoke("What are the notice periods for dismissal?")
print(state.answer)
```

---

## 🗄️ Backends

### ☁️ Azure AI Search

Native hybrid search — no client-side embeddings needed when the index has an integrated vectorizer:

```python
from rag7 import Agent, AzureAISearchBackend

# Native vectorization — service embeds the query server-side
rag = Agent(
    index="my-index",
    backend=AzureAISearchBackend(
        "my-index",
        endpoint="https://my-search.search.windows.net",
        api_key="...",
    ),
)

# Client-side vectorization
rag = Agent(
    index="my-index",
    backend=AzureAISearchBackend(
        "my-index",
        endpoint="https://my-search.search.windows.net",
        api_key="...",
        embed_fn=my_embed_fn,
    ),
)

# With Azure semantic reranking
rag = Agent(
    index="my-index",
    backend=AzureAISearchBackend(
        "my-index",
        endpoint="https://my-search.search.windows.net",
        api_key="...",
        semantic_config="my-semantic-config",
    ),
)
```

### 🟡 Qdrant

```python
from rag7 import Agent, QdrantBackend

rag = Agent(
    index="my_collection",
    backend=QdrantBackend("my_collection", url="http://localhost:6333", embed_fn=my_embed_fn),
)
```

### 🟣 ChromaDB

```python
from rag7 import Agent, ChromaDBBackend

rag = Agent(
    index="my_collection",
    backend=ChromaDBBackend("my_collection", path="./chroma_db", embed_fn=my_embed_fn),
)
```

### 🏹 LanceDB

```python
from rag7 import Agent, LanceDBBackend

rag = Agent(
    index="docs",
    backend=LanceDBBackend("docs", db_uri="./lancedb", embed_fn=my_embed_fn),
)
```

### 🐘 PostgreSQL + pgvector

```python
from rag7 import Agent, PgvectorBackend

rag = Agent(
    index="documents",
    backend=PgvectorBackend(
        "documents",
        dsn="postgresql://user:pass@localhost:5432/mydb",
        embed_fn=my_embed_fn,
    ),
)
```

### 🦆 DuckDB

```python
from rag7 import Agent, DuckDBBackend

rag = Agent(
    index="vectors",
    backend=DuckDBBackend("vectors", db_path="./my.duckdb", embed_fn=my_embed_fn),
)
```

### 🔎 Meilisearch

```python
from rag7 import Agent, MeilisearchBackend

rag = Agent(
    index="articles",
    backend=MeilisearchBackend("articles", url="http://localhost:7700", api_key="masterKey"),
)
```

### 📦 InMemory (default, zero dependencies)

```python
from rag7 import Agent, InMemoryBackend

backend = InMemoryBackend(embed_fn=my_embed_fn)
backend.add_documents([
    {"content": "RAG combines retrieval with generation", "source": "wiki"},
    {"content": "Vector search finds similar embeddings", "source": "docs"},
])

rag = Agent(index="demo", backend=backend)
```

---

## 🤖 LLM Configuration

Pass a pre-built LangChain model or use `init_agent` / `Agent.from_model` for string-based init.  
When using `Agent` directly, configure via env vars or pass an explicit model instance.

### OpenAI

```python
from langchain_openai import ChatOpenAI
from rag7 import Agent

rag = Agent(
    index="articles",
    llm=ChatOpenAI(model="gpt-5.4", api_key="sk-..."),
    gen_llm=ChatOpenAI(model="gpt-5.4", api_key="sk-..."),
)
```

### Azure OpenAI (explicit keys)

```python
from langchain_openai import AzureChatOpenAI
from rag7 import Agent

llm = AzureChatOpenAI(
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="gpt-5.4",
    api_key="...",
    api_version="2024-12-01-preview",
)
rag = Agent(index="articles", llm=llm, gen_llm=llm)
```

### Azure OpenAI (env vars)

```python
# Set: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
from rag7 import Agent

rag = Agent(index="articles")  # auto-detected
```

### Azure OpenAI with Managed Identity (no API key)

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from rag7 import Agent

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
llm = AzureChatOpenAI(
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="gpt-5.4",
    azure_ad_token_provider=token_provider,
    api_version="2024-12-01-preview",
)
rag = Agent(index="articles", llm=llm, gen_llm=llm)
```

### Anthropic Claude

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from rag7 import Agent

llm = ChatAnthropic(model="claude-sonnet-4-6", api_key="sk-ant-...")
rag = Agent(index="articles", llm=llm, gen_llm=llm)
```

### Ollama (local, no API key)

```bash
pip install langchain-ollama
```

```python
from langchain_ollama import ChatOllama
from rag7 import Agent

rag = Agent(
    index="articles",
    llm=ChatOllama(model="llama3.2", base_url="http://localhost:11434"),
    gen_llm=ChatOllama(model="llama3.2", base_url="http://localhost:11434"),
)
```

### Google Vertex AI

```bash
pip install langchain-google-vertexai
```

```python
from langchain_google_vertexai import ChatVertexAI
from rag7 import Agent

llm = ChatVertexAI(model="gemini-2.0-flash", project="my-gcp-project", location="us-central1")
rag = Agent(index="articles", llm=llm, gen_llm=llm)
```

### Separate fast and generation models

Use a cheap/fast model for query rewriting and routing, a powerful model for the final answer:

```python
from langchain_openai import AzureChatOpenAI
from rag7 import Agent

fast_llm = AzureChatOpenAI(azure_deployment="gpt-5.4-mini", api_key="...", api_version="2024-12-01-preview")
gen_llm  = AzureChatOpenAI(azure_deployment="gpt-5.4",      api_key="...", api_version="2024-12-01-preview")

rag = Agent(index="articles", llm=fast_llm, gen_llm=gen_llm)
```

---

## 🏆 Rerankers

### 🏅 Cohere

```python
from rag7 import Agent, CohereReranker

rag = Agent(index="articles", reranker=CohereReranker(model="rerank-v3.5", api_key="..."))
```

### 🤗 HuggingFace cross-encoder (local, no API key)

```bash
pip install rag7[huggingface]
```

```python
from rag7 import Agent, HuggingFaceReranker

rag = Agent(index="articles", reranker=HuggingFaceReranker())

# Multilingual
rag = Agent(index="articles", reranker=HuggingFaceReranker(model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
```

### 🌊 Jina (multilingual API)

```bash
pip install rag7[jina]
```

```python
from rag7 import Agent, JinaReranker

rag = Agent(index="articles", reranker=JinaReranker(api_key="..."))  # or JINA_API_KEY env var
```

### 🎯 rerankers — ColBERT / Flashrank / RankGPT / any cross-encoder

Unified bridge to the [`rerankers`](https://github.com/AnswerDotAI/rerankers) library by answer.ai:

```bash
pip install rag7[rerankers]
```

```python
from rag7 import Agent, RerankersReranker

rag = Agent(index="articles", reranker=RerankersReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", model_type="cross-encoder"))
rag = Agent(index="articles", reranker=RerankersReranker("colbert-ir/colbertv2.0", model_type="colbert"))
rag = Agent(index="articles", reranker=RerankersReranker("flashrank", model_type="flashrank"))
rag = Agent(index="articles", reranker=RerankersReranker("gpt-5.4-mini", model_type="rankgpt", api_key="..."))
```

### 🦀 Embed-anything — Rust-accelerated local embeddings + reranking

*"I never leave home without my gadgets, Q."* — Embeddings and reranking in a single Rust-powered package. No API keys, no network calls, no asking permission. The silent operative in your stack. Powered by [embed-anything](https://github.com/StarlightSearch/EmbedAnything).

```bash
pip install rag7[embed-anything]
```

```python
from rag7 import Agent, EmbedAnythingEmbedder, EmbedAnythingReranker

# Local embeddings — works as embed_fn (callable)
embedder = EmbedAnythingEmbedder("sentence-transformers/all-MiniLM-L6-v2")

# Local reranker — implements Reranker protocol
reranker = EmbedAnythingReranker("jinaai/jina-reranker-v1-turbo-en")

rag = Agent(
    index="articles",
    backend=QdrantBackend("articles", url="http://localhost:6333", embed_fn=embedder),
    embed_fn=embedder,
    reranker=reranker,
)
```

Mix and match freely — use embed-anything for one piece and a cloud provider for the other:

```python
from rag7 import Agent, EmbedAnythingEmbedder, CohereReranker

# Local embeddings + cloud reranker
rag = Agent(index="docs", embed_fn=EmbedAnythingEmbedder(), reranker=CohereReranker())

# Cloud embeddings + local reranker
from rag7 import EmbedAnythingReranker
rag = Agent(index="docs", embed_fn=azure_embed_fn, reranker=EmbedAnythingReranker())
```

### 🔧 Custom reranker

```python
from rag7 import Agent, RerankResult

class MyReranker:
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        return [RerankResult(index=i, relevance_score=1.0 / (i + 1)) for i in range(top_n)]

rag = Agent(index="articles", reranker=MyReranker())
```

---

## 🛠️ Tools

When using `invoke_agent`, the LLM has access to a set of tools it can call in any order. No fixed pipeline — the agent decides what it needs.

| Tool | Description |
|------|-------------|
| `get_index_settings()` | Discover filterable, searchable, sortable, and boost fields from the index schema |
| `get_filter_values(field)` | Sample real stored values for a field — used to build precise filter expressions |
| `search_hybrid(query, filter_expr, semantic_ratio, sort_fields)` | BM25 + vector hybrid search with optional filter and sort boost |
| `search_bm25(query, filter_expr)` | Pure keyword search — fallback when hybrid returns poor results |
| `rerank_results(query, hits)` | Re-rank a list of hits with the configured reranker |

The agent follows this reasoning pattern:

1. Call `get_index_settings()` to learn the schema
2. If the question names a specific entity, call `get_filter_values(field)` to find the exact stored value
3. Call `search_hybrid()` with a filter and/or sort if relevant, otherwise broad hybrid search
4. Fall back to `search_bm25()` if results are thin
5. Call `rerank_results()` to surface the most relevant hits
6. Summarise — explaining which filters and signals influenced the answer

```python
from rag7 import Agent

rag = Agent(index="products")

# Agent inspects schema, detects brand field, samples values,
# builds filter, sorts by popularity signal — all autonomously
result = rag.invoke_agent("Show me the most popular Bosch power tools")
print(result)
```

---

## ⚙️ Constructor Reference

```python
Agent(
    index="my_index",           # collection / index name
    backend=...,                # SearchBackend (default: InMemoryBackend)
    llm=...,                    # fast LLM — routing, rewrite, filter
    gen_llm=...,                # generation LLM — final answer
    reranker=...,               # Cohere / HuggingFace / Jina / custom
    top_k=10,                   # final result count            [RAG_TOP_K]
    rerank_top_n=5,             # reranker top-n                [RAG_RERANK_TOP_N]
    retrieval_factor=4,         # over-retrieval multiplier     [RAG_RETRIEVAL_FACTOR]
    max_iter=20,                # max retrieve-rewrite cycles   [RAG_MAX_ITER]
    semantic_ratio=0.5,         # hybrid semantic weight        [RAG_SEMANTIC_RATIO]
    fusion="rrf",               # "rrf" or "dbsf"               [RAG_FUSION]
    instructions="",            # extra system prompt for generation
    embed_fn=None,              # (str) -> list[float]
    boost_fn=None,              # (doc_dict) -> float score boost
    filter=None,                # always-on Meilisearch filter expr (e.g. "brand = 'Bosch'")
    category_fields=None,       # fields used by alternative retrieve (None → auto-detect via regex)
    hyde_min_words=8,           # min words to trigger HyDE     [RAG_HYDE_MIN_WORDS]
    hyde_style_hint="",         # style hint for HyDE prompt
    auto_strategy=True,         # auto-tune from index samples
)
```

### 🔒 Always-on filter

Pin every search to a subset of the index with `filter` — Meilisearch syntax,
AND-joined with any per-call filter (intent, language, ...):

```python
rag = init_agent("products", filter="brand = 'Bosch'")
# every BM25 + vector + hybrid search scoped to Bosch only
```

The legacy `base_filter` kwarg still works but emits a `DeprecationWarning` —
migrate to `filter` at your convenience.

### 🏷️ Category fields (alternative retrieve)

The alternative-retrieve fallback broadens the search by pivoting on
category-like fields (product groups, taxonomy levels, sections, ...). By
default, rag7 auto-detects them from the index schema via regex — matching
names like `category`, `product_group_l3`, `article_group_name`, `kategorie`,
`family`, `section`, ... — and prioritises deeper taxonomy levels
(`_l3 > _l2 > _l1`).

Override explicitly when your schema uses unusual names:

```python
rag = init_agent(
    "products",
    category_fields=["taxonomy_leaf", "taxonomy_parent", "department"],
)
```

Pass `category_fields=[]` to disable the fallback entirely.

---

## 📡 API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `rag.invoke(query)` | `RAGState` | Full RAG pipeline (sync) |
| `rag.ainvoke(query)` | `RAGState` | Full RAG pipeline (async) |
| `rag.chat(query, history)` | `RAGState` | Multi-turn chat (sync) |
| `rag.achat(query, history)` | `RAGState` | Multi-turn chat (async) |
| `rag.retrieve_documents(query, top_k)` | `(str, list[Document])` | Retrieve only, no answer |
| `rag.query(query)` | `str` | Answer string directly |
| `rag.invoke_agent(query)` | `str` | Tool-calling agent mode (sync) |
| `rag.ainvoke_agent(query)` | `str` | Tool-calling agent mode (async) |

`RAGState` fields: `answer` · `documents` · `query` · `question` · `history` · `iterations`

---

## 🌍 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | — |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | — |
| `AZURE_OPENAI_DEPLOYMENT` | Default deployment | — |
| `AZURE_OPENAI_FAST_DEPLOYMENT` | Fast model deployment | → `DEPLOYMENT` |
| `AZURE_OPENAI_GENERATION_DEPLOYMENT` | Generation deployment | → `DEPLOYMENT` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-12-01-preview` |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | — |
| `OPENAI_MODEL` | OpenAI model name | `gpt-5.4` |
| `AZURE_COHERE_ENDPOINT` | Azure Cohere endpoint | — |
| `AZURE_COHERE_API_KEY` | Azure Cohere API key | — |
| `COHERE_API_KEY` | Cohere API key (fallback) | — |
| `JINA_API_KEY` | Jina reranker API key | — |
| `MEILI_URL` | Meilisearch URL | `http://localhost:7700` |
| `MEILI_KEY` | Meilisearch API key | `masterKey` |
| `RAG_TOP_K` | Final result count | `10` |
| `RAG_RERANK_TOP_N` | Reranker top-n | `5` |
| `RAG_RETRIEVAL_FACTOR` | Over-retrieval multiplier | `4` |
| `RAG_SEMANTIC_RATIO` | Hybrid semantic weight | `0.5` |
| `RAG_FUSION` | Fusion strategy | `rrf` |
| `RAG_HYDE_MIN_WORDS` | Min words to trigger HyDE | `8` |

---

## 🎯 Tune It For Your Data

*"Q-Branch fits the gadget to the agent, not the other way around."*

**`rag7` ships with curated tuned defaults** in `[tool.rag7]` of `pyproject.toml`,
found by running the built-in tuner against real German/Swiss product catalog
data. These are better than hand-picked defaults for most retrieval tasks.

For peak performance on **your** corpus (product vs. legal vs. support vs.
scientific), run the tuner yourself — it searches the config space with
[Optuna](https://optuna.org) and usually beats defaults by 5–15% on hit@5.

### 1. Install

```bash
pip install 'rag7[tune]'
```

### 2. Write a testset

A list of `(query, expected_doc_ids, id_field)` tuples — or a JSON file:

```json
[
  {"query": "Makita Akku Bohrhammer 18V", "expected_ids": ["SKU-1065144"], "id_field": "sku"},
  {"query": "Bosch Winkelschleifer 125mm", "expected_ids": ["SKU-1057802"], "id_field": "sku"}
]
```

### 3. Run the tuner

```python
from rag7 import MeilisearchBackend, RAGConfig
from rag7.tuner import RAGTuner, load_testset
from rag7.utils import _make_azure_embed_fn

tuner = RAGTuner(
    backend_factory=lambda: MeilisearchBackend(index="my_index"),
    embed_fn=_make_azure_embed_fn(),
    hit_cases=load_testset("testset.json"),
    eval_k=5,
)

best = tuner.tune(n_trials=50)
best.save_toml("rag7.config.toml")   # gitignored — local override (recommended)
# or: best.save_pyproject()          # [tool.rag7] — commit if your team shares tuning
```

### 4. Use the tuned config

No code change required — `AgenticRAG` picks up `[tool.rag7]` automatically:

```python
from rag7 import AgenticRAG, RAGConfig, MeilisearchBackend

rag = AgenticRAG(
    index="my_index",
    backend=MeilisearchBackend("my_index"),
    embed_fn=embed_fn,
    config=RAGConfig.auto(),   # discovers pyproject.toml → rag7.config.toml → env
)
```

### Config discovery order

1. **Runtime kwarg** — `AgenticRAG(config=RAGConfig(...))`
2. **`[tool.rag7]` in `pyproject.toml`** — matches ruff/black/mypy convention.
   `rag7` ships with curated defaults here; override or delete the block to
   fall through to your own tuning.
3. **`rag7.config.toml`** — per-deployment tuning. **Gitignored by default**
   because values here are corpus-specific. This is the recommended place for
   your own tuned config unless your whole team uses the same data.
4. **`RAG_*` env vars** — containers/CI overrides.
5. **Library defaults** — fallback if nothing else is set.

### What gets tuned

| Parameter | Range | Effect |
|-----------|-------|--------|
| `retrieval_factor` | 2–8 | Over-retrieve multiplier before reranking |
| `rerank_top_n` | 3–10 | Docs kept post-rerank |
| `rerank_cap_multiplier` | 1.5–4 | Caps reranker input at `top_k × m` |
| `semantic_ratio` | 0.3–0.9 | BM25 ⇄ vector balance |
| `fusion` | `rrf` / `dbsf` | Score fusion strategy |
| `short_query_threshold` | 3–8 | When to skip LLM preprocessing |
| `short_query_sort_tokens` | bool | Sort tokens for paraphrase invariance |
| `bm25_fallback_threshold` | 0.2–0.6 | When weak BM25 triggers semantic boost |
| `bm25_fallback_semantic_ratio` | 0.7–1.0 | Boost target ratio |
| `expert_threshold` | 0.05–0.3 | Expert reranker escalation |

Optuna's TPE sampler learns from prior trials — **50 trials usually beats
hand-tuning**. Use a cheap LLM (e.g. `gpt-4o-mini`) during tuning to keep cost
down; swap to your production LLM afterwards.

---

## 🖥️ CLI

*"The gadgets are ready."*

```bash
pip install rag7[recommended]

# 🧙 Guided setup wizard — choose LLM, embedder, backend, reranker
rag7

# 💬 Chat mode — full agentic pipeline
rag7 --chat -c my_index

# 🔍 Retriever mode — documents only, no LLM
rag7 --retriever -c my_index

# ⚡ Skip wizard, use env vars
rag7 --skip-wizard -c my_index
```

The wizard guides you through:
1. **LLM provider** — OpenAI, Anthropic, Ollama, or env default
2. **Embedding model** — OpenAI, Azure OpenAI, Ollama, or none (BM25 only)
3. **Vector store** — InMemory, Meilisearch, ChromaDB, Qdrant, pgvector, DuckDB, LanceDB, Azure AI Search
4. **Reranker** — Cohere, Jina, HuggingFace, LLM-based, or none
5. **Mode** — Chat (with answers) or Retriever (documents only)

---

## 📄 License

MIT — *Licence to code.*
