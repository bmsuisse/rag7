---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# 🕵️ rag7

**Not just hybrid search. A true autonomous retrieval agent.**

Shaken, not stirred — plug in any vector store, any LLM, any reranker.  
The mission: find the right documents, neutralise irrelevant noise, and deliver the answer. Every time.

<div class="hero-buttons">
  <a href="quickstart/" class="btn-primary">Get Started</a>
  <a href="https://github.com/bmsuisse/rag7" class="btn-secondary">View on GitHub</a>
  <a href="https://pypi.org/project/rag7/" class="btn-secondary">PyPI</a>
</div>

</div>

---

```python
from rag7 import init_agent

rag = init_agent("documents", model="openai:gpt-4o", backend="qdrant")
state = rag.chat("What is the status of operation overlord?")
print(state.answer)  # Cited. Grounded. Delivered.
```

---

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### 🔄 True Agentic Loop

Retrieve → judge → rewrite → retry. Fully autonomous up to `max_iter` rounds. Never satisfied with *good enough*.

</div>

<div class="feature-card" markdown>

### 🔍 Hybrid Search

BM25 + vector search fused with Reciprocal Rank Fusion or Distribution-Based Score Fusion. Multi-query variants in parallel.

</div>

<div class="feature-card" markdown>

### 🗄️ 8 Backends

Meilisearch, Azure AI Search, ChromaDB, LanceDB, Qdrant, pgvector, DuckDB, InMemory. Swap with one line.

</div>

<div class="feature-card" markdown>

### 🏆 Multi-Reranker

Cohere, HuggingFace, Jina, ColBERT, RankGPT, embed-anything, or any custom reranker.

</div>

<div class="feature-card" markdown>

### 🧠 HyDE

Hypothetical Document Embeddings boost recall on vague or descriptive queries automatically.

</div>

<div class="feature-card" markdown>

### 🎯 Auto Strategy

LLM samples your collection at init, tunes `hyde_style_hint`, `semantic_ratio`, and domain hints — zero per-query overhead.

</div>

<div class="feature-card" markdown>

### 🛠️ Tool-Calling Agent

`invoke_agent` gives the LLM a toolset: inspect schema, sample field values, build filters on the fly, boost by business signals.

</div>

<div class="feature-card" markdown>

### 📂 Multi-Collection

Route queries to the right subset of collections automatically. LLM picks which indexes to search per request.

</div>

</div>

---

## Install

```bash
pip install rag7[recommended]   # Meilisearch + Cohere + CLI
pip install rag7                # Base only — InMemory backend
pip install rag7[all]           # Everything
```

See [Quick Start](quickstart.md) for all install options and a 5-minute walkthrough.
