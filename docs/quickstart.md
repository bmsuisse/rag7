# Quick Start

## Install

```bash
# Recommended — Meilisearch + Cohere reranker + interactive CLI
pip install rag7[recommended]

# Base only — in-memory backend, no external services
pip install rag7
```

??? "🍸 Bond Edition extras"
    | Extra | Code name | Stack |
    |-------|-----------|-------|
    | `goldeneye` | GoldenEye | Meilisearch + Cohere + CLI |
    | `skyfall` | Skyfall | Everything — all backends, all rerankers |
    | `thunderball` | Thunderball | Qdrant + Cohere + CLI |
    | `moonraker` | Moonraker | ChromaDB + HuggingFace — fully local |
    | `goldfinger` | Goldfinger | Azure AI Search + Cohere |
    | `spectre` | Spectre | pgvector + HuggingFace — no paid APIs |
    | `casino-royale` | Casino Royale | ChromaDB + Jina — lightweight |
    | `licence-to-kill` | Licence to Kill | embed-anything + ChromaDB — Rust-powered |

    ```bash
    pip install rag7[goldeneye]
    pip install rag7[skyfall]
    pip install rag7[moonraker]
    ```

??? "Individual backends & rerankers"
    ```bash
    pip install rag7[meilisearch]
    pip install rag7[azure]
    pip install rag7[chromadb]
    pip install rag7[lancedb]
    pip install rag7[pgvector]
    pip install rag7[qdrant]
    pip install rag7[duckdb]
    pip install rag7[cohere]
    pip install rag7[huggingface]
    pip install rag7[jina]
    pip install rag7[rerankers]
    pip install rag7[embed-anything]
    ```
    Mix freely: `pip install rag7[qdrant,cohere,cli]`

---

## One-liner with `init_agent`

The fastest path — string aliases for everything, no imports:

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

# Fully local — Ollama + ChromaDB + HuggingFace
rag = init_agent(
    "docs",
    model="ollama:llama3",
    backend="chroma",
    reranker="huggingface",
)

# Anthropic + Azure AI Search (server-side vectorisation)
rag = init_agent(
    "my-index",
    model="anthropic:claude-sonnet-4-6",
    gen_model="anthropic:claude-opus-4-7",
    backend="azure",
    backend_url="https://my-search.search.windows.net",
)
```

**Model strings** follow `"provider:model-name"` — `openai`, `anthropic`, `azure_openai`, `google_vertexai`, `ollama`, `groq`, `mistralai`, and any other LangChain provider.

---

## Your first query

```python
from rag7 import init_agent

rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant",
                 backend_url="http://localhost:6333")

# Full agentic answer
state = rag.invoke("What is hybrid search?")
print(state.answer)
print(f"Sources: {len(state.documents)}")

# Multi-turn chat
from rag7 import ConversationTurn
history: list[ConversationTurn] = []

state = rag.chat("What is hybrid search?", history)
history.append(ConversationTurn(question="What is hybrid search?", answer=state.answer))
state = rag.chat("How does it compare to pure vector search?", history)
```

---

## Manual setup

For full control over the backend instance:

```python
from rag7 import Agent, InMemoryBackend

backend = InMemoryBackend(embed_fn=my_embed_fn)
backend.add_documents([
    {"content": "RAG combines retrieval with generation", "source": "wiki"},
    {"content": "Vector search finds similar embeddings", "source": "docs"},
])

rag = Agent(index="demo", backend=backend)

state = rag.invoke("What is retrieval-augmented generation?")
print(state.answer)
```

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint (auto-detected) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name |
| `COHERE_API_KEY` | Cohere reranker key |
| `JINA_API_KEY` | Jina reranker key |

Set in a `.env` file — rag7 loads it automatically via `python-dotenv` if installed.

---

## Next steps

- [Backends](backends.md) — configure each backend
- [Reranking](reranking.md) — choose and tune rerankers
- [Filtering](filtering.md) — narrow results with metadata filters
- [Multi-Collection](collections.md) — route queries across multiple indexes
- [Examples](examples.md) — 10 ready-to-run patterns
