# Examples

Ten ready-to-run patterns covering the most common rag7 use cases.

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

rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
                 backend_url="http://localhost:6333")

query, docs = rag.retrieve_documents("What is retrieval-augmented generation?")
for doc in docs:
    print(doc.page_content[:100])
```

---

### 3. Multi-turn chat

```python
from rag7 import init_agent, ConversationTurn

rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
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
    model="openai:gpt-4o",
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
    model="openai:gpt-4o",
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
    rag = init_agent("kb", model="openai:gpt-4o", backend="qdrant",
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

rag = init_agent("products", model="openai:gpt-4o", backend="meilisearch")

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
    rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
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
    model="openai:gpt-4o",
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
