# Backends

rag7 supports 8 backends. Swap with a single string alias in `init_agent`, or pass a backend instance directly to `Agent`.

## Backend aliases

| Alias | Class | Install |
|-------|-------|---------|
| `"memory"` / `"in_memory"` | `InMemoryBackend` | _(built-in)_ |
| `"meilisearch"` | `MeilisearchBackend` | `rag7[meilisearch]` |
| `"azure"` | `AzureAISearchBackend` | `rag7[azure]` |
| `"chroma"` / `"chromadb"` | `ChromaDBBackend` | `rag7[chromadb]` |
| `"lancedb"` / `"lance"` | `LanceDBBackend` | `rag7[lancedb]` |
| `"qdrant"` | `QdrantBackend` | `rag7[qdrant]` |
| `"pgvector"` / `"pg"` | `PgvectorBackend` | `rag7[pgvector]` |
| `"duckdb"` | `DuckDBBackend` | `rag7[duckdb]` |

---

## InMemory

Zero dependencies. Good for testing, prototyping, and small datasets.

```python
from rag7 import Agent, InMemoryBackend

backend = InMemoryBackend(embed_fn=my_embed_fn)
backend.add_documents([{"content": "...", "id": "1"}])
rag = Agent(index="docs", backend=backend)
```

Or via `init_agent`:

```python
rag = init_agent("docs", backend="memory", embed_fn=my_embed_fn)
```

---

## Meilisearch

Full-text BM25 search with vector hybrid mode. Requires `pip install rag7[meilisearch]`.

```python
from rag7 import init_agent

rag = init_agent(
    "my-index",
    backend="meilisearch",
    backend_url="http://localhost:7700",   # default
    backend_kwargs={"api_key": "masterKey"},
    embed_fn=my_embed_fn,
    model="openai:gpt-5.4",
)
```

Or directly:

```python
from rag7.backend import MeilisearchBackend

backend = MeilisearchBackend(
    index="my-index",
    url="http://localhost:7700",
    api_key="masterKey",
    embed_fn=my_embed_fn,
)
```

---

## Azure AI Search

Supports native server-side vectorisation — no client-side `embed_fn` needed when an integrated vectorizer is configured.

```python
from rag7 import init_agent

rag = init_agent(
    "my-index",
    backend="azure",
    backend_url="https://my-search.search.windows.net",
    backend_kwargs={"api_key": "your-admin-key"},
    model="openai:gpt-5.4",
)
```

Credentials can also come from environment variables (`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`) or managed identity via `azure-identity`.

---

## ChromaDB

Local or remote. In-memory by default for testing.

```python
from rag7 import init_agent

rag = init_agent(
    "my-collection",
    backend="chroma",
    embed_fn=my_embed_fn,
    model="openai:gpt-5.4",
)
```

Persistent storage:

```python
from rag7.backend import ChromaDBBackend

backend = ChromaDBBackend(
    collection="my-collection",
    embed_fn=my_embed_fn,
    client_settings={"chroma_db_impl": "duckdb+parquet", "persist_directory": "./chroma"},
)
```

---

## LanceDB

Columnar vector store. Fast for large datasets, no server required.

```python
from rag7.backend import LanceDBBackend

backend = LanceDBBackend(
    table="docs",
    db_uri="./lancedb",
    embed_fn=my_embed_fn,
    vector_column="vector",
)
```

---

## Qdrant

High-performance vector database. Supports both `:memory:` and server modes.

```python
from rag7 import init_agent

rag = init_agent(
    "my-collection",
    backend="qdrant",
    backend_url="http://localhost:6333",
    embed_fn=my_embed_fn,
    model="openai:gpt-5.4",
)
```

In-memory (testing):

```python
from rag7.backend import QdrantBackend

backend = QdrantBackend("my-collection", location=":memory:", embed_fn=my_embed_fn)
```

---

## pgvector

PostgreSQL with the pgvector extension. Supports full SQL filtering via ILIKE.

```python
from rag7.backend import PgvectorBackend

backend = PgvectorBackend(
    table="documents",
    connection_string="postgresql://user:pass@localhost/mydb",
    embed_fn=my_embed_fn,
)
```

Docker Compose for local dev:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

---

## DuckDB

Embedded analytical database. No server, fast for local workloads.

```python
from rag7.backend import DuckDBBackend

backend = DuckDBBackend(
    table="docs",
    db_path="./mydb.duckdb",
    embed_fn=my_embed_fn,
)
```

---

## Custom backend

Implement `SearchBackend` from `rag7.backend`:

```python
from rag7.backend import SearchBackend, SearchRequest, IndexConfig

class MyBackend(SearchBackend):
    def search(self, req: SearchRequest) -> list[dict]:
        ...

    def get_index_config(self) -> IndexConfig:
        ...

    def sample_documents(self, limit: int = 5, ...) -> list[dict]:
        ...
```
