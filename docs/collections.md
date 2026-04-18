# Multi-Collection Routing

When your data lives in multiple indexes, rag7 can route each query to the right subset automatically. Before retrieval, an LLM routing step selects which collections are relevant — only those are searched.

## Setup

=== "List (route by name)"
    ```python
    from rag7 import init_agent

    rag = init_agent(
        collections=["products", "faq", "policies"],
        backend="qdrant",
        backend_url="http://localhost:6333",
        model="openai:gpt-5.4",
    )
    ```

=== "Dict (route with descriptions)"
    ```python
    from rag7 import init_agent

    rag = init_agent(
        collections={
            "products": "Product catalog: SKUs, prices, specs, availability",
            "faq":      "Customer-facing FAQ, troubleshooting, return policy",
            "policies": "Internal HR and compliance policy documents",
        },
        backend="qdrant",
        backend_url="http://localhost:6333",
        model="openai:gpt-5.4",
    )
    ```

=== "Manual (pre-built backends)"
    ```python
    from rag7 import Agent
    from rag7.backend import MeilisearchBackend

    backends = {
        "products": MeilisearchBackend("products"),
        "manuals":  MeilisearchBackend("manuals"),
    }
    rag = Agent(
        index="catalog",
        collections=backends,
        collection_descriptions={
            "products": "Product listings with prices and specs",
            "manuals":  "Installation and user guides",
        },
    )
    ```

## How it works

1. The query arrives at `invoke` / `chat`.
2. An LLM call selects the relevant collection names (using names + optional descriptions).
3. Only the selected backends are searched — the context variable `_ACTIVE_COLLECTIONS` scopes retrieval.
4. Each retrieved document gets a `metadata["_collection"]` tag with its source collection.

If the LLM returns an empty selection (uncertain), rag7 falls back to searching all collections.

## Collection metadata

```python
state = rag.invoke("What's our return policy?")
for doc in state.documents:
    print(doc.metadata["_collection"], doc.page_content[:80])
```

## Accessing the collection map

```python
print(rag.collections.keys())   # dict[str, SearchBackend]
```
