# Filtering

rag7 supports metadata filters on every search request. Filters use a Meilisearch-style string syntax, automatically translated to each backend's native dialect.

## Syntax

```python
# Equality
'brand = "Bosch"'
'in_stock = true'
'price < 100'

# Inequality
'category != "archived"'

# AND
'in_stock = true AND price < 100'

# NOT CONTAINS (substring exclusion)
'NOT supplier CONTAINS "house-brand"'

# Chained
'category = "tools" AND brand != "generic" AND price < 50'
```

## Always-on filter

Pass `filter=` to `Agent` or `init_agent` to apply the same filter to every query — BM25, vector, and hybrid:

```python
from rag7 import init_agent

rag = init_agent(
    "products",
    model="openai:gpt-5.4",
    backend="meilisearch",
    filter="in_stock = true",
)
```

Per-query filters (auto-detected via the tool-calling agent) are AND-joined with this base filter.

## Per-query filter

```python
from rag7.backend import SearchRequest

state = rag.invoke("brake cleaner", filter_expr='brand = "WD-40"')
```

## Backend filter dialects

| Backend | Equality | NOT CONTAINS | AND |
|---------|----------|-------------|-----|
| Meilisearch | `field = "v"` | `NOT field CONTAINS "v"` | `A AND B` |
| LanceDB | `field = 'v'` | `field NOT ILIKE '%v%'` | `A AND B` |
| DuckDB | `field = 'v'` | `field NOT ILIKE '%v%'` | `A AND B` |
| pgvector | `field = 'v'` | `field NOT ILIKE '%v%'` | `A AND B` |
| Qdrant | `FieldCondition(MatchValue)` | `FieldCondition(MatchText)` in must_not | native dict |
| ChromaDB | `{"field": "v"}` | _(dropped — no substring op)_ | `{"$and": [...]}` |
| Azure AI Search | OData `field eq 'v'` | `not search.ismatch(...)` | `A and B` |

rag7's filter translator handles all dialect differences automatically. You write Meilisearch syntax; rag7 converts it for the active backend.

!!! note "ChromaDB limitation"
    ChromaDB has no metadata substring operation. `NOT CONTAINS` clauses are silently dropped rather than raising an error, so queries still run — just without that constraint.
