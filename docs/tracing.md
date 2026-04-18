# Tracing & Verbose

Every `invoke` / `chat` call returns a `RAGState` object that includes full pipeline metadata — which path was taken, timings per node, how many iterations ran, and whether the quality gate passed.

!!! info "It's just LangGraph"
    rag7 is built on [LangGraph](https://github.com/langchain-ai/langgraph). `RAGState` is a standard LangGraph state — the raw graph output is returned as-is, so you have full access to everything LangGraph tracks. You can also access `rag.graph` directly to inspect, extend, or wire the graph into a larger LangGraph application.

    ```python
    # Access the underlying LangGraph graph
    graph = rag.graph

    # Run it directly with full LangGraph control
    result = await graph.ainvoke({"question": "What is RAG?"})
    ```

## State fields

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Generated answer |
| `documents` | `list[Document]` | Retrieved + reranked sources |
| `query` | `str` | Final (possibly rewritten) query |
| `query_variants` | `list[str]` | All query variants used in search |
| `iterations` | `int` | Number of retrieve-rewrite cycles |
| `quality_ok` | `bool` | Whether the quality gate passed |
| `selected_collections` | `list[str]` | Collections searched (multi-collection mode) |
| `adaptive_semantic_ratio` | `float \| None` | Semantic/BM25 ratio chosen by preprocessor |
| `adaptive_fusion` | `str \| None` | Fusion strategy used (`"rrf"` or `"dbsf"`) |
| `expert_fired` | `bool` | Whether the expert reranker ran |
| `trace` | `list[dict]` | Per-node trace with timings and path info |

## Verbose mode

Enable `verbose=True` to print each pipeline step to stdout as it runs:

```python
from rag7 import init_agent

rag = init_agent(
    "my-index",
    model="openai:gpt-5.4",
    backend="meilisearch",
    reranker="cohere",
    verbose=True,
)

state = rag.invoke("What is hybrid search?")
# stdout: [preprocess] 0.31s path=full query=...
#         [parallel_start] 0.00s docs=40
#         [rerank] 0.44s docs=10 expert_fired=False
#         [quality_gate] 0.18s path=llm ok=True
#         [generate] 1.20s docs=10
```

Or via env var: `RAG_VERBOSE=1`.

## Reading the trace

`state.trace` is always populated — no extra config needed:

```python
state = rag.invoke("What is hybrid search?")

for step in state.trace:
    print(step)
```

Example output:

```python
{'node': 'preprocess',    'dur_s': 0.31, 'path': 'full',  'query': 'hybrid search BM25 vector'}
{'node': 'parallel_start','dur_s': 0.00, 'docs': 40}
{'node': 'rerank',        'dur_s': 0.44, 'docs': 10,      'expert_fired': False}
{'node': 'quality_gate',  'dur_s': 0.18, 'path': 'llm',   'ok': True}
{'node': 'generate',      'dur_s': 1.20, 'docs': 10}
```

## Full example

```python
from rag7 import init_agent

rag = init_agent("my-index", model="openai:gpt-5.4", backend="meilisearch")

state = rag.invoke("What is hybrid search?")

print(state.answer)
print()
print(f"query rewritten : {state.query}")
print(f"variants used   : {state.query_variants}")
print(f"iterations      : {state.iterations}")
print(f"quality ok      : {state.quality_ok}")
print(f"semantic ratio  : {state.adaptive_semantic_ratio}")
print(f"fusion          : {state.adaptive_fusion}")
print(f"expert fired    : {state.expert_fired}")
print(f"collections     : {state.selected_collections}")
print()
print("── trace ──")
for step in state.trace:
    node = step["node"]
    dur  = step["dur_s"]
    rest = {k: v for k, v in step.items() if k not in ("node", "dur_s")}
    print(f"  [{node}] {dur}s  {rest}")
```

## Sending trace to your backend

`state.trace` is a plain list of dicts — forward it to any observability tool:

=== "OpenTelemetry"
    ```python
    from opentelemetry import trace as otel_trace

    tracer = otel_trace.get_tracer("rag7")

    with tracer.start_as_current_span("rag7.invoke") as span:
        state = rag.invoke(question)
        for step in state.trace:
            span.add_event(step["node"], attributes=step)
    ```

=== "Datadog / structured log"
    ```python
    import json, logging
    log = logging.getLogger("rag7")

    state = rag.invoke(question)
    log.info("rag7.invoke", extra={
        "question": question,
        "iterations": state.iterations,
        "quality_ok": state.quality_ok,
        "trace": state.trace,
    })
    ```

=== "Simple dict response"
    ```python
    state = rag.invoke(question)

    return {
        "answer": state.answer,
        "sources": [d.metadata for d in state.documents],
        "meta": {
            "query": state.query,
            "iterations": state.iterations,
            "quality_ok": state.quality_ok,
            "trace": state.trace,
        },
    }
    ```
