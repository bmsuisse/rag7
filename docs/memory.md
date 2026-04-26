# Memory & Persistence

rag7 is built on LangGraph and supports two levels of memory:

| | `checkpointer=` | `memory_store=` |
|--|----------------|-----------------|
| **Scope** | Per thread (conversation) | Per user (cross-thread) |
| **What's stored** | Full graph state | Key Q&A facts |
| **Survives restarts** | With SQLite/Postgres | With SQLite/Postgres |
| **Use case** | Resume a conversation | Remember user preferences across sessions |

For simple multi-turn chat within a single session, the `history=` parameter on `chat()` is enough — no config needed.

## In-process memory (MemorySaver)

Lost on restart. Good for single-session apps or testing.

```python
from rag7 import init_agent
from langgraph.checkpoint.memory import MemorySaver

rag = init_agent(
    "docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    backend_url="http://localhost:6333",
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "user-alice"}}

state = rag.invoke("What is hybrid search?", config=config)
state = rag.invoke("Give me an example.", config=config)   # graph remembers the first turn
```

## Persistent memory (SQLite)

Survives restarts. Good for chatbots and long-running apps.

```python
from rag7 import init_agent
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("./memory.db") as checkpointer:
    rag = init_agent(
        "docs",
        model="openai:gpt-5.4",
        backend="qdrant",
        backend_url="http://localhost:6333",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "user-alice"}}
    state = rag.invoke("What is hybrid search?", config=config)
```

## Persistent memory (PostgreSQL)

Production-grade. Requires `pip install langgraph-checkpoint-postgres`.

```python
from rag7 import init_agent
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/mydb"
)
checkpointer.setup()   # creates the checkpoint tables on first run

rag = init_agent(
    "docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "user-alice"}}
state = rag.invoke("What is hybrid search?", config=config)
```

## Thread IDs

Each `thread_id` is an independent memory scope. Use one per user, session, or conversation:

```python
# Different users — separate memory
rag.invoke("Who are you?", config={"configurable": {"thread_id": "user-alice"}})
rag.invoke("Who are you?", config={"configurable": {"thread_id": "user-bob"}})
```

---

## Smarter long-term memory (mem0)

[mem0](https://docs.mem0.ai) uses an LLM to extract discrete facts from
each exchange, deduplicate them, and resolve conflicts — so "I moved
to Berlin" replaces "I live in Munich" rather than creating two entries.
It is the recommended choice when memory quality matters.

### How rag7 wires it in

The LangGraph state machine has two memory nodes — `read_memory` runs
before retrieval, `write_memory` runs after generation. When you pass
`mem0_memory=` to the agent, rag7 routes both nodes through mem0:

| Node | mem0 call | What rag7 passes |
| ---- | --------- | ----------------- |
| `read_memory` | `search(question, filters={"user_id": ...})` | Top 5 hits become a context block prepended to retrieval. |
| `write_memory` | `add(messages, user_id=...)` after the answer | mem0's LLM extracts facts; conflicts get resolved. |

Both nodes read `user_id` from the request `config` (`config["configurable"]["user_id"]`).
If a request arrives without a `user_id`, rag7 falls back to `"default"`.
async-vs-sync detection happens automatically — pass `AsyncMemory()`
and rag7 uses `asearch`/`aadd` directly; pass `Memory()` and rag7
runs the sync calls in a thread pool so the event loop stays free.

### Install

```bash
pip install mem0ai
```

### Minimal example

```python
from rag7 import init_agent
from mem0 import Memory  # or AsyncMemory

rag = init_agent(
    "docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    backend_url="http://localhost:6333",
    mem0_memory=Memory(),
)

config = {"configurable": {"user_id": "alice"}}

rag.invoke("I prefer answers in German.", config=config)
rag.invoke("What is hybrid search?", config=config)
# mem0 extracted the language preference and recalled it on the second call.
```

### Configuring mem0's own backends

`Memory()` defaults to OpenAI embeddings + an embedded vector DB. To
point it at the same vector store you already use for retrieval, pass
mem0 a config dict:

```python
from mem0 import Memory

mem = Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333, "collection_name": "user_memories"},
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-5.4-mini"},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"},
    },
})

rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant", mem0_memory=mem)
```

See the [mem0 docs](https://docs.mem0.ai) for the full provider list
(Anthropic, Azure OpenAI, Postgres + pgvector, Chroma, etc.).

### Inspecting recalled memories

`state.trace` contains a `read_memory` entry whenever mem0 returned hits:

```python
state = rag.invoke("What is hybrid search?", config={"configurable": {"user_id": "alice"}})
for step in state.trace:
    if step["node"] == "read_memory":
        print("Recalled facts:\n", step["memories"])
```

### Manually scoping users

`user_id` partitions all memory ops. Two ways to set it:

```python
# 1. Via the per-call config (most common)
rag.invoke(question, config={"configurable": {"user_id": "alice"}})

# 2. Or bake a default into the agent and let mem0 see it
import functools
ainvoke = functools.partial(rag.ainvoke, config={"configurable": {"user_id": "alice"}})
```

### Failure modes & gotchas

- **No mem0 LLM** → mem0 still stores raw turns but skips fact
  extraction; quality matches `memory_store` (raw Q&A). Configure
  mem0's own LLM to get the deduplication benefit.
- **Slow first call** → mem0's first call seeds embeddings; expect
  ~1–2 s extra latency on cold start. Subsequent calls are fast.
- **Async event loop already running** (Jupyter, FastAPI handlers)
  → use `AsyncMemory()`. rag7 detects `aadd`/`asearch` and avoids
  the thread-pool round-trip.
- **mem0 errors are swallowed** — rag7's memory nodes catch
  exceptions silently so memory hiccups never break the QA path.
  Check `state.trace` for missing `read_memory` events when
  debugging.

!!! tip "Combining with checkpointer"
    `mem0_memory=` is orthogonal to `checkpointer=`. The checkpointer
    persists *graph state* per `thread_id` (resume a conversation);
    mem0 persists *extracted facts* per `user_id` (carry preferences
    across sessions). Pass both for full coverage.

---

## Long-term memory (memory_store)

Cross-thread memory that persists facts across different conversations and users. After each answer the agent writes a Q&A summary; before each retrieval it reads relevant past exchanges and uses them as context.

```python
from rag7 import init_agent
from langgraph.store.memory import InMemoryStore

rag = init_agent(
    "docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    backend_url="http://localhost:6333",
    memory_store=InMemoryStore(),
)

# Scope memories to a user with user_id
config = {"configurable": {"user_id": "alice"}}

rag.invoke("I prefer answers in German.", config=config)
rag.invoke("What is hybrid search?", config=config)
# Second call remembers the language preference from the first
```

Combine both for full memory:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

rag = init_agent(
    "docs",
    model="openai:gpt-5.4",
    backend="qdrant",
    checkpointer=SqliteSaver.from_conn_string("./memory.db"),
    memory_store=InMemoryStore(),
)

config = {"configurable": {"thread_id": "session-1", "user_id": "alice"}}
state = rag.invoke("What is hybrid search?", config=config)

# state.trace includes a 'read_memory' entry showing what was recalled
for step in state.trace:
    if step["node"] == "read_memory":
        print("Recalled:", step.get("memories"))
```

For production, replace `InMemoryStore` with `AsyncPostgresStore`:

```python
from langgraph.store.postgres import AsyncPostgresStore

store = AsyncPostgresStore.from_conn_string("postgresql://user:pass@localhost/mydb")
await store.setup()  # creates tables on first run

rag = init_agent("docs", model="openai:gpt-5.4", memory_store=store)
```

---

## When to use memory vs history

| | `history=` on `chat()` | `checkpointer=` | `memory_store=` | `mem0_memory=` |
|--|------------------------|-----------------|-----------------|----------------|
| Scope | Single session | Per thread | Per user (cross-thread) | Per user (cross-thread) |
| What's stored | Answer text | Full graph state | Raw Q&A strings | Extracted facts |
| Deduplication | — | — | No | Yes (LLM-based) |
| Survives restarts | No | With SQLite/Postgres | With Postgres store | With mem0 store |
| Use case | Simple multi-turn | Resumable chatbots | Basic long-term context | Smart user preferences |
| Config key | _(none)_ | `thread_id` | `user_id` | `user_id` |

!!! tip
    Combine all for full coverage: `history=` for the current turn, `checkpointer=` to resume the thread, `mem0_memory=` to recall extracted facts from previous sessions.
