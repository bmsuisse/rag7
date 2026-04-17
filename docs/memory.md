# Memory & Persistence

rag7 is built on LangGraph. Adding memory means passing a LangGraph **checkpointer** — the graph stores and resumes state per `thread_id` automatically.

You only need this when you want the graph itself to remember previous turns across separate `invoke` calls. For simple multi-turn chat within a single session, the `history` parameter on `chat()` is enough.

## In-process memory (MemorySaver)

Lost on restart. Good for single-session apps or testing.

```python
from rag7 import init_agent
from langgraph.checkpoint.memory import MemorySaver

rag = init_agent(
    "docs",
    model="openai:gpt-4o",
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
        model="openai:gpt-4o",
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
    model="openai:gpt-4o",
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

## When to use memory vs history

| | `history=` on `chat()` | `checkpointer=` |
|--|------------------------|-----------------|
| How it works | You manage the list | LangGraph manages state |
| Persistence | Only within your process | Survives restarts (SQLite/Postgres) |
| Use case | Simple multi-turn in one session | Chatbots, resumable conversations |
| Setup | No config needed | Pass checkpointer + thread_id |

!!! tip
    Under the hood `chat()` with `history=` is rag7's own lightweight memory. `checkpointer=` hands memory to LangGraph directly — the full graph state (including trace, iterations, quality signals) is checkpointed, not just the answer text.
