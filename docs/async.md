# Async & Sync

Every rag7 operation has both a sync and an async variant. The sync variants work even inside a running event loop (Databricks, Jupyter, FastAPI startup).

## Sync usage

```python
from rag7 import init_agent

rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
                 backend_url="http://localhost:6333")

state = rag.invoke("What is hybrid search?")
print(state.answer)
```

## Async usage

```python
import asyncio
from rag7 import init_agent

async def main():
    rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
                     backend_url="http://localhost:6333")

    state = await rag.ainvoke("What is hybrid search?")
    print(state.answer)

asyncio.run(main())
```

## FastAPI

```python
from fastapi import FastAPI
from rag7 import init_agent

app = FastAPI()
rag = init_agent("docs", model="openai:gpt-4o", backend="qdrant",
                 backend_url="http://localhost:6333")

@app.post("/ask")
async def ask(question: str):
    state = await rag.ainvoke(question)
    return {"answer": state.answer, "sources": len(state.documents)}
```

## Running sync from a running loop

rag7's sync wrappers use an internal `_run_sync` helper that spawns a new thread if an event loop is already running — so `rag.invoke()` works from Jupyter notebooks and Databricks cells without modification.

```python
# Works in Jupyter / Databricks — no asyncio.run() needed
state = rag.invoke("What is hybrid search?")
```

## Streaming

```python
async for chunk in rag.astream("Explain retrieval-augmented generation"):
    print(chunk, end="", flush=True)
```
