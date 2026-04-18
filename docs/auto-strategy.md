# Auto Strategy

`auto_strategy=True` (the default) makes rag7 self-configure at initialisation time by sampling a few documents from your index and asking the LLM to infer optimal settings.

## What it does

On the first `invoke` / `chat`, rag7 samples up to 5 documents from your backend and asks the LLM to infer:

| Setting | What it controls |
|---------|-----------------|
| `hyde_style_hint` | Short phrase injected into the HyDE prompt to match your document style (e.g. `"German product spec sheet"`) |
| `hyde_min_words` | Minimum query word count to trigger HyDE expansion |
| `domain_hint` | Domain description passed to the preprocessor to improve keyword extraction |

This is a **one-time LLM call** at startup — zero per-query overhead.

## Usage

```python
from rag7 import init_agent

# auto_strategy is True by default
rag = init_agent("docs", model="openai:gpt-5.4", backend="qdrant",
                 backend_url="http://localhost:6333")
```

Disable for full manual control or in tests with stub LLMs:

```python
rag = init_agent("docs", auto_strategy=False,
                 hyde_style_hint="technical product datasheet",
                 hyde_min_words=6)
```

## Manual override

You can set the same values explicitly without auto-detection:

```python
from rag7 import Agent, MeilisearchBackend

rag = Agent(
    index="docs",
    backend=MeilisearchBackend("docs"),
    auto_strategy=False,
    hyde_style_hint="legal contract document",
    hyde_min_words=5,
)
```

## HyDE

Hypothetical Document Embeddings: when a query is long enough (≥ `hyde_min_words` words), rag7 generates a *hypothetical* answer document and embeds it instead of the raw query. This improves vector recall for vague or descriptive questions where the query text is semantically distant from the documents.

HyDE runs in **parallel** with preprocessing — no sequential latency penalty.
