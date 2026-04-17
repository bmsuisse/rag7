# Reranking

Reranking re-scores the initial retrieval candidates with a more powerful model before passing them to the LLM. rag7 fetches `top_k × retrieval_factor` documents, reranks them, and surfaces the top `rerank_top_n`.

## Reranker aliases

| Alias | Class | Default model | Install |
|-------|-------|---------------|---------|
| `"cohere"` | `CohereReranker` | `rerank-v3.5` | `rag7[cohere]` |
| `"huggingface"` / `"hf"` | `HuggingFaceReranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rag7[huggingface]` |
| `"jina"` | `JinaReranker` | `jina-reranker-v2-base-multilingual` | `rag7[jina]` |
| `"llm"` | `LLMReranker` | _(agent's LLM)_ | _(built-in)_ |
| `"rerankers"` | `RerankersReranker` | ColBERT, Flashrank, RankGPT… | `rag7[rerankers]` |
| `"embed-anything"` | `EmbedAnythingReranker` | `jina-reranker-v1-turbo-en` | `rag7[embed-anything]` |

---

## Usage

```python
from rag7 import init_agent

# Cohere
rag = init_agent("docs", model="openai:gpt-4o", reranker="cohere")

# HuggingFace — runs locally, no API key
rag = init_agent("docs", model="openai:gpt-4o", reranker="huggingface")

# HuggingFace with a multilingual model
rag = init_agent("docs", model="openai:gpt-4o", reranker="huggingface",
                 reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Jina (uses JINA_API_KEY)
rag = init_agent("docs", model="openai:gpt-4o", reranker="jina")

# ColBERT via the rerankers library
rag = init_agent("docs", model="openai:gpt-4o", reranker="rerankers",
                 reranker_model="colbert-ir/colbertv2.0",
                 reranker_kwargs={"model_type": "colbert"})

# Pre-built instance
from rag7 import CohereReranker
rag = init_agent("docs", reranker=CohereReranker(model="rerank-v3.5", api_key="..."))
```

---

## Tuning

```python
rag = init_agent(
    "docs",
    model="openai:gpt-4o",
    reranker="cohere",
    top_k=10,              # final result count returned to the LLM
    rerank_top_n=5,        # how many of top_k the reranker sees
    retrieval_factor=4,    # fetch top_k × retrieval_factor candidates first
)
```

With these defaults rag7 fetches 40 candidates (10 × 4), reranks all 40, then passes the top 5 to the quality gate and LLM. Increase `retrieval_factor` for better recall at the cost of reranker latency.

---

## Expert reranker

A second, more expensive reranker run on a smaller candidate set for high-stakes queries:

```python
from rag7 import Agent, CohereReranker
from rag7.rerankers import HuggingFaceReranker

rag = Agent(
    index="docs",
    reranker=HuggingFaceReranker(),      # fast first pass
    expert_reranker=CohereReranker(),    # precision second pass
    expert_top_n=3,                      # only top-3 go to expert
    expert_threshold=0.7,                # only if first-pass confidence < 0.7
)
```
