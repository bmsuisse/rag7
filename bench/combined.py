"""Combined speed bench: LLM calls + retrieval work + wall-time.

Bigger corpus (3000 docs) so Python retrieval/fusion/rerank paths matter
alongside simulated LLM latency. Reports wall-ms, LLM calls, and a
retrieval-only sub-timer to isolate backend-code cost.

Run:
  .venv/bin/python bench/combined.py

Prints:
  calls=<int>
  wall_ms=<float>
  retrieval_ms=<float>
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from rag7 import AgenticRAG, InMemoryBackend
from rag7.models import (
    CollectionIntent,
    FilterIntent,
    MultiQuery,
    QualityAssessment,
    ReasoningVerdict,
    RelevanceCheck,
    SearchQuery,
)

PER_CALL_S = 0.05
N_DOCS_PER_QUERY = 150  # 20q × 150 = 3000 docs

QUERIES = [
    "how does python handle memory",
    "rust vs go performance",
    "explain vector search",
    "what is retrieval augmented generation",
    "fix ssl handshake error",
    "postgresql index types",
    "kubernetes liveness probe",
    "how to debug segfault",
    "what is bm25 scoring",
    "dockerfile best practices",
    "typescript generics",
    "what is a monad",
    "http 2 vs http 3",
    "react hooks useEffect",
    "design database schema",
    "async await python vs javascript",
    "rate limiting strategies",
    "oauth2 vs oidc",
    "tls 1.3 improvements",
    "nginx reverse proxy setup",
]

DOCS = [
    {
        "id": f"q{i}_{j}",
        "content": (
            f"Document {j} for {q}: detailed guide with examples, code snippets, "
            f"common pitfalls, and references. Additional context about {q} "
            f"including best practices and related topics like scaling, debugging, "
            f"and production deployment scenarios. Section {j % 5}."
        ),
    }
    for i, q in enumerate(QUERIES)
    for j in range(N_DOCS_PER_QUERY)
]


class _CallCounter:
    def __init__(self) -> None:
        self.calls = 0

    async def tick(self) -> None:
        self.calls += 1
        await asyncio.sleep(PER_CALL_S)

    def tick_sync(self) -> None:
        self.calls += 1
        time.sleep(PER_CALL_S)


def _build_stub_llm(counter: _CallCounter):
    defaults = {
        SearchQuery: SearchQuery(
            query="", variants=[], semantic_ratio=0.5, fusion="rrf"
        ),
        QualityAssessment: QualityAssessment(sufficient=True, reason=""),
        MultiQuery: MultiQuery(queries=[]),
        FilterIntent: FilterIntent(field=None, value="", operator=""),
        CollectionIntent: CollectionIntent(collections=[]),
        RelevanceCheck: RelevanceCheck(makes_sense=True, confidence=0.9),
        ReasoningVerdict: ReasoningVerdict(),
    }

    class _Chain:
        def __init__(self, value) -> None:
            self._v = value

        def invoke(self, _):
            counter.tick_sync()
            return self._v

        async def ainvoke(self, _):
            await counter.tick()
            return self._v

    llm = MagicMock()
    llm.with_structured_output.side_effect = lambda m, **_kw: _Chain(defaults[m])

    async def _ainvoke(_):
        await counter.tick()
        return MagicMock(content="answer.")

    def _invoke(_):
        counter.tick_sync()
        return MagicMock(content="answer.")

    llm.ainvoke = _ainvoke
    llm.invoke = _invoke
    return llm


async def _run() -> tuple[int, float, float]:
    counter = _CallCounter()
    stub = _build_stub_llm(counter)
    backend = InMemoryBackend(documents=DOCS)

    # Wrap backend.search to measure retrieval-only time
    retrieval_ns = 0

    orig = backend.search

    def _timed_search(req):
        nonlocal retrieval_ns
        t = time.perf_counter_ns()
        out = orig(req)
        retrieval_ns += time.perf_counter_ns() - t
        return out

    backend.search = _timed_search  # type: ignore[method-assign]

    rag = AgenticRAG(
        index="bench",
        backend=backend,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )
    t0 = time.perf_counter()
    for q in QUERIES:
        await rag.ainvoke(q)
    wall = (time.perf_counter() - t0) * 1000
    return counter.calls, wall, retrieval_ns / 1e6


def main() -> None:
    calls, wall, retrieval_ms = asyncio.run(_run())
    print(f"calls={calls}")
    print(f"wall_ms={wall:.2f}")
    print(f"retrieval_ms={retrieval_ms:.2f}")


if __name__ == "__main__":
    main()
