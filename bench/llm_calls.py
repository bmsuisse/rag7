"""Speed bench: counts LLM calls + measures wall-time over a fixed query set.

Simulates real LLM latency via ``asyncio.sleep(PER_CALL_S)`` so wall-ms tracks
what a real cloud LLM would contribute. Backend is InMemoryBackend over fixture
docs — no network, no embeddings.

Run:
  .venv/bin/python bench/llm_calls.py

Prints:
  calls=<int>
  wall_ms=<float>       serial for-loop (per-query latency)
  wall_ms_batch=<float> abatch (concurrent throughput)
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

# Simulated latency per LLM call — mimics fast cloud model.
PER_CALL_S = 0.05  # 50ms
N_QUERIES = 20

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

# Each query gets N matching docs so retrieval returns a full top_k window
# (simulates realistic corpus where many docs match a keyword-ish query).
_MATCHES_PER_QUERY = 15
DOCS = [
    {
        "id": f"q{i}_{j}",
        "content": f"Document {j}: {q} detailed guide and examples with context.",
    }
    for i, q in enumerate(QUERIES)
    for j in range(_MATCHES_PER_QUERY)
]


class _CallCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.by_tag: dict[str, int] = {}

    async def tick(self, tag: str = "raw") -> None:
        self.calls += 1
        self.by_tag[tag] = self.by_tag.get(tag, 0) + 1
        await asyncio.sleep(PER_CALL_S)

    def tick_sync(self, tag: str = "raw") -> None:
        self.calls += 1
        self.by_tag[tag] = self.by_tag.get(tag, 0) + 1
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
        def __init__(self, value, tag: str) -> None:
            self._v = value
            self._tag = tag

        def invoke(self, _):
            counter.tick_sync(self._tag)
            return self._v

        async def ainvoke(self, _):
            await counter.tick(self._tag)
            return self._v

    llm = MagicMock()

    def _with_structured_output(model, **_kw):
        return _Chain(defaults[model], tag=model.__name__)

    llm.with_structured_output.side_effect = _with_structured_output

    async def _ainvoke(_):
        await counter.tick("gen")
        return MagicMock(content="answer.")

    def _invoke(_):
        counter.tick_sync("gen")
        return MagicMock(content="answer.")

    llm.ainvoke = _ainvoke
    llm.invoke = _invoke
    return llm


async def _run() -> tuple[int, float, float, dict[str, int]]:
    counter = _CallCounter()
    stub = _build_stub_llm(counter)
    rag = AgenticRAG(
        index="bench",
        backend=InMemoryBackend(documents=DOCS),
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )
    t0 = time.perf_counter()
    for q in QUERIES:
        await rag.ainvoke(q)
    wall_serial = (time.perf_counter() - t0) * 1000
    calls_after_serial = counter.calls

    t1 = time.perf_counter()
    await rag.abatch(QUERIES)
    wall_batch = (time.perf_counter() - t1) * 1000

    return calls_after_serial, wall_serial, wall_batch, counter.by_tag


def main() -> None:
    calls, wall, wall_batch, by_tag = asyncio.run(_run())
    print(f"calls={calls}")
    print(f"wall_ms={wall:.2f}")
    print(f"wall_ms_batch={wall_batch:.2f}")
    for tag, n in sorted(by_tag.items(), key=lambda kv: -kv[1]):
        print(f"  by_tag.{tag}={n}")


if __name__ == "__main__":
    main()
