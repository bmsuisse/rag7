"""Fixed 20-query latency benchmark for autoresearch speed loop.

Usage: uv run python tests/bench_latency.py
Outputs: avg_latency_ms=XXX  (lower is better)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fixed representative mix: short/long, filter-word/no-filter, ProOne, natural-language
QUERIES = [
    # Short / likely fast-accept
    "Wandklosett",
    "Trockenbeton",
    "Gipsplatte",
    "Dusch WC",
    "Spiegelschrank Bad",
    # Long without filter word (hit relevance_check path)
    "Bosch Winkelschleifer 125mm",
    "Makita Flex 125mm",
    "freistehende Acryl Badewanne 170cm",
    "Duschwand Walk-In Glas 150",
    "Kreissägeblatt Bosch 190mm",
    # Filter-word queries (brand filter path)
    "Akkuschrauber von Bosch",
    "Makita Akku Bohrhammer 18V",
    "bieröffner von proone",
    "Eckrohrzange Rothenberger",
    "Klosettsitz Laufen",
    # Natural language
    "ich brauche einen 18V Bohrhammer von Makita",
    "gibt es Schuhreiniger bei euch?",
    "welches Radio gibt es von ProOne?",
    "einen Hammer mit kurzem Stiel",
    "Stiel für einen Hammer",
]


async def _build_rag():
    from rag7 import AgenticRAG, RAGConfig
    from rag7.backend import MeilisearchBackend
    from rag7.utils import _make_azure_embed_fn

    backend = MeilisearchBackend(index="onetrade_articles_de")
    cfg = backend.get_index_config()
    dims = set(cfg.embedder_dims.values())
    target_dim = next(iter(dims)) if len(dims) == 1 else None
    return AgenticRAG(
        index="onetrade_articles_de",
        backend=backend,
        embed_fn=_make_azure_embed_fn(dimensions=target_dim),
        config=RAGConfig.auto(),
        auto_strategy=True,
    )


async def main() -> None:
    rag = await _build_rag()
    latencies: list[float] = []
    for q in QUERIES:
        t0 = time.perf_counter()
        await rag._aretrieve_documents(q, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    print(f"avg_latency_ms={avg:.0f}")
    print(f"p50_latency_ms={p50:.0f}")
    print(f"p90_latency_ms={p90:.0f}")
    for q, ms in zip(QUERIES, latencies):
        print(f"  {ms:6.0f}ms  {q}")


if __name__ == "__main__":
    asyncio.run(main())
