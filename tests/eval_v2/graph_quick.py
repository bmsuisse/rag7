"""Quick eval using full graph pipeline (includes grader retry cycles).

Unlike quick.py (which calls _aretrieve_documents directly), this runs
rag.ainvoke to exercise the entire graph including final_grade reruns.
Use this to measure the benefit of enable_final_grade.

Usage: uv run python -m tests.eval_v2.graph_quick [--verbose]
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

from tests.eval_v2.adversarial import HIT_CASE, build_adversarial_cases
from tests.eval_v2.quick import MINI_HITS

_TOTAL_TIMEOUT_S: float = 600.0
_QUERY_TIMEOUT_S: float = 120.0

_ENABLE_FINAL_GRADE: bool = True
_FINAL_GRADE_THRESHOLD: float = 0.9


async def _build_rag(index: str) -> Any:
    from rag7 import AgenticRAG, RAGConfig
    from rag7.backend import MeilisearchBackend
    from rag7.utils import _make_azure_embed_fn

    backend = MeilisearchBackend(index=index)
    cfg = backend.get_index_config()
    dims = set(cfg.embedder_dims.values())
    target_dim = next(iter(dims)) if len(dims) == 1 else None
    config = RAGConfig.auto().model_copy(
        update={
            "thinking_model": "azure:brain",
            "strong_model": "azure:brain",
            "grader_model": "azure:brain",
            "enable_final_grade": _ENABLE_FINAL_GRADE,
            "final_grade_threshold": _FINAL_GRADE_THRESHOLD,
        }
    )
    return AgenticRAG(
        index=index,
        backend=backend,
        embed_fn=_make_azure_embed_fn(dimensions=target_dim),
        config=config,
        auto_strategy=True,
    )


async def main(verbose: bool = False) -> None:
    rag = await _build_rag("onetrade_articles_de")
    adversarial = build_adversarial_cases(MINI_HITS)
    sem = asyncio.Semaphore(10)

    async def _one(case: HIT_CASE) -> tuple[bool, list[str]]:
        query, expected, id_field = case
        async with sem:
            try:
                state = await asyncio.wait_for(
                    rag.ainvoke(query),
                    timeout=_QUERY_TIMEOUT_S,
                )
                retrieved = [
                    str(d.metadata.get(id_field, "")) for d in state.documents[:5]
                ]
                return any(str(e) in retrieved for e in expected), retrieved
            except asyncio.TimeoutError:
                if verbose:
                    print(f"TIMEOUT: {query!r}", flush=True)
                return False, []
            except Exception as e:
                if verbose:
                    print(f"ERROR: {query!r}: {type(e).__name__}", flush=True)
                return False, []

    try:
        async with asyncio.timeout(_TOTAL_TIMEOUT_S):
            t0 = time.perf_counter()
            results = await asyncio.gather(*(_one(c) for c in adversarial))
            elapsed = time.perf_counter() - t0
    except asyncio.TimeoutError:
        print(f"\ngraph eval: TIMED OUT after {_TOTAL_TIMEOUT_S:.0f}s")
        return

    hits = sum(hit for hit, _ in results)
    total = len(adversarial)

    if verbose:
        for (q, expected, _), (hit, retrieved) in zip(adversarial, results):
            status = "✓" if hit else "✗"
            print(f"{status} {q!r}")
            if not hit:
                print(f"    expected: {expected}  got: {retrieved[:3]}")

    print(f"\ngraph eval: {hits}/{total} = {hits / total:.4f}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    import sys
    asyncio.run(main(verbose="--verbose" in sys.argv or "-v" in sys.argv))
