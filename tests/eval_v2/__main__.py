"""Run the eval_v2 suite against your configured indexes.

Configure SUITES below to point to your own Meilisearch indexes and hit cases.
Hit cases are 3-tuples: (query, [expected_ids], id_field).

Usage: uv run python -m tests.eval_v2
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tests.eval_v2.adversarial import HIT_CASE, build_adversarial_cases
from tests.eval_v2.runner import (
    paraphrase_groups,
    run_consistency,
    run_hits,
    synthetic_cases,
)

# ── Configure your eval suites here ──────────────────────────────────────────
# Each entry: (index_name, label, hit_cases)
# hit_cases: list of (query, [expected_ids], id_field) tuples
#
# Example:
#   MY_HITS: list[HIT_CASE] = [
#       ("Bosch SDS Plus Bohrer 10mm", ["ARTICLE-123", "ARTICLE-456"], "id"),
#       ("Hilti Anchor Bolt M12", ["ARTICLE-789"], "id"),
#   ]
#   SUITES = [("my_meili_index", "My Product Catalog", MY_HITS)]

SUITES: list[tuple[str, str, list[HIT_CASE]]] = []
# ─────────────────────────────────────────────────────────────────────────────


async def _build_rag(index: str, embed_fn: Any) -> Any:
    from rag7 import AgenticRAG
    from rag7.backend import MeilisearchBackend

    return AgenticRAG(
        index=index,
        backend=MeilisearchBackend(index=index),
        embed_fn=embed_fn,
        auto_strategy=True,
    )


async def main() -> None:
    if not SUITES:
        print("No eval suites configured. Edit SUITES in tests/eval_v2/__main__.py.")
        return

    from rag7.utils import _make_azure_embed_fn

    embed_fn = _make_azure_embed_fn()
    sem = asyncio.Semaphore(20)

    t0 = time.perf_counter()
    grand_hits = 0
    grand_total = 0
    consistency_sum = 0.0
    consistency_groups = 0
    stable_sum = 0.0

    for index, label, base in SUITES:
        rag = await _build_rag(index, embed_fn)

        adversarial = build_adversarial_cases(base)
        print(f"\n══ {label}: adversarial ({len(adversarial)} variants) ══")
        hits, total = await run_hits(rag, adversarial, sem, label=f"{label}/adv")
        grand_hits += hits
        grand_total += total

        groups = paraphrase_groups(base)
        if groups:
            print(f"══ {label}: paraphrase consistency ({len(groups)} groups) ══")
            avg, stable, n = await run_consistency(
                rag, groups, sem, label=f"{label}/para"
            )
            consistency_sum += avg * n
            consistency_groups += n
            stable_sum += stable * n

        syn = synthetic_cases()
        if syn:
            print(f"══ {label}: synthetic ({len(syn)} queries) ══")
            h, t = await run_hits(rag, syn, sem, label=f"{label}/syn")
            grand_hits += h
            grand_total += t

    elapsed = time.perf_counter() - t0
    avg_consistency = consistency_sum / consistency_groups if consistency_groups else 0.0
    avg_stable = stable_sum / consistency_groups if consistency_groups else 0.0

    print("\n════════════════════════════════════════════════════════════")
    print(f"eval_v2 summary: {elapsed:.1f}s")
    if grand_total:
        print(f"  adversarial+synthetic hit@5: {grand_hits}/{grand_total} = "
              f"{grand_hits / grand_total:.4f}")
    print(f"  paraphrase consistency:      {avg_consistency:.4f}")
    print(f"  paraphrase stable_top1:      {avg_stable:.4f}")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
