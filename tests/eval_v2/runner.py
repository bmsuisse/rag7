"""Runner for eval_v2: adversarial, paraphrase-consistency, and synthetic suites.

Metrics:
- `hit@5`: was any expected ID in top-5?
- `consistency`: across variants of the same seed query, fraction of variants
  that retrieved at least one expected ID. 1.0 = fully robust.
- `stable_top1`: fraction of variant groups where top-1 ID is the same as
  the seed query's top-1. Measures ranking stability under perturbation.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HIT_CASE = tuple[str, list[str], str]

_DATA_DIR = Path(__file__).parent


async def _retrieve_ids(rag: Any, query: str, id_field: str, k: int = 5) -> list[str]:
    _, docs = await rag._aretrieve_documents(query, top_k=k)
    return [str(d.metadata.get(id_field, "")) for d in docs]


async def run_hits(
    rag: Any,
    cases: Sequence[HIT_CASE],
    sem: asyncio.Semaphore,
    label: str = "eval_v2",
) -> tuple[int, int]:
    """Plain hit@5 over a list of HIT_CASE tuples."""
    hits = 0
    total = len(cases)

    async def _one(case: HIT_CASE) -> bool:
        query, expected_ids, id_field = case
        async with sem:
            retrieved = await _retrieve_ids(rag, query, id_field)
        return any(str(e) in retrieved for e in expected_ids)

    results = await asyncio.gather(*(_one(c) for c in cases))
    hits = sum(results)
    print(
        f"[{label}] hit@5 = {hits}/{total} = {hits / total:.4f}"
        if total
        else f"[{label}] no cases",
        flush=True,
    )
    return hits, total


async def run_consistency(
    rag: Any,
    groups: Sequence[tuple[HIT_CASE, Sequence[str]]],
    sem: asyncio.Semaphore,
    label: str = "consistency",
) -> tuple[float, float, int]:
    """For each (seed, [variants]), measure consistency + stable_top1.

    Returns (avg_consistency, stable_top1_rate, group_count).
    """

    async def _top_ids(query: str, field: str) -> list[str]:
        async with sem:
            return await _retrieve_ids(rag, query, field)

    consistencies: list[float] = []
    stable = 0
    total = 0

    for (seed_query, expected, field), variants in groups:
        seed_ids = await _top_ids(seed_query, field)
        seed_top1 = seed_ids[0] if seed_ids else None
        variant_tops: list[list[str]] = await asyncio.gather(
            *(_top_ids(v, field) for v in variants)
        )
        if not variants:
            continue
        total += 1
        expected_set = {str(e) for e in expected}
        hit_count = sum(
            1 for ids in variant_tops if any(i in expected_set for i in ids)
        )
        consistencies.append(hit_count / len(variants))
        if seed_top1 and all(ids and ids[0] == seed_top1 for ids in variant_tops):
            stable += 1

    avg = sum(consistencies) / len(consistencies) if consistencies else 0.0
    stable_rate = stable / total if total else 0.0
    print(
        f"[{label}] consistency={avg:.4f}  stable_top1={stable_rate:.4f}  "
        f"groups={total}",
        flush=True,
    )
    return avg, stable_rate, total


def load_json_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def paraphrase_groups(
    base: Sequence[HIT_CASE],
    paraphrase_path: Path = _DATA_DIR / "paraphrases.json",
) -> list[tuple[HIT_CASE, list[str]]]:
    """Pair each base case with its LLM-generated paraphrase variants."""
    entries = load_json_cases(paraphrase_path)
    by_seed = {e["seed"]: e.get("variants", []) for e in entries}
    return [(case, by_seed.get(case[0], [])) for case in base if by_seed.get(case[0])]


def synthetic_cases(
    path: Path = _DATA_DIR / "synthetic.json",
) -> list[HIT_CASE]:
    """Load LLM-generated (query → doc_id) pairs."""
    entries = load_json_cases(path)
    return [(e["query"], [str(e["doc_id"])], e.get("id_field", "id")) for e in entries]
