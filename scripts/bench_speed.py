"""Speed benchmark for AgenticRAG end-to-end latency.

Populates a Meilisearch index with fake product data (idempotent — skips if already
populated), then runs a fixed set of queries through `rag.ainvoke` and reports
mean wall-clock latency plus a simple accuracy count.

Output (stdout, last line):
    mean_s=<float>  hits=<hit>/<total>

Requires: Meilisearch at localhost:7700 (key=masterKey), AZURE_OPENAI_* env.

Usage:
    uv run python scripts/bench_speed.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Force local meili regardless of .env (cloud MEILI_URL would be used otherwise).
MEILI_URL = "http://localhost:7700"
MEILI_KEY = "masterKey"
os.environ["MEILI_URL"] = MEILI_URL
os.environ["MEILI_KEY"] = MEILI_KEY

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import meilisearch  # noqa: E402

from rag7 import AgenticRAG, MeilisearchBackend, _make_azure_embed_fn  # noqa: E402

INDEX = "bench_fake_products"

# (query, needle, should_hit)
# should_hit=True  → needle must appear in top-10 docs
# should_hit=False → needle must NOT appear (negative/no-hit case)
QUERIES: list[tuple[str, str, bool]] = [
    ("UltraWidget Pro features", "ultrawidget", True),
    ("Apple products in Electronics category", "apple", True),
    ("cheap office supplies under 50 euros", "office supplies", True),
    ("Samsung networking gear", "samsung", True),
    ("enterprise storage solutions", "storage", True),
    ("Logitech peripherals", "logitech", True),
    ("Siemens software product", "siemens", True),
    ("HP servers for data center", "hp", True),
    ("Sony electronics new release", "sony", True),
    ("Microsoft software license", "microsoft", True),
    ("Dell laptops available in stock", "dell", True),
    ("Bosch industrial networking equipment", "bosch", True),
    ("Lenovo peripherals for office", "lenovo", True),
    ("product with SKU COF-1381", "cof-1381", True),
    ("product with SKU XYY-0307", "xyy-0307", True),
    # negative cases — these brands/skus don't exist in the data
    ("Zephyrox quantum drive lineup", "zephyrox", False),
    ("product with SKU ZZZ-9999", "zzz-9999", False),
    ("Nabucoo enterprise software", "nabucoo", False),
]


def _ensure_index() -> None:
    client = meilisearch.Client(MEILI_URL, MEILI_KEY)
    try:
        stats = client.index(INDEX).get_stats()
        if stats.number_of_documents >= 5000:
            return
    except Exception:
        pass

    products = json.loads((REPO_ROOT / "tests" / "fake_products.json").read_text())
    confusion_path = REPO_ROOT / "tests" / "fake_confusion_products.json"
    if confusion_path.exists():
        products += json.loads(confusion_path.read_text())

    try:
        client.index(INDEX).delete()
        time.sleep(0.5)
    except Exception:
        pass

    task = client.create_index(INDEX, {"primaryKey": "id"})
    client.wait_for_task(task.task_uid)
    idx = client.index(INDEX)
    for update, args in [
        (
            idx.update_filterable_attributes,
            ["category", "supplier", "language", "in_stock", "price", "tier"],
        ),
        (idx.update_searchable_attributes, ["content", "name", "description", "sku"]),
        (idx.update_sortable_attributes, ["price", "released_at"]),
    ]:
        client.wait_for_task(update(args).task_uid)

    for i in range(0, len(products), 1000):
        batch = [
            {k: v for k, v in d.items() if k != "embedding"}
            for d in products[i : i + 1000]
        ]
        client.wait_for_task(idx.add_documents(batch).task_uid)

    print(f"populated {INDEX} with {len(products)} docs", file=sys.stderr)


async def _run() -> None:
    _ensure_index()
    rag = AgenticRAG(
        INDEX,
        backend=MeilisearchBackend(INDEX),
        embed_fn=_make_azure_embed_fn(),
        embedder_name="azure_openai",
        auto_strategy=True,
    )

    async def _one(q: str, needle: str, should_hit: bool) -> tuple[float, bool, bool]:
        t0 = time.perf_counter()
        state = await rag.ainvoke(q)
        elapsed = time.perf_counter() - t0
        docs = (
            state.get("documents", []) if isinstance(state, dict) else state.documents
        )
        blob = " ".join(
            str(d.metadata) + " " + (d.page_content or "") for d in docs[:10]
        ).lower()
        found = needle.lower() in blob
        correct = found == should_hit
        return elapsed, correct, should_hit

    results = await asyncio.gather(*(_one(q, n, h) for q, n, h in QUERIES))
    times = [r[0] for r in results]
    pos_correct = sum(1 for _, ok, h in results if h and ok)
    neg_correct = sum(1 for _, ok, h in results if not h and ok)
    pos_total = sum(1 for _, _, h in results if h)
    neg_total = sum(1 for _, _, h in results if not h)
    total_correct = pos_correct + neg_correct
    mean_s = sum(times) / len(times)
    print(f"per_query={[round(t, 3) for t in times]}")
    print(
        f"mean_s={mean_s:.3f}  "
        f"hits={pos_correct}/{pos_total}  "
        f"rejects={neg_correct}/{neg_total}  "
        f"total={total_correct}/{len(QUERIES)}"
    )


if __name__ == "__main__":
    asyncio.run(_run())
