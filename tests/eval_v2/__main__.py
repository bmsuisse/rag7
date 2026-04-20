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

ONETRADE_DE_HITS: list[HIT_CASE] = [
    ("Makita Akku Bohrhammer 18V", ["1065144", "8170146", "1059195"], "article_id"),
    ("Bosch Winkelschleifer 125mm", ["1057802", "1058233", "1075261"], "article_id"),
    ("Schutzhelm Bauhelm", ["1054731", "1060660", "9137783"], "article_id"),
    ("Hilti Anker Bolzen M12", ["6150515", "1143910"], "article_id"),
    # Regression guard: supplier-filter + structured-attribute lookup.
    # "bieröffner von proone" must surface the ProOne Multi-Tool (which
    # lists "Flaschenöffner" in akeneo_values) — not a random ProOne
    # bestseller like Trockenbeton. Tests the quality-gate score floor
    # + _hit_to_text structured-field expansion together.
    ("bieröffner von proone", ["1050541", "7794905"], "article_id"),
    # Natural-language variant: the user's real wording, tests that the
    # brand-filter + variants + own-brand OR filter together reach a doc
    # whose searchable name/search_term contains neither "bieröffner" nor
    # "flaschenöffner" (the latter sits in akeneo_values).
    (
        "haben wir bieröffner von proone?",
        ["1050541", "7794905"],
        "article_id",
    ),
    (
        "proone tool mit flaschenöffner",
        ["1050541", "7794905"],
        "article_id",
    ),
    # Natural-language queries — prose phrasing users actually type.
    ("Akkuschrauber von Bosch", ["6260639", "5377125"], "article_id"),
    ("Winkelschleifer Makita 125mm", ["1056160", "1056161"], "article_id"),
    ("Schutzhelm in weiss", ["9137783", "1055431"], "article_id"),
    ("Taschenmesser Victorinox", ["1143505"], "article_id"),
    ("Stiel für einen Hammer", ["1055521", "1048808"], "article_id"),
    ("Silikon schwarz Kartusche", ["1006780", "9251987"], "article_id"),
    ("LED Taschenlampe mit USB", ["7779803"], "article_id"),
    ("Werkzeugkoffer Aluminium", ["1147106", "7984433"], "article_id"),
    ("Rohrzange Rothenberger", ["6204301", "1149577"], "article_id"),
    # ProOne own-brand catalog coverage (pro-one.ch products).
    ("ProOne Schuhreiniger Edi", ["9225388"], "article_id"),
    ("ProOne Baustellenradio Rock One", ["9183348"], "article_id"),
    ("ProOne Kartuschenpistole 225mm", ["1050095"], "article_id"),
    ("ProOne Silikon Sanitär transparent", ["1003220"], "article_id"),
    ("ProOne Elastisches Maleracryl weiss", ["8172012"], "article_id"),
    ("ProOne Montageschaum 1K", ["8171999"], "article_id"),
    # Natural-language synonym/brand-filter stress cases — these tend to
    # need the LLM variants + brand filter to resolve.
    ("Schaumpistole von ProOne", ["1050511", "7504843"], "article_id"),
    ("Schuhputzer proone", ["9225388"], "article_id"),
]

ARTICLE_HITS: list[HIT_CASE] = [
    ("Wedi Bauplatte 10mm", ["1003118", "01509094", "01509098"], "id"),
    ("Sand gewaschen 0-4mm", ["01561902", "01580072"], "id"),
]

SUPPLIER_CATALOGS_DE_HITS: list[HIT_CASE] = [
    (
        "ACO Drain Rinne Monoblock",
        ["-1284896587609186235", "7647252346056341609"],
        "id",
    ),
    ("Entwässerung Ablauf", ["6470805727571075019", "-1284896587609186235"], "id"),
]

SUITES: list[tuple[str, str, list[HIT_CASE]]] = [
    ("onetrade_articles_de", "OneTrade DE Articles", ONETRADE_DE_HITS),
    ("article", "Article Catalog", ARTICLE_HITS),
    ("supplier_catalogs_de", "Supplier Catalogs DE", SUPPLIER_CATALOGS_DE_HITS),
]
# ─────────────────────────────────────────────────────────────────────────────


async def _build_rag(index: str) -> Any:
    from rag7 import AgenticRAG, RAGConfig
    from rag7.backend import MeilisearchBackend
    from rag7.utils import _make_azure_embed_fn

    # Match the client embedder dimension to whatever the index declares, so
    # text-embedding-3-small returns native 512-d (or whatever is needed)
    # instead of getting sliced on the client. Server-side truncation is
    # cheaper in bandwidth and avoids the lossy L2-renorm path.
    backend = MeilisearchBackend(index=index)
    cfg = backend.get_index_config()
    dims = set(cfg.embedder_dims.values())
    target_dim = next(iter(dims)) if len(dims) == 1 else None

    return AgenticRAG(
        index=index,
        backend=backend,
        embed_fn=_make_azure_embed_fn(dimensions=target_dim),
        config=RAGConfig.auto(),
        auto_strategy=True,
    )


async def main() -> None:
    if not SUITES:
        print("No eval suites configured. Edit SUITES in tests/eval_v2/__main__.py.")
        return

    sem = asyncio.Semaphore(20)

    t0 = time.perf_counter()
    grand_hits = 0
    grand_total = 0
    consistency_sum = 0.0
    consistency_groups = 0
    stable_sum = 0.0

    for index, label, base in SUITES:
        rag = await _build_rag(index)

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
    avg_consistency = (
        consistency_sum / consistency_groups if consistency_groups else 0.0
    )
    avg_stable = stable_sum / consistency_groups if consistency_groups else 0.0

    avg_query_latency_ms = elapsed / grand_total * 1000 if grand_total else 0.0
    print("\n════════════════════════════════════════════════════════════")
    print(f"eval_v2 summary: {elapsed:.1f}s total")
    if grand_total:
        print(
            f"  adversarial+synthetic hit@5: {grand_hits}/{grand_total} = "
            f"{grand_hits / grand_total:.4f}"
        )
    print(f"  paraphrase consistency:      {avg_consistency:.4f}")
    print(f"  paraphrase stable_top1:      {avg_stable:.4f}")
    print(f"  avg query latency:           {avg_query_latency_ms:.0f}ms")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
