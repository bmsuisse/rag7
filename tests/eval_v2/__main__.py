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
    EXCLUSION_CASE,
    paraphrase_groups,
    run_consistency,
    run_exclusion_hits,
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
    # ── Boost tests ───────────────────────────────────────────────────────────
    # Generic category queries: sales_l12m ranking decides the winner.
    # Expected IDs ordered by descending sales — passes only if boosting
    # correctly surfaces high-sales items into top-5.
    #
    # Trockenbeton: ProOne 016 (5.6M) > ProOne 008 (3.3M) > Fixit (1.2M).
    # Also tests is_own_brand boost stacking on top of sales signal.
    ("Trockenbeton", ["4457227", "4477141"], "article_id"),
    # Wandklosett: Laufen Moderna R UP (14.2M) and Moderna R AP (6.6M).
    ("Wandklosett", ["5414193", "6993231"], "article_id"),
    # Geberit Spülkasten: AP128 (10.9M) clear category winner.
    ("Geberit Spülkasten", ["7501080", "6944630"], "article_id"),
    # Gipsplatte: Rigips RB-Vario (7.2M) and Knauf GKB (5.3M).
    ("Gipsplatte", ["1002586", "1002844"], "article_id"),
    # Dusch-WC: Geberit Aquaclean Mera Classic (17.8M) first.
    ("Dusch WC", ["7501020", "6190872"], "article_id"),
    # Badewanne Stahl: Procasa Uno 170x70 (2.4M) — own-brand steel bathtub.
    ("Badewanne Stahl", ["01802416", "01802418"], "article_id"),
    # Klebeband Beton: ProOne Betonband (1.6M) — own-brand dominates.
    ("Klebeband Beton", ["7882736"], "article_id"),
    # Zement Sack: Jura Flex CEM II (7.8M) then Jura Fix CEM I (6.5M).
    ("Portlandzement Sack", ["1007008", "7011625"], "article_id"),
    # Geberit Duschrinne: CleanLine20 30-90cm (6.2M) and 30-130cm (3.5M).
    ("Geberit Duschrinne", ["6143138", "6143140"], "article_id"),
    # Spiegelschrank: ProCasa Uno LED (4.8M) — own-brand bestseller.
    ("Spiegelschrank Bad", ["7998175", "9039415"], "article_id"),

    # ── Brand-filter tests ────────────────────────────────────────────────────
    # Brand explicitly named — LLM must apply supplier filter.
    # Without filtering a higher-selling competitor would win.
    #
    # Makita Bohrhammer: DHR243ZJ (343k) — clear #1 within Makita.
    ("Makita Akku Bohrhammer 18V", ["1065144", "8170146", "1059195"], "article_id"),
    # Bosch Winkelschleifer: filter must suppress Makita (which outsells Bosch here).
    ("Bosch Winkelschleifer 125mm", ["1057802", "1058233", "1075261"], "article_id"),
    # Bosch Akkuschrauber: GDS 18V-1000 (40k) and GO (37k) are top Bosch sellers.
    # Without brand filter retriever surfaces Makita (172k) instead.
    ("Akkuschrauber von Bosch", ["7972540", "7937478"], "article_id"),
    # Makita Flex (colloquial for Winkelschleifer): Akku-variant (237k) is #1,
    # followed by corded models. Tests synonym "Flex" → Winkelschleifer.
    ("Makita Flex 125mm", ["1056499", "1056160", "1056161"], "article_id"),
    # Eckrohrzange Rothenberger: 45° Set (50k) is their bestseller.
    ("Eckrohrzange Rothenberger", ["1147007", "1148381"], "article_id"),
    # Kreissägeblatt Bosch 190mm: Wood 24Z (1585 units) is top seller.
    ("Kreissägeblatt Bosch 190mm", ["1058304", "1058596", "1058595"], "article_id"),
    # Klosettsitz Laufen: Moderna R SLIM (3.4M) is the clear bestseller.
    ("Klosettsitz Laufen", ["6187134", "01780544"], "article_id"),
    # Geberit Dusch-WC: Aquaclean Mera Classic (17.8M) dominates.
    ("Dusch-WC Geberit", ["7501020", "6190872", "7843172"], "article_id"),
    # Kartell Laufen collection: Wandklosett KARTELL LAUFEN UP (652k, 532k).
    ("Kartell Laufen", ["7889586", "7889588", "7889632"], "article_id"),
    # Laufen Auflage-Waschtisch: VAL 55cm (131k) and VAL 45cm (117k).
    ("Auflage Waschtisch Laufen", ["6207643", "6187001"], "article_id"),

    # ── ProOne own-brand catalog ──────────────────────────────────────────────
    # Regression guard: "bieröffner von proone" must surface the ProOne Multi-Tool
    # (Flaschenöffner in akeneo_values) — not a random bestseller like Trockenbeton.
    # Tests quality-gate score floor + _hit_to_text structured-field expansion.
    ("bieröffner von proone", ["1050541", "7794905"], "article_id"),
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
    ("ProOne Schuhreiniger Edi", ["9225388"], "article_id"),
    ("ProOne Baustellenradio Rock One", ["9183348"], "article_id"),
    ("ProOne Kartuschenpistole 225mm", ["1050095"], "article_id"),
    ("ProOne Silikon Sanitär transparent", ["1003220"], "article_id"),
    ("ProOne Elastisches Maleracryl weiss", ["8172012"], "article_id"),
    ("ProOne Montageschaum 1K", ["8171999"], "article_id"),
    ("Schaumpistole von ProOne", ["1050511", "7504843"], "article_id"),

    # ── Synonym / colloquial ──────────────────────────────────────────────────
    # Silikon schwarz: PCI Silcoferm S 40 (64k) is actual bestseller.
    ("Silikon schwarz Kartusche", ["7855547", "7520808"], "article_id"),
    # Hammerstiel: top seller is 1144601 (345 units), niche category.
    ("Stiel für einen Hammer", ["1144601", "1055521", "1048808"], "article_id"),
    # Victorinox: Swiss Tool Spirit X (9k) and Ranger Grip 61 (7.5k) are top.
    ("Messer Victorinox", ["1143516", "1143469"], "article_id"),

    # ── Bathroom fixtures ─────────────────────────────────────────────────────
    ("freistehende Acryl Badewanne 170cm", ["7826586"], "article_id"),
    ("Duschwand Walk-In Glas 150", ["7885103"], "article_id"),
    ("Hansgrohe Brause sBox", ["7869167", "8188090"], "article_id"),

    # ── General tools / hardware ──────────────────────────────────────────────
    ("Schutzhelm Bauhelm", ["1054731", "1060660", "9137783"], "article_id"),
    ("Schutzhelm in weiss", ["9137783", "1055431"], "article_id"),
    ("LED Taschenlampe mit USB", ["7779803"], "article_id"),
    ("Werkzeugkoffer Aluminium", ["1147106", "7984433"], "article_id"),
    ("Arbeitshandschuh Leder", ["6243635"], "article_id"),
    ("Bleistift Zimmermann Caran d'Ache", ["7918735", "7919406"], "article_id"),
    ("Maurerkelle 180mm", ["7918976"], "article_id"),
    ("Wasserwaage 60cm BMI", ["7995907"], "article_id"),
    ("Zollstock 2 Meter", ["9174216", "7919077"], "article_id"),

    # ── Natural-language / user-phrasing variants ─────────────────────────────
    ("welches Radio gibt es von ProOne?", ["9183348"], "article_id"),
    ("ich brauche einen 18V Bohrhammer von Makita", ["1065144"], "article_id"),
    ("einen Hammer mit kurzem Stiel", ["1144601", "1055521", "1048808"], "article_id"),
    ("gibt es Schuhreiniger bei euch?", ["9225388"], "article_id"),
    ("ProOne Silikon Sanitär transparent 310ml", ["1003220"], "article_id"),
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

# Negative tests: query must NOT surface results containing the excluded terms.
# Each entry: (index_name, label, exclusion_cases)
ONETRADE_DE_EXCLUSIONS: list[EXCLUSION_CASE] = [
    # User wants trockenbeton but explicitly excludes Fixit brand.
    # Tests negation filter-intent detection + reasoning node exclusion path.
    ("trockenbeton aber nicht von fixit", ["fixit"], "article_id"),
    # Same intent phrased differently.
    ("Trockenbeton ohne Fixit", ["fixit"], "article_id"),
    # Exclude own-brand: user wants Trockenbeton but not ProOne.
    # Tests that exclusion overrides the is_own_brand boost.
    ("Trockenbeton nicht von ProOne", ["proone", "procasa"], "article_id"),
    # Klosettsitz without Laufen — should return Geberit/Neoperl/other brands.
    ("Klosettsitz nicht Laufen", ["laufen"], "article_id"),
]

EXCLUSION_SUITES: list[tuple[str, str, list[EXCLUSION_CASE]]] = [
    ("onetrade_articles_de", "OneTrade DE Articles", ONETRADE_DE_EXCLUSIONS),
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
    excl_passes = 0
    excl_total = 0

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

    for index, label, excl_cases in EXCLUSION_SUITES:
        rag = await _build_rag(index)
        print(f"\n══ {label}: negative exclusion ({len(excl_cases)} cases) ══")
        p, t = await run_exclusion_hits(rag, excl_cases, sem, label=f"{label}/excl")
        excl_passes += p
        excl_total += t

    elapsed = time.perf_counter() - t0
    avg_consistency = (
        consistency_sum / consistency_groups if consistency_groups else 0.0
    )
    avg_stable = stable_sum / consistency_groups if consistency_groups else 0.0
    excl_rate = excl_passes / excl_total if excl_total else 0.0

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
    if excl_total:
        print(f"  negative exclusion pass:     {excl_passes}/{excl_total} = {excl_rate:.4f}")
    print(f"  avg query latency:           {avg_query_latency_ms:.0f}ms")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
