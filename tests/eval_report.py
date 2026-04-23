"""Generate a per-query hit/miss markdown report for eval_v2.

Usage: uv run --env-file .env python tests/eval_report.py [output.md]
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.eval_v2.__main__ import (
    EXCLUSION_SUITES,
    SUITES,
)
from tests.eval_v2.adversarial import build_adversarial_cases
from tests.eval_v2.runner import (
    EXCLUSION_CASE,
    HIT_CASE,
    _retrieve_docs,
    _retrieve_ids,
)


@dataclass
class QueryResult:
    query: str
    expected: list[str]
    retrieved: list[str]
    hit: bool
    suite: str
    kind: str  # "base" | "adversarial" | "exclusion"


@dataclass
class SuiteReport:
    label: str
    results: list[QueryResult] = field(default_factory=list)


async def _run_suite_detailed(
    rag: Any,
    label: str,
    base_cases: list[HIT_CASE],
    sem: asyncio.Semaphore,
) -> SuiteReport:
    report = SuiteReport(label=label)

    adversarial = build_adversarial_cases(base_cases)

    async def _one(case: HIT_CASE, kind: str) -> QueryResult:
        query, expected, id_field = case
        async with sem:
            retrieved = await _retrieve_ids(rag, query, id_field)
        hit = any(str(e) in retrieved for e in expected)
        return QueryResult(
            query=query,
            expected=expected,
            retrieved=retrieved,
            hit=hit,
            suite=label,
            kind=kind,
        )

    base_results = await asyncio.gather(*(_one(c, "base") for c in base_cases))
    adv_results = await asyncio.gather(*(_one(c, "adversarial") for c in adversarial))

    report.results.extend(base_results)
    report.results.extend(adv_results)
    return report


async def _run_exclusion_detailed(
    rag: Any,
    label: str,
    cases: list[EXCLUSION_CASE],
    sem: asyncio.Semaphore,
) -> list[QueryResult]:
    async def _one(case: EXCLUSION_CASE) -> QueryResult:
        query, excluded_terms, _ = case
        async with sem:
            docs = await _retrieve_docs(rag, query)
        content_hits = [
            d.page_content[:120] for d in docs
            if any(t.lower() in (d.page_content or "").lower() for t in excluded_terms)
        ]
        passed = len(content_hits) == 0
        return QueryResult(
            query=query,
            expected=[f"NOT: {', '.join(excluded_terms)}"],
            retrieved=content_hits or ["(none containing excluded terms)"],
            hit=passed,
            suite=label,
            kind="exclusion",
        )

    return list(await asyncio.gather(*(_one(c) for c in cases)))


def _md_table_rows(results: list[QueryResult], kind: str) -> list[str]:
    rows = []
    for r in results:
        if r.kind != kind:
            continue
        status = "✅" if r.hit else "❌"
        expected = ", ".join(r.expected[:3])
        retrieved = ", ".join(r.retrieved[:3])
        rows.append(f"| {status} | `{r.query}` | `{expected}` | `{retrieved}` |")
    return rows


def build_markdown(suite_reports: list[SuiteReport], excl_results: list[QueryResult]) -> str:
    lines: list[str] = []
    today = date.today().isoformat()

    lines += [f"# eval_v2 Hit/Miss Report — {today}", ""]

    # Summary table
    lines += ["## Summary", "", "| Suite | Kind | Hits | Total | Rate |", "|---|---|---|---|---|"]
    for sr in suite_reports:
        for kind in ("base", "adversarial"):
            subset = [r for r in sr.results if r.kind == kind]
            if not subset:
                continue
            h = sum(r.hit for r in subset)
            t = len(subset)
            lines.append(f"| {sr.label} | {kind} | {h} | {t} | {h/t:.1%} |")
    if excl_results:
        h = sum(r.hit for r in excl_results)
        t = len(excl_results)
        lines.append(f"| exclusion | negative | {h} | {t} | {h/t:.1%} |")

    lines.append("")

    # Per-suite detail
    for sr in suite_reports:
        lines += [f"## {sr.label}", ""]

        for kind, heading in [("base", "Base queries"), ("adversarial", "Adversarial variants")]:
            subset = [r for r in sr.results if r.kind == kind]
            if not subset:
                continue
            hits = sum(r.hit for r in subset)
            lines += [
                f"### {heading} — {hits}/{len(subset)} ({hits/len(subset):.1%})",
                "",
                "| | Query | Expected IDs | Top-5 retrieved |",
                "|---|---|---|---|",
            ]
            lines += _md_table_rows(subset, kind)
            lines.append("")

        # Misses only callout
        misses = [r for r in sr.results if not r.hit]
        if misses:
            lines += ["### Misses only", ""]
            for r in misses:
                exp = ", ".join(r.expected[:4])
                got = ", ".join(r.retrieved[:5])
                lines.append(f"- **[{r.kind}]** `{r.query}`  ")
                lines.append(f"  expected: `{exp}`  ")
                lines.append(f"  got: `{got}`")
            lines.append("")

    # Exclusion section
    if excl_results:
        lines += ["## Negative Exclusion Tests", ""]
        lines += [
            "| | Query | Excluded terms | Offending snippets |",
            "|---|---|---|---|",
        ]
        lines += _md_table_rows(excl_results, "exclusion")
        lines.append("")

    return "\n".join(lines)


async def _build_rag(index: str) -> Any:
    from rag7 import AgenticRAG, RAGConfig
    from rag7.backend import MeilisearchBackend
    from rag7.utils import _make_azure_embed_fn

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


async def main(out_path: Path) -> None:
    sem = asyncio.Semaphore(20)
    suite_reports: list[SuiteReport] = []

    for index, label, base in SUITES:
        print(f"Running {label}...", flush=True)
        rag = await _build_rag(index)
        report = await _run_suite_detailed(rag, label, base, sem)
        suite_reports.append(report)

    excl_results: list[QueryResult] = []
    for index, label, cases in EXCLUSION_SUITES:
        print(f"Running exclusion: {label}...", flush=True)
        rag = await _build_rag(index)
        excl_results.extend(await _run_exclusion_detailed(rag, label, cases, sem))

    md = build_markdown(suite_reports, excl_results)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_report.md")
    asyncio.run(main(out))
