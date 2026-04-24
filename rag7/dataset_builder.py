"""Interactive test-dataset builder for rag7 tuning."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .cli import _banner, _print, _prompt, _section


@dataclass
class TestCase:
    queries: list[str]
    expected_ids: list[str]
    id_field: str = "id"
    note: str = ""

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {"query": q, "expected_ids": self.expected_ids, "id_field": self.id_field}
            for q in self.queries
        ]


def _doc_label(doc: Document, id_field: str) -> str:
    doc_id = doc.metadata.get(id_field, "?")
    title = doc.metadata.get("name") or doc.metadata.get("title") or ""
    snippet = doc.page_content[:70].replace("\n", " ")
    label = title or snippet
    return f"id={doc_id}  {label}"


def _show_results(docs: list[Document], id_field: str) -> None:
    if not docs:
        _print("  [dim]No results.[/dim]")
        return
    for i, doc in enumerate(docs, 1):
        _print(f"  [bold]{i:>2}.[/bold]  {_doc_label(doc, id_field)}")


def _search(rag: Any, query: str, top_k: int) -> tuple[str, list[Document]]:
    from .cli import _HAS_RICH, _console

    async def _run() -> tuple[str, list[Document]]:
        return await rag._aretrieve_documents(query, top_k=top_k)

    if _HAS_RICH and _console is not None:
        with _console.status("[bold yellow]Searching…[/bold yellow]", spinner="dots"):
            return asyncio.run(_run())
    print("Searching…")
    return asyncio.run(_run())


def _handle_not_found(
    rag: Any,
    original_query: str,
    top_k: int,
    id_field: str,
) -> tuple[str, list[str]] | None:
    """Retry with alternative queries until the target doc is found.

    All queries tried (including the original failing one) are returned
    as variations — they're all valid search intents the tuner must fix.
    Returns None if user skips.
    """
    tried = [original_query]
    _print("  [dim]Try a different query to locate the doc.[/dim]")
    _print("  [dim]'s' = skip case  |  'm' = enter doc ID manually[/dim]")

    while True:
        new_query = _prompt("  New query").strip()
        if new_query.lower() == "s" or not new_query:
            return None
        if new_query.lower() == "m":
            doc_id = _prompt("  Doc ID").strip()
            if doc_id:
                return doc_id, tried
            continue

        tried.append(new_query)
        _, docs = _search(rag, new_query, top_k)
        if not docs:
            _print("  [yellow]Still no results.[/yellow]")
            continue

        _show_results(docs, id_field)
        pick = _prompt(f"  Correct? (1-{len(docs)} / blank=try again)").strip()
        if not pick:
            continue
        try:
            idx = int(pick) - 1
            if not (0 <= idx < len(docs)):
                raise ValueError
        except ValueError:
            _print("  [red]Invalid.[/red]")
            continue

        doc = docs[idx]
        doc_id = str(doc.metadata.get(id_field, ""))
        if not doc_id or doc_id == "None":
            _print(
                f"  [yellow]No '{id_field}'. Keys: {list(doc.metadata.keys())}[/yellow]"
            )
            doc_id = _prompt("  Enter ID manually").strip()
        if doc_id:
            return doc_id, tried


def _ask_variations(current: list[str]) -> list[str]:
    """Prompt for extra query phrasings; returns current + new ones."""
    _print("  [dim]Other ways users phrase this (comma-separated, blank=skip):[/dim]")
    raw = _prompt("  Variations", default="").strip()
    extras = [v.strip() for v in raw.split(",") if v.strip()]
    return current + extras


def build_testset_interactive(
    rag: Any,
    output_path: str | Path,
    *,
    top_k: int = 10,
    id_field: str = "id",
) -> None:
    output = Path(output_path)
    cases: list[TestCase] = []

    _banner("rag7 — Dataset Builder")
    _print(f"  Output → [bold]{output}[/bold]")
    _print("  Each case: query + the expected top doc.")
    _print("  Query [bold]done[/bold] to finish.\n")

    id_field = _prompt("  Doc ID field", default=id_field)
    top_k = int(_prompt("  Top-k to show per search", default=str(top_k)))

    while True:
        n = len(cases) + 1
        _section(f"Test Case #{n}")
        query = _prompt("  Query (or 'done')").strip()
        if not query or query.lower() == "done":
            break

        _, docs = _search(rag, query, top_k)

        if not docs:
            _print("  [yellow]Zero results.[/yellow]")
            choice = _prompt("  [s]kip / [r]etry with different query", default="s")
            if choice.lower().startswith("r"):
                result = _handle_not_found(rag, query, top_k, id_field)
                if result:
                    doc_id, all_queries = result
                    all_queries = _ask_variations(all_queries)
                    note = _prompt("  Note (optional)", default="").strip()
                    cases.append(
                        TestCase(
                            queries=all_queries,
                            expected_ids=[doc_id],
                            id_field=id_field,
                            note=note,
                        )
                    )
                    _print(
                        f"  [green]Saved: {len(all_queries)} queries → {doc_id}[/green]"
                    )
            continue

        _show_results(docs, id_field)
        choice = (
            _prompt(f"  Pick (1-{len(docs)}) / [n]ot found / [s]kip").strip().lower()
        )

        if choice == "s" or not choice:
            continue

        if choice == "n":
            result = _handle_not_found(rag, query, top_k, id_field)
            if result:
                doc_id, all_queries = result
                all_queries = _ask_variations(all_queries)
                note = _prompt("  Note (optional)", default="").strip()
                cases.append(
                    TestCase(
                        queries=all_queries,
                        expected_ids=[doc_id],
                        id_field=id_field,
                        note=note,
                    )
                )
                _print(f"  [green]Saved: {len(all_queries)} queries → {doc_id}[/green]")
            continue

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(docs)):
                raise ValueError
        except ValueError:
            _print("  [red]Invalid.[/red]")
            continue

        doc = docs[idx]
        doc_id = str(doc.metadata.get(id_field, ""))
        if not doc_id or doc_id == "None":
            _print(
                f"  [yellow]No '{id_field}'. Keys: {list(doc.metadata.keys())}[/yellow]"
            )
            doc_id = _prompt("  Enter ID manually").strip()
        if not doc_id:
            continue

        all_queries = _ask_variations([query])
        note = _prompt("  Note (optional)", default="").strip()
        cases.append(
            TestCase(
                queries=all_queries,
                expected_ids=[doc_id],
                id_field=id_field,
                note=note,
            )
        )
        _print(f"  [green]Saved: {len(all_queries)} queries → {doc_id}[/green]")

    if not cases:
        _print("  No cases collected. Exiting.")
        return

    all_dicts = [d for case in cases for d in case.to_dicts()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        if output.suffix == ".jsonl":
            for d in all_dicts:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        else:
            json.dump(all_dicts, f, indent=2, ensure_ascii=False)

    _print(
        f"\n  [bold green]Saved {len(all_dicts)} hit cases "
        f"({len(cases)} unique docs) → {output}[/bold green]"
    )
    _print(f"  Tune: [bold]python -m rag7.tuner --hits {output} --index <name>[/bold]")


def _cli_main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m rag7.dataset",
        description="Interactively build a rag7 tuning test-dataset.",
    )
    parser.add_argument("--output", "-o", default="testset.json")
    parser.add_argument("--index", "-i", default=None)
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--id-field", default="id")
    args = parser.parse_args()

    from .factory import init_agent
    from .utils import _make_azure_embed_fn

    envd: dict[str, Any] = {}
    import os

    if os.getenv("MEILI_URL"):
        envd["backend"] = "meilisearch"
        envd["backend_url"] = os.getenv("MEILI_URL")
        if os.getenv("MEILI_KEY") or os.getenv("MEILI_MASTER_KEY"):
            envd["backend_kwargs"] = {
                "api_key": os.getenv("MEILI_KEY") or os.getenv("MEILI_MASTER_KEY")
            }

    index = args.index
    if not index:
        index = os.getenv("MS_INDEX") or _prompt(
            "  Index / collection", default="documents"
        )

    embed_fn = _make_azure_embed_fn()
    if embed_fn is None:
        _print("[red]Azure OpenAI embed env vars missing.[/red]")
        sys.exit(1)

    config: dict[str, Any] = {"index": index, "embed_fn": embed_fn, **envd}
    try:
        rag = init_agent(**{k: v for k, v in config.items() if v is not None})
    except Exception as exc:
        _print(f"[red]Failed to init: {exc}[/red]")
        sys.exit(1)

    build_testset_interactive(
        rag,
        args.output,
        top_k=args.top_k,
        id_field=args.id_field,
    )


if __name__ == "__main__":
    _cli_main()
