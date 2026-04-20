"""One-shot generator: build paraphrases.json and synthetic.json for the eval_v2 suite.

Configure SEED_QUERIES and SAMPLE_INDEX below, then run:
    uv run python -m tests.eval_v2.generate

Outputs are checked in; running again refreshes them from cache or LLM.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rag7._cache import load, save  # type: ignore[import-not-found]

_OUT_DIR = Path(__file__).parent

# ── Configure seed queries for paraphrase generation ─────────────────────────
# Add representative queries from your domain to build a paraphrase corpus.
SEED_QUERIES: list[str] = [
    # Examples — replace with queries from your own index:
    # "Bosch SDS Plus drill bit 10mm",
    # "waterproof sealant for bathroom tiles",
    # "M10 hex bolt stainless steel",
]

# Meilisearch index to sample documents from for synthetic query generation.
SAMPLE_INDEX: str = ""  # e.g. "my_product_catalog"
# ─────────────────────────────────────────────────────────────────────────────


def _get_llm() -> Any:
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(  # type: ignore[call-arg]
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],  # type: ignore[arg-type]
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini"),
        temperature=0.3,
    )


async def _paraphrase_one(llm: Any, query: str, n: int = 3) -> list[str]:
    cached = load("paraphrase-v1", n, query)
    if isinstance(cached, list) and len(cached) >= n:
        return cached[:n]
    prompt = (
        f"Rewrite this search query in {n} different ways that a real user might "
        "phrase it. Keep the intent identical. Vary wording, order, and specificity "
        "— but do NOT introduce new constraints or remove required attributes. "
        "Return a JSON list of strings only, no prose.\n\n"
        f"Query: {query}"
    )
    from langchain_core.messages import HumanMessage

    try:
        resp = await llm.ainvoke([HumanMessage(prompt)])
        text = str(resp.content).strip()
        text = (
            text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        variants = json.loads(text)
        if isinstance(variants, list):
            cleaned = [
                str(v).strip()
                for v in variants
                if str(v).strip() and str(v).strip() != query
            ]
            save("paraphrase-v1", n, query, value=cleaned[:n])
            return cleaned[:n]
    except Exception as e:
        print(f"  paraphrase error for {query!r}: {e}")
    return []


async def build_paraphrases(seeds: list[str], n_per_seed: int = 3) -> None:
    llm = _get_llm()
    sem = asyncio.Semaphore(10)

    async def _one(seed: str) -> dict[str, Any]:
        async with sem:
            variants = await _paraphrase_one(llm, seed, n_per_seed)
        return {"seed": seed, "variants": variants}

    entries = await asyncio.gather(*(_one(s) for s in seeds))
    entries = [e for e in entries if e["variants"]]
    (_OUT_DIR / "paraphrases.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {len(entries)} paraphrase groups.")


async def _synthetic_one(
    llm: Any, doc_text: str, doc_id: str, id_field: str, n: int = 2
) -> list[dict[str, Any]]:
    cached = load("synthetic-v1", n, doc_id)
    if isinstance(cached, list) and len(cached) >= n:
        return cached[:n]
    prompt = (
        f"You are shown a catalog entry. Write {n} plausible user search queries "
        "that a customer would type to find this item. Queries should be diverse "
        "(short keyword, technical question, brand-agnostic need). Each query must "
        "be solvable from the entry alone — no invented facts. Return a JSON list "
        "of strings only.\n\n"
        f"Entry:\n{doc_text[:1500]}"
    )
    from langchain_core.messages import HumanMessage

    try:
        resp = await llm.ainvoke([HumanMessage(prompt)])
        text = str(resp.content).strip()
        text = (
            text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        queries = json.loads(text)
        if isinstance(queries, list):
            out = [
                {"query": str(q).strip(), "doc_id": doc_id, "id_field": id_field}
                for q in queries
                if str(q).strip()
            ][:n]
            save("synthetic-v1", n, doc_id, value=out)
            return out
    except Exception as e:
        print(f"  synthetic error for {doc_id}: {e}")
    return []


async def build_synthetic(docs: list[tuple[str, str, str]], n_per_doc: int = 2) -> None:
    """docs: list of (doc_text, doc_id, id_field) triples sampled from an index."""
    llm = _get_llm()
    sem = asyncio.Semaphore(10)

    async def _one(doc_text: str, doc_id: str, id_field: str) -> list[dict[str, Any]]:
        async with sem:
            return await _synthetic_one(llm, doc_text, doc_id, id_field, n_per_doc)

    batches = await asyncio.gather(*(_one(t, i, f) for t, i, f in docs))
    entries = [e for batch in batches for e in batch]
    (_OUT_DIR / "synthetic.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {len(entries)} synthetic queries from {len(docs)} docs.")


def sample_docs_from_meili(
    index: str,
    id_field: str = "id",
    n: int = 50,
    filter_expr: str | None = None,
    text_fields: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Sample `n` documents from a Meilisearch index for synthetic query generation.

    Returns list of (doc_text, doc_id, id_field) triples.
    """
    import meilisearch

    client = meilisearch.Client(
        os.getenv("MEILI_URL", "http://localhost:7700"),
        os.getenv("MEILI_KEY", "masterKey"),
    )
    params: dict[str, Any] = {"limit": n}
    if filter_expr:
        params["filter"] = filter_expr
    hits = client.index(index).search("", params)["hits"]
    skip = {"id", id_field, "url"}
    out: list[tuple[str, str, str]] = []
    for h in hits:
        doc_id = str(h.get(id_field) or h.get("id") or "")
        if not doc_id:
            continue
        fields = text_fields or [
            k
            for k, v in h.items()
            if isinstance(v, str) and v.strip() and k not in skip
        ]
        text_parts = [
            f"{k}: {h[k]}" for k in fields if isinstance(h.get(k), str) and h[k].strip()
        ]
        out.append(("\n".join(text_parts), doc_id, id_field))
    return out


async def main() -> None:
    if SEED_QUERIES:
        await build_paraphrases(SEED_QUERIES, n_per_seed=3)
    else:
        print("No SEED_QUERIES configured — skipping paraphrase generation.")

    if SAMPLE_INDEX:
        docs = sample_docs_from_meili(SAMPLE_INDEX, n=40)
        await build_synthetic(docs, n_per_doc=2)
    else:
        print("No SAMPLE_INDEX configured — skipping synthetic generation.")


if __name__ == "__main__":
    asyncio.run(main())
