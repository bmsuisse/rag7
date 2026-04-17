"""
Multilingual Open-Data Downloader → Parquet + JSON
────────────────────────────────────────────────────
Downloads open datasets and saves them as:
  data/<source>/<lang>.parquet
  data/<source>/<lang>.json

Sources:
  wiki   Wikipedia        DE / FR / IT   HuggingFace streaming
  swiss  opendata.swiss   DE / FR / IT   CKAN API
  opus   OPUS Books       DE / FR / IT   HuggingFace streaming
  rteb   RTEB datasets    DE / FR / EN   HuggingFace

Usage:
  python ingest.py                        # all sources, all languages
  python ingest.py --sources wiki,swiss   # subset of sources
  python ingest.py --limit 500            # cap per source per language
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
LANGUAGES = ["de", "fr", "it"]


def _id(*parts: str) -> str:
    return hashlib.sha1(":".join(p[:120] for p in parts).encode()).hexdigest()


def _save(rows: list[dict], source: str, lang: str) -> pd.DataFrame:
    out = DATA_DIR / source
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / f"{lang}.parquet", index=False)
    df.to_json(out / f"{lang}.json", orient="records", force_ascii=False, indent=2)
    return df


# ── Wikipedia ─────────────────────────────────────────────────────────────────


def fetch_wikipedia(langs: list[str], limit: int) -> None:
    from datasets import load_dataset  # type: ignore

    for lang in langs:
        print(f"  wikipedia/{lang} …")
        ds = load_dataset(
            "wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            rows.append(
                {
                    "id": _id("wikipedia", lang, row.get("url", ""), row["title"]),
                    "title": row["title"],
                    "content": (row["text"] or "")[:4000],
                    "language": lang,
                    "source": "wikipedia",
                    "url": row.get("url", ""),
                }
            )
        df = _save(rows, "wikipedia", lang)
        print(f"    {len(df)} docs → data/wikipedia/{lang}.parquet")


# ── opendata.swiss ────────────────────────────────────────────────────────────


def fetch_opendata_swiss(langs: list[str], limit: int) -> None:
    base = "https://ckan.opendata.swiss/api/3/action/package_search"

    for lang in langs:
        print(f"  opendata.swiss/{lang} …")
        rows: list[dict] = []
        start = 0

        while len(rows) < limit:
            r = requests.get(
                base, params={"q": "*:*", "rows": 100, "start": start}, timeout=20
            )
            results = r.json().get("result", {}).get("results", [])
            if not results:
                break
            for pkg in results:
                desc = pkg.get("description") or {}
                title = pkg.get("title") or {}
                text = (
                    desc.get(lang) or desc.get("en") or next(iter(desc.values()), "")
                    if isinstance(desc, dict)
                    else desc or ""
                ).strip()
                name = (
                    title.get(lang) or title.get("en") or next(iter(title.values()), "")
                    if isinstance(title, dict)
                    else title or ""
                ).strip()
                if not text:
                    continue
                rows.append(
                    {
                        "id": _id("swiss", lang, pkg.get("name", ""), name),
                        "title": name,
                        "content": text[:4000],
                        "language": lang,
                        "source": "opendata_swiss",
                        "url": f"https://opendata.swiss/dataset/{pkg.get('name', '')}",
                    }
                )
                if len(rows) >= limit:
                    break
            start += 100

        df = _save(rows, "opendata_swiss", lang)
        print(f"    {len(df)} docs → data/opendata_swiss/{lang}.parquet")


# ── OPUS Books ────────────────────────────────────────────────────────────────


def fetch_opus(langs: list[str], limit: int) -> None:
    from datasets import load_dataset  # type: ignore

    for lang in langs:
        print(f"  opus_books/{lang} …")
        ds = None
        for config in (f"en-{lang}", f"{lang}-en"):
            try:
                ds = load_dataset("opus_books", config, split="train", streaming=True)
                break
            except Exception:
                pass
        if ds is None:
            print(f"    skipped (no config for {lang})")
            continue
        rows = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            text = row.get("translation", {}).get(lang, "")
            if not text:
                continue
            rows.append(
                {
                    "id": _id("opus", lang, str(i), text[:50]),
                    "title": "",
                    "content": text[:4000],
                    "language": lang,
                    "source": "opus_books",
                    "url": "",
                }
            )
        df = _save(rows, "opus_books", lang)
        print(f"    {len(df)} docs → data/opus_books/{lang}.parquet")


# ── RTEB datasets ─────────────────────────────────────────────────────────────

# (hf_id, config, lang, split, text_field, title_field)
RTEB_DATASETS: list[tuple[str, str | None, str, str, str, str]] = [
    # Legal / DE
    ("mteb/LegalQuAD", "corpus", "de", "corpus", "text", "_id"),
    # Legal / EN
    ("mteb/AILA_casedocs", "corpus", "en", "corpus", "text", "title"),
    ("mteb/AILA_statutes", "corpus", "en", "corpus", "text", "title"),
    ("mteb/legal_summarization", "corpus", "en", "corpus", "text", "title"),
    # Science / EN  (BEIR SciFact — 5k docs, fully ingestible)
    ("mteb/scifact", "corpus", "en", "corpus", "text", "title"),
    # Finance / EN  (BEIR FiQA — 57k docs, limit applied)
    ("mteb/fiqa", "corpus", "en", "corpus", "text", "title"),
    # Medical / EN  (BEIR NFCorpus — 3.6k docs, fully ingestible)
    ("mteb/nfcorpus", "corpus", "en", "corpus", "text", "title"),
    # General QA / EN
    ("virattt/financebench", None, "en", "train", "answer", "question"),
    ("lavita/ChatDoctor-HealthCareMagic-100k", None, "en", "train", "output", "input"),
    # Wikipedia / General (NQ)
    ("mteb/nq", "corpus", "en", "corpus", "text", "title"),
    # Multi-hop General (HotpotQA)
    ("mteb/hotpotqa", "corpus", "en", "corpus", "text", "title"),
    # Diverse FAQ (Quora)
    ("mteb/quora", "corpus", "en", "corpus", "text", "title"),
    # French medical
    ("clinia/CUREv1", "corpus", "fr", "dentistry_and_oral_health", "text", "title"),
]


def fetch_rteb(_langs: list[str], limit: int) -> None:
    from datasets import load_dataset  # type: ignore

    buckets: dict[str, list[dict]] = {}

    for hf_id, config, lang, split, text_field, title_field in RTEB_DATASETS:
        print(f"  rteb/{hf_id.split('/')[-1]} ({lang}) …")
        try:
            ds = load_dataset(hf_id, config, split=split, streaming=True)
        except Exception as e:
            print(f"    skipped ({e})")
            continue

        count = 0
        for row in ds:
            if count >= limit:
                break
            val = row.get(text_field, "")
            text = (
                " ".join(str(v) for v in val if v)
                if isinstance(val, list)
                else str(val or "")
            )
            if not text:
                continue
            buckets.setdefault(lang, []).append(
                {
                    "id": _id("rteb", lang, hf_id, text[:60]),
                    "corpus_id": str(
                        row.get("_id", "") or ""
                    ),  # original MTEB corpus _id
                    "title": str(row.get(title_field, "") or "")[:200],
                    "content": text[:4000],
                    "language": lang,
                    "source": f"rteb/{hf_id}",
                    "url": f"https://huggingface.co/datasets/{hf_id}",
                }
            )
            count += 1
        print(f"    {count} docs collected")

    for lang, rows in buckets.items():
        df = _save(rows, "rteb", lang)
        print(f"    saved {len(df)} total rteb/{lang} → data/rteb/{lang}.parquet")


# ── CLI ───────────────────────────────────────────────────────────────────────

SOURCES = {
    "wiki": fetch_wikipedia,
    "swiss": fetch_opendata_swiss,
    "opus": fetch_opus,
    "rteb": fetch_rteb,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download multilingual open data to Parquet + JSON"
    )
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--langs", default=",".join(LANGUAGES))
    parser.add_argument(
        "--limit", type=int, default=500, help="Max docs per source per language"
    )
    args = parser.parse_args()

    sources = args.sources.split(",")
    langs = args.langs.split(",")

    print(f"Sources: {sources}  |  langs: {langs}  |  limit: {args.limit}\n")
    for key in sources:
        fn = SOURCES.get(key)
        if fn is None:
            print(f"Unknown source '{key}', skipping.")
            continue
        fn(langs, args.limit)

    print("\nDone. Files saved in data/")


if __name__ == "__main__":
    main()
