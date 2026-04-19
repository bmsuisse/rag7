"""Lightweight retrieval regression guard — runs in CI with zero external deps.

Unlike ``tests/test_mteb_benchmark.py`` (needs Ollama) and ``tests/eval_v2``
(needs Meilisearch cloud), this test is fully self-contained:

* A bundled 60-doc fixture across 6 topical clusters.
* A deterministic hash-based bag-of-terms pseudo-embedding — no model
  download, no network, no API keys.
* ``InMemoryBackend`` (langchain-core vector store, already a core dep).

The pseudo-embedding is NOT a real quality metric. Its only job is to
exercise the ``SearchBackend`` → ``InMemoryVectorStore`` → ``search(...)``
plumbing end-to-end, so that a structural retrieval regression (bad
``SearchRequest`` fields, broken vector wiring, wrong hit shape) fails
loudly. If the plumbing is intact, clustered docs will easily clear an
80 %% recall@10 bar.

Run: ``uv run pytest tests/test_mteb_ci.py -v``
"""

from __future__ import annotations

import hashlib
import math
import time

import pytest

from rag7.backend import InMemoryBackend, SearchRequest

# ── Pseudo-embedding ─────────────────────────────────────────────────────────

EMBED_DIM = 128


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, length >= 2."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if len(buf) >= 2:
                out.append("".join(buf))
            buf = []
    if len(buf) >= 2:
        out.append("".join(buf))
    return out


def _hash_bucket(token: str, dim: int) -> int:
    # Stable across runs — sha1 first 4 bytes as uint.
    return int.from_bytes(hashlib.sha1(token.encode()).digest()[:4], "big") % dim


def pseudo_embed(text: str) -> list[float]:
    """Hash-bucket bag-of-terms vector, L2-normalised.

    Docs sharing vocabulary → aligned vectors → high cosine similarity.
    Good enough to verify retrieval wiring, nothing more.
    """
    vec = [0.0] * EMBED_DIM
    for tok in _tokenize(text):
        vec[_hash_bucket(tok, EMBED_DIM)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


# ── Fixture corpus (60 docs, 6 clusters of 10) ───────────────────────────────

CLUSTERS: dict[str, list[str]] = {
    "python": [
        "Python is a high-level interpreted programming language.",
        "Python uses significant whitespace and dynamic typing.",
        "List comprehensions are a concise Python idiom.",
        "Decorators in Python wrap functions to extend behaviour.",
        "Python's asyncio enables concurrent coroutines.",
        "Python package management is handled by pip and uv.",
        "The Python standard library includes many modules.",
        "Python dataclasses generate boilerplate for simple classes.",
        "Type hints in Python improve static analysis.",
        "Python virtual environments isolate project dependencies.",
    ],
    "database": [
        "PostgreSQL is a relational database management system.",
        "SQL is the standard query language for relational databases.",
        "A database index speeds up query lookups dramatically.",
        "Transactions in databases satisfy ACID properties.",
        "Joins combine rows from multiple database tables.",
        "Database normalisation reduces data redundancy.",
        "Replication keeps multiple database copies in sync.",
        "MySQL is a widely used relational database engine.",
        "A primary key uniquely identifies database rows.",
        "Sharding partitions a database horizontally for scale.",
    ],
    "cooking": [
        "Pasta is cooked in salted boiling water until al dente.",
        "Searing meat in a hot pan develops rich umami flavour.",
        "A chef's knife is the primary cooking tool in any kitchen.",
        "Simmering a sauce reduces volume and concentrates flavour.",
        "Baking bread requires yeast, flour, water, and salt.",
        "A sourdough starter ferments wild yeast cultures.",
        "Caramelising onions slowly brings out their sweetness.",
        "Roasting vegetables in olive oil enhances their flavour.",
        "Pan frying fish skin-side down crisps the skin.",
        "A cast-iron skillet retains heat evenly for cooking.",
    ],
    "space": [
        "NASA sent the Apollo astronauts to the Moon in 1969.",
        "A rocket escapes Earth gravity by reaching orbital velocity.",
        "The Hubble telescope orbits Earth observing distant galaxies.",
        "Mars is the fourth planet from the Sun in our solar system.",
        "A black hole has gravity so strong even light cannot escape.",
        "The International Space Station orbits Earth every 90 minutes.",
        "SpaceX reuses rocket boosters to lower launch costs.",
        "Jupiter is the largest planet in the solar system.",
        "A solar eclipse happens when the Moon blocks the Sun.",
        "The Voyager probes carry a golden record into interstellar space.",
    ],
    "music": [
        "A guitar has six strings tuned to standard EADGBE pitches.",
        "The piano is a keyboard instrument with 88 keys.",
        "A symphony orchestra groups strings, winds, brass, and percussion.",
        "Jazz improvisation builds melodies over chord progressions.",
        "A drum kit anchors the rhythm of a rock band.",
        "Classical music composers include Bach, Mozart, and Beethoven.",
        "Electric guitar pedals shape tone through effects.",
        "A saxophone is a reed instrument common in jazz ensembles.",
        "Vinyl records reproduce music via a physical groove.",
        "A music producer shapes the final sound of an album recording.",
    ],
    "finance": [
        "A stock represents partial ownership of a public company.",
        "A bond is a loan an investor makes to a government or firm.",
        "Inflation erodes the purchasing power of currency over time.",
        "A central bank sets interest rates to steer the economy.",
        "Index funds track a broad basket of stocks passively.",
        "Compound interest grows savings exponentially over many years.",
        "A mortgage is a loan used to purchase real estate property.",
        "Diversification across asset classes reduces portfolio risk.",
        "A credit score summarises an individual's borrowing history.",
        "A recession is a sustained decline in economic activity.",
    ],
}


def _build_corpus() -> tuple[list[dict], dict[str, set[str]]]:
    """Return (docs, cluster_to_doc_ids)."""
    docs: list[dict] = []
    groups: dict[str, set[str]] = {}
    for cluster, texts in CLUSTERS.items():
        ids: set[str] = set()
        for i, text in enumerate(texts):
            doc_id = f"{cluster}-{i}"
            docs.append({"id": doc_id, "content": text, "cluster": cluster})
            ids.add(doc_id)
        groups[cluster] = ids
    return docs, groups


# Queries paired with the specific doc ids that the query terms occur in.
# Each query's gold set is small (2-4 docs), so recall@10 = fraction of those
# gold docs that appear in the top-10 hits. With a working vector-search
# path this should comfortably clear 80 %.
QUERIES: list[tuple[str, set[str]]] = [
    ("python typing dataclasses hints", {"python-7", "python-8"}),
    ("pip uv virtual environment package", {"python-5", "python-9"}),
    ("asyncio coroutines concurrent", {"python-4"}),
    (
        "postgresql relational database SQL",
        {"database-0", "database-1", "database-7"},
    ),
    ("database index primary key joins", {"database-2", "database-4", "database-8"}),
    ("sourdough bread yeast baking flour", {"cooking-4", "cooking-5"}),
    ("skillet cast-iron pan searing", {"cooking-1", "cooking-9"}),
    ("NASA Apollo astronauts Moon", {"space-0"}),
    ("Mars Jupiter planet solar system", {"space-3", "space-7"}),
    ("Hubble telescope galaxies orbit Earth", {"space-2"}),
    ("guitar strings pedals electric", {"music-0", "music-6"}),
    ("jazz saxophone improvisation", {"music-3", "music-7"}),
    ("piano keyboard classical composers", {"music-1", "music-5"}),
    ("bond investor interest rate", {"finance-1", "finance-3"}),
    ("mortgage credit score borrowing", {"finance-6", "finance-8"}),
    ("inflation recession economic activity", {"finance-2", "finance-9"}),
]

RECALL_K = 10
RECALL_THRESHOLD = 0.8


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus() -> tuple[list[dict], dict[str, set[str]]]:
    return _build_corpus()


@pytest.fixture(scope="module")
def vector_backend(
    corpus: tuple[list[dict], dict[str, set[str]]],
) -> InMemoryBackend:
    docs, _ = corpus
    return InMemoryBackend(embed_fn=pseudo_embed, documents=docs)


@pytest.fixture(scope="module")
def substring_backend(
    corpus: tuple[list[dict], dict[str, set[str]]],
) -> InMemoryBackend:
    docs, _ = corpus
    return InMemoryBackend(documents=docs)  # no embed_fn → substring fallback


def test_pseudo_embed_is_deterministic() -> None:
    """Hash-based embedding must be stable run-to-run (regression canary)."""
    v1 = pseudo_embed("Python programming language")
    v2 = pseudo_embed("Python programming language")
    assert v1 == v2
    assert len(v1) == EMBED_DIM
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-9


def test_corpus_shape(corpus: tuple[list[dict], dict[str, set[str]]]) -> None:
    docs, groups = corpus
    assert len(docs) == 60
    assert len(groups) == 6
    assert all(len(ids) == 10 for ids in groups.values())


def test_vector_recall_at_10(
    corpus: tuple[list[dict], dict[str, set[str]]],
    vector_backend: InMemoryBackend,
) -> None:
    """End-to-end: SearchRequest → InMemoryVectorStore → hits with metadata.

    Asserts mean recall@10 clears a threshold easily reachable with the
    clustered fixture. A regression in the vector-search wiring will
    collapse recall well below 0.8 and fail loudly.
    """
    recalls: list[float] = []
    t0 = time.monotonic()

    for query, gold in QUERIES:
        hits = vector_backend.search(SearchRequest(query=query, limit=RECALL_K))
        assert len(hits) == RECALL_K, f"expected {RECALL_K} hits, got {len(hits)}"
        assert all("id" in h and "cluster" in h for h in hits), (
            "hit metadata lost in round-trip"
        )
        assert all("_rankingScore" in h for h in hits), (
            "ranking score missing — InMemoryVectorStore path not taken"
        )
        ids = [h["id"] for h in hits]
        recalls.append(_recall_at_k(ids, gold, RECALL_K))

    elapsed = time.monotonic() - t0
    mean_recall = sum(recalls) / len(recalls)

    # Surface the numbers in -v output.
    print(
        f"\n  vector recall@{RECALL_K}: mean={mean_recall:.3f} "
        f"min={min(recalls):.3f} n={len(recalls)} elapsed={elapsed * 1000:.0f}ms"
    )

    assert mean_recall >= RECALL_THRESHOLD, (
        f"mean recall@{RECALL_K}={mean_recall:.3f} < {RECALL_THRESHOLD} — "
        f"retrieval regressed. per-query: {[round(r, 2) for r in recalls]}"
    )


def test_substring_fallback_finds_exact_matches(
    substring_backend: InMemoryBackend,
) -> None:
    """InMemoryBackend without embed_fn falls back to substring matching.

    This is the shape CI sees when the backend is instantiated without
    an embedder (e.g. early smoke tests). Verify the fallback still
    returns hits for obvious substring queries.
    """
    hits = substring_backend.search(SearchRequest(query="sourdough", limit=5))
    assert len(hits) >= 1
    assert any("sourdough" in h["content"].lower() for h in hits)

    hits = substring_backend.search(SearchRequest(query="voyager", limit=5))
    assert len(hits) >= 1
    assert any("voyager" in h["content"].lower() for h in hits)


def test_batch_search_matches_sequential(
    corpus: tuple[list[dict], dict[str, set[str]]],
    vector_backend: InMemoryBackend,
) -> None:
    """Default ``batch_search`` must fan out to ``search`` without altering results."""
    reqs = [SearchRequest(query=q, limit=RECALL_K) for q, _ in QUERIES[:4]]
    batch = vector_backend.batch_search(reqs)
    sequential = [vector_backend.search(r) for r in reqs]

    assert len(batch) == len(sequential) == 4
    for b, s in zip(batch, sequential):
        assert [h["id"] for h in b] == [h["id"] for h in s]
