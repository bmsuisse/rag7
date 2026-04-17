"""Cross-backend regression test for Databricks/Jupyter event-loop reentry.

Bug: `rag.invoke(q)` used `asyncio.run(self.ainvoke(q))` which raises
RuntimeError when a loop is already running (the default in Databricks,
Jupyter, FastAPI). Fixed by `_run_sync` — see tests/test_run_sync.py for
the helper's own regression tests.

This test verifies the integration across every backend we ship sync
support for: build a minimal AgenticRAG with the backend, then call
`invoke` / `retrieve_documents` / `batch` from inside a running loop.
If the helper is load-bearing, removing it makes every case below raise
`RuntimeError: asyncio.run() cannot be called from a running event loop`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from rag7 import AgenticRAG, InMemoryBackend  # noqa: E402
from rag7.backend import IndexConfig  # noqa: E402
from rag7.models import (  # noqa: E402
    CollectionIntent,
    FilterIntent,
    MultiQuery,
    QualityAssessment,
    ReasoningVerdict,
    RelevanceCheck,
    SearchQuery,
)


class _StubStructuredChain:
    def __init__(self, value: Any) -> None:
        self._value = value

    def invoke(self, _messages: Any) -> Any:
        return self._value

    async def ainvoke(self, _messages: Any) -> Any:
        return self._value


def _stub_llm() -> MagicMock:
    defaults = {
        SearchQuery: SearchQuery(
            query="", variants=[], semantic_ratio=0.5, fusion="rrf"
        ),
        QualityAssessment: QualityAssessment(sufficient=True, reason=""),
        MultiQuery: MultiQuery(queries=[]),
        FilterIntent: FilterIntent(field=None, value="", operator=""),
        CollectionIntent: CollectionIntent(collections=[]),
        RelevanceCheck: RelevanceCheck(makes_sense=True, confidence=1.0),
        ReasoningVerdict: ReasoningVerdict(),
    }
    llm = MagicMock()
    llm.with_structured_output.side_effect = lambda m, **_k: _StubStructuredChain(
        defaults[m]
    )
    answer = MagicMock()
    answer.content = "stub answer"
    llm.ainvoke = AsyncMock(return_value=answer)
    llm.invoke = MagicMock(return_value=answer)
    return llm


SAMPLE_DOCS = [
    {"id": "1", "content": "Python is a programming language", "category": "tech"},
    {"id": "2", "content": "Dogs are loyal companions", "category": "animals"},
    {"id": "3", "content": "PostgreSQL is a relational database", "category": "tech"},
]

DIM = 8


def _fake_embed(text: str) -> list[float]:
    import hashlib

    h = hashlib.sha256(text.encode()).digest()
    vec = [float(b) / 255.0 for b in h[:DIM]]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


# ── Backend fixtures (one per supported backend) ─────────────────────────────


def _inmemory_backend():
    return InMemoryBackend(documents=SAMPLE_DOCS)


def _chroma_backend():
    pytest.importorskip("chromadb")
    from rag7.backend import ChromaDBBackend

    b = ChromaDBBackend(collection="test_reentry_chroma", embed_fn=_fake_embed)
    b._collection.add(
        ids=[d["id"] for d in SAMPLE_DOCS],
        documents=[d["content"] for d in SAMPLE_DOCS],
        metadatas=[{"category": d["category"]} for d in SAMPLE_DOCS],
    )
    return b


def _lancedb_backend(tmp_path):
    lancedb = pytest.importorskip("lancedb")
    from rag7.backend import LanceDBBackend

    db = lancedb.connect(str(tmp_path / "r.lance"))
    db.create_table(
        "r",
        data=[
            {
                "id": d["id"],
                "content": d["content"],
                "category": d["category"],
                "vector": _fake_embed(d["content"]),
            }
            for d in SAMPLE_DOCS
        ],
    )
    return LanceDBBackend(
        table="r",
        db_uri=str(tmp_path / "r.lance"),
        embed_fn=_fake_embed,
        vector_column="vector",
        text_column="content",
    )


def _duckdb_backend():
    pytest.importorskip("duckdb")
    from rag7.backend import DuckDBBackend

    b = DuckDBBackend(
        table="r",
        db_path=":memory:",
        embed_fn=_fake_embed,
        vector_column="embedding",
        content_column="content",
    )
    b._conn.execute(
        f"CREATE TABLE r (id VARCHAR, content VARCHAR, category VARCHAR, embedding FLOAT[{DIM}])"
    )
    for d in SAMPLE_DOCS:
        b._conn.execute(
            "INSERT INTO r VALUES (?, ?, ?, ?::FLOAT[8])",
            [d["id"], d["content"], d["category"], _fake_embed(d["content"])],
        )
    return b


class _RecordingBackend(InMemoryBackend):
    """Minimal sync backend — proves the reentry fix is backend-agnostic."""

    def __init__(self) -> None:
        super().__init__(documents=SAMPLE_DOCS)

    def get_index_config(self) -> IndexConfig:
        return IndexConfig()


# ── Tests ────────────────────────────────────────────────────────────────────


def _build_rag(backend) -> AgenticRAG:
    llm = _stub_llm()
    return AgenticRAG(
        "reentry",
        backend=backend,
        llm=llm,
        gen_llm=llm,
        auto_strategy=False,
        embed_fn=_fake_embed,
    )


@pytest.mark.parametrize(
    "backend_factory,needs_tmp",
    [
        (_inmemory_backend, False),
        (_chroma_backend, False),
        (_lancedb_backend, True),
        (_duckdb_backend, False),
    ],
)
def test_invoke_from_running_loop(backend_factory, needs_tmp, tmp_path):
    """`rag.invoke(q)` must not raise from inside an async context."""
    backend = backend_factory(tmp_path) if needs_tmp else backend_factory()
    rag = _build_rag(backend)

    async def outer():
        return rag.invoke("what is python?")

    state = asyncio.run(outer())
    assert state is not None


def test_retrieve_documents_from_running_loop():
    rag = _build_rag(_RecordingBackend())

    async def outer():
        return rag.retrieve_documents("programming", top_k=2)

    _, docs = asyncio.run(outer())
    assert isinstance(docs, list)


def test_batch_from_running_loop():
    rag = _build_rag(_RecordingBackend())

    async def outer():
        return rag.batch(["python", "dogs"])

    results = asyncio.run(outer())
    assert len(results) == 2
