"""Multi-collection routing tests.

Verifies that AgenticRAG can be initialised with multiple collections
and that the agent routes queries to the correct subset via an LLM
selection step. Uses InMemoryBackend (no external services) and a stub
LLM (no API calls).

Run: uv run pytest tests/test_multi_collection.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag7 import InMemoryBackend
from rag7.backend import SearchRequest, _ACTIVE_COLLECTIONS, _MultiBackend
from rag7.models import CollectionIntent

# ── Fixture documents ─────────────────────────────────────────────────────────

TECH_DOCS = [
    {"id": "t1", "content": "Python is a programming language", "category": "tech"},
    {
        "id": "t2",
        "content": "Machine learning uses neural networks",
        "category": "tech",
    },
    {"id": "t3", "content": "PostgreSQL is a relational database", "category": "tech"},
]

ANIMAL_DOCS = [
    {"id": "a1", "content": "Cats sit on mats", "category": "animals"},
    {"id": "a2", "content": "Dogs are loyal companions", "category": "animals"},
    {"id": "a3", "content": "Birds sing in the morning", "category": "animals"},
]

FOOD_DOCS = [
    {"id": "f1", "content": "Pizza originated in Italy", "cuisine": "italian"},
    {"id": "f2", "content": "Sushi is a Japanese dish", "cuisine": "japanese"},
]


def _make_multi() -> _MultiBackend:
    return _MultiBackend(
        {
            "tech": InMemoryBackend(documents=TECH_DOCS),
            "animals": InMemoryBackend(documents=ANIMAL_DOCS),
            "food": InMemoryBackend(documents=FOOD_DOCS),
        }
    )


# ── _MultiBackend unit tests ──────────────────────────────────────────────────


def test_search_fanout_tags_collection():
    mb = _make_multi()
    hits = mb.search(SearchRequest(query="python", limit=10))
    assert hits, "expected at least one hit"
    assert all("_collection" in h for h in hits)
    tech_hits = [h for h in hits if h["_collection"] == "tech"]
    assert tech_hits, "expected tech hits for 'python'"


def test_search_respects_active_contextvar():
    mb = _make_multi()
    tok = _ACTIVE_COLLECTIONS.set(["animals"])
    try:
        hits = mb.search(SearchRequest(query="python", limit=10))
    finally:
        _ACTIVE_COLLECTIONS.reset(tok)
    assert all(h["_collection"] == "animals" for h in hits)
    assert not any("Python" in str(h.get("content", "")) for h in hits)


def test_search_empty_active_falls_back_to_all():
    mb = _make_multi()
    tok = _ACTIVE_COLLECTIONS.set([])
    try:
        hits = mb.search(SearchRequest(query="python", limit=10))
    finally:
        _ACTIVE_COLLECTIONS.reset(tok)
    collections = {h["_collection"] for h in hits}
    assert "tech" in collections, "empty active should fall back to all"


def test_search_unknown_collection_ignored():
    mb = _make_multi()
    tok = _ACTIVE_COLLECTIONS.set(["ghost", "tech"])
    try:
        hits = mb.search(SearchRequest(query="python", limit=10))
    finally:
        _ACTIVE_COLLECTIONS.reset(tok)
    assert all(h["_collection"] == "tech" for h in hits)


def test_batch_search_fanout():
    mb = _make_multi()
    results = mb.batch_search(
        [
            SearchRequest(query="cats", limit=5),
            SearchRequest(query="python", limit=5),
        ]
    )
    assert len(results) == 2
    assert any(h["_collection"] == "animals" for h in results[0])
    assert any(h["_collection"] == "tech" for h in results[1])


def test_get_index_config_unions_attributes():
    mb = _MultiBackend(
        {
            "a": InMemoryBackend(documents=[{"id": "x", "title": "foo"}]),
            "b": InMemoryBackend(documents=[{"id": "y", "kind": "bar"}]),
        }
    )
    cfg = mb.get_index_config()
    assert "title" in cfg.filterable_attributes
    assert "kind" in cfg.filterable_attributes


def test_sample_documents_covers_all():
    mb = _make_multi()
    samples = mb.sample_documents(limit=30)
    cats = {s.get("category") for s in samples if "category" in s}
    assert "tech" in cats and "animals" in cats


def test_names_property():
    mb = _make_multi()
    assert set(mb.names) == {"tech", "animals", "food"}


# ── Factory integration ───────────────────────────────────────────────────────


def test_factory_accepts_list_collections():
    from rag7.factory import _build_collections_map

    backends = _build_collections_map(
        ["tech", "animals"],
        backend_str="memory",
        url=None,
        embed_fn=None,
        backend_kwargs={},
    )
    assert set(backends.keys()) == {"tech", "animals"}
    assert all(isinstance(b, InMemoryBackend) for b in backends.values())


def test_factory_accepts_dict_collections():
    from rag7.factory import _build_collections_map

    backends = _build_collections_map(
        {"tech": "programming docs", "animals": "pet facts"},
        backend_str="memory",
        url=None,
        embed_fn=None,
        backend_kwargs={},
    )
    assert set(backends.keys()) == {"tech", "animals"}


# ── Agent-level routing with stub LLM ─────────────────────────────────────────


class _StubStructuredChain:
    """Mimics `llm.with_structured_output(...)` chain — returns a canned value."""

    def __init__(self, value):
        self._value = value

    def invoke(self, _messages):
        return self._value

    async def ainvoke(self, _messages):
        return self._value


def _build_stub_llm(collection_intent: CollectionIntent):
    """Stub LLM whose `with_structured_output` returns canned Pydantic models."""
    from rag7.models import (
        FilterIntent,
        MultiQuery,
        QualityAssessment,
        ReasoningVerdict,
        RelevanceCheck,
        SearchQuery,
    )

    defaults = {
        SearchQuery: SearchQuery(
            query="", variants=[], semantic_ratio=0.5, fusion="rrf"
        ),
        QualityAssessment: QualityAssessment(sufficient=True, reason=""),
        MultiQuery: MultiQuery(queries=[]),
        FilterIntent: FilterIntent(field=None, value="", operator=""),
        CollectionIntent: collection_intent,
        RelevanceCheck: RelevanceCheck(makes_sense=False, confidence=0.0),
        ReasoningVerdict: ReasoningVerdict(),
    }

    llm = MagicMock()

    def _with_structured_output(model, **_kw):
        return _StubStructuredChain(defaults[model])

    llm.with_structured_output.side_effect = _with_structured_output
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=""))
    llm.invoke = MagicMock(return_value=MagicMock(content=""))
    return llm


def _build_agent_with_collections(collection_intent: CollectionIntent):
    from rag7.core import AgenticRAG

    backends = {
        "tech": InMemoryBackend(documents=TECH_DOCS),
        "animals": InMemoryBackend(documents=ANIMAL_DOCS),
        "food": InMemoryBackend(documents=FOOD_DOCS),
    }
    descriptions = {
        "tech": "Programming languages, databases, ML",
        "animals": "Pets, wildlife, animal facts",
        "food": "Cuisine, dishes, recipes",
    }
    stub = _build_stub_llm(collection_intent)
    return AgenticRAG(
        index="multi",
        collections=backends,
        collection_descriptions=descriptions,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )


def test_agent_exposes_collections_map():
    rag = _build_agent_with_collections(CollectionIntent(collections=["tech"]))
    assert set(rag.collections.keys()) == {"tech", "animals", "food"}


def test_agent_routes_to_selected_only():
    rag = _build_agent_with_collections(CollectionIntent(collections=["animals"]))
    selected = asyncio.run(rag._aselect_collections("tell me about cats"))
    assert selected == ["animals"]


def test_agent_routing_empty_falls_back_to_all():
    rag = _build_agent_with_collections(CollectionIntent(collections=[]))
    selected = asyncio.run(rag._aselect_collections("whatever"))
    assert set(selected) == {"tech", "animals", "food"}


def test_agent_routing_unknown_names_filtered():
    rag = _build_agent_with_collections(
        CollectionIntent(collections=["tech", "ghost", "animals"])
    )
    selected = asyncio.run(rag._aselect_collections("python and cats"))
    assert set(selected) == {"tech", "animals"}


def test_search_only_hits_selected_collection():
    rag = _build_agent_with_collections(CollectionIntent(collections=["animals"]))
    selected = asyncio.run(rag._aselect_collections("cats"))
    tok = _ACTIVE_COLLECTIONS.set(selected)
    try:
        hits = rag.backend.search(SearchRequest(query="cats", limit=10))
    finally:
        _ACTIVE_COLLECTIONS.reset(tok)
    assert hits
    assert all(h["_collection"] == "animals" for h in hits)


# ── Back-compat: single index still works ─────────────────────────────────────


def test_single_index_unchanged():
    from rag7.core import AgenticRAG

    llm = _build_stub_llm(CollectionIntent(collections=[]))
    rag = AgenticRAG(
        index="docs",
        backend=InMemoryBackend(documents=TECH_DOCS),
        llm=llm,
        gen_llm=llm,
        auto_strategy=False,
    )
    assert rag.index == "docs"
    assert rag.collections is None
    assert isinstance(rag.backend, InMemoryBackend)


@pytest.mark.parametrize(
    "collections,expected_names",
    [
        (["a", "b"], {"a", "b"}),
        ({"a": "desc a", "b": "desc b", "c": "desc c"}, {"a", "b", "c"}),
    ],
)
def test_init_agent_factory_builds_multi(collections, expected_names):
    from rag7 import init_agent

    llm = _build_stub_llm(CollectionIntent(collections=list(expected_names)))
    rag = init_agent(
        index="ignored-when-collections-given",
        collections=collections,
        backend="memory",
        llm=llm,
        gen_llm=llm,
    )
    assert set(rag.collections.keys()) == expected_names
    assert isinstance(rag.backend, _MultiBackend)
