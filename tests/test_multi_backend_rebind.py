"""Regression test: `_MultiBackend` must rebind `index_uid` per sub-backend.

Bug: when the caller's SearchRequest carried ``index_uid=<primary>``, every
sub-backend received it and queried the primary index instead of its own,
which made multi-collection search silently return only primary-collection
hits (or nothing when the primary backend was unavailable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag7.backend import IndexConfig, SearchRequest
from rag7.backend import SearchBackend, _MultiBackend


@dataclass
class _CapturingBackend(SearchBackend):
    """Records every SearchRequest it received on batch_search."""

    name: str
    received: list[SearchRequest] = field(default_factory=list)

    def search(self, request: SearchRequest) -> list[dict]:
        return []

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        self.received.extend(requests)
        return [[{"id": self.name, "_rankingScore": 0.5}] for _ in requests]

    def get_index_config(self) -> IndexConfig:
        return IndexConfig()

    def sample_documents(self, limit: int = 100) -> list[dict]:
        return []


def test_multi_backend_strips_foreign_index_uid() -> None:
    de = _CapturingBackend(name="de")
    fr = _CapturingBackend(name="fr")
    multi = _MultiBackend({"de": de, "fr": fr})

    # Request targets primary "de". Must not leak to fr.
    req = SearchRequest(query="fixit 516", limit=5, index_uid="de")
    out = multi.batch_search([req])

    assert len(out) == 1
    hits = out[0]
    assert {h["id"] for h in hits} == {"de", "fr"}, "both collections must respond"

    # de keeps its index_uid (matches own name); fr got it nulled out.
    assert de.received[0].index_uid == "de"
    assert fr.received[0].index_uid is None


def test_multi_backend_tolerates_short_batch_response() -> None:
    """One sub-backend returning fewer rows than requests must not crash."""

    class _Bad(_CapturingBackend):
        def batch_search(self, requests):
            return []  # shorter than requests

    de = _CapturingBackend(name="de")
    bad = _Bad(name="bad")
    multi = _MultiBackend({"de": de, "bad": bad})

    reqs = [SearchRequest(query=q, limit=2) for q in ("a", "b")]
    out = multi.batch_search(reqs)
    assert len(out) == 2
    # de still contributes; bad silently skipped.
    assert all(h["id"] == "de" for hits in out for h in hits)
