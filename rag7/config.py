"""Centralized configuration for AgenticRAG.

Discovery order (standard Python tooling convention):

    1. Runtime:   ``AgenticRAG(config=RAGConfig(...))``
    2. Project:   ``[tool.rag7]`` table in ``pyproject.toml``
    3. Dedicated: ``rag7.config.toml`` with a ``[rag7]`` table
    4. Env vars:  ``RAG_*`` (good for containers/CI)
    5. Defaults

Use ``RAGConfig.auto()`` for automatic discovery, or the explicit loaders
(``from_pyproject``, ``from_toml``, ``from_env``) to control precedence.

Tune for your own corpus with ``RAGTuner`` (install ``rag7[tune]``).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class RAGConfig(BaseModel):
    """Tunable parameters for retrieval, reranking, and generation.

    Defaults match current `AgenticRAG` behavior. Override via kwargs,
    `from_toml`, `from_env`, or `RAGTuner`.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = Field(default=10, ge=1, le=50)
    retrieval_factor: int = Field(default=4, ge=1, le=16)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    rerank_cap_multiplier: float = Field(default=2.0, ge=1.0, le=8.0)
    semantic_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion: Literal["rrf", "dbsf"] = "rrf"

    # ── HyDE (hypothetical document expansion) ───────────────────────────────
    hyde_min_words: int = Field(default=8, ge=2, le=20)

    # ── Short-query handling ─────────────────────────────────────────────────
    short_query_threshold: int = Field(default=6, ge=1, le=10)
    short_query_sort_tokens: bool = True
    bm25_fallback_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    bm25_fallback_semantic_ratio: float = Field(default=0.9, ge=0.5, le=1.0)

    # ── Reranker expert escalation ───────────────────────────────────────────
    expert_top_n: int = Field(default=10, ge=2, le=50)
    expert_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    # ── Agentic loop ─────────────────────────────────────────────────────────
    max_iter: int = Field(default=3, ge=1, le=20)
    n_swarm_queries: int = Field(default=4, ge=1, le=12)

    # ── Reranker doc truncation ──────────────────────────────────────────────
    rerank_chars: int = Field(default=2048, ge=256, le=16384)

    @classmethod
    def from_env(cls) -> Self:
        """Build config from `RAG_*` env vars, falling back to defaults."""
        fusion = os.getenv("RAG_FUSION", "rrf")
        if fusion not in ("rrf", "dbsf"):
            fusion = "rrf"
        return cls(
            top_k=int(os.getenv("RAG_TOP_K", "10")),
            retrieval_factor=int(os.getenv("RAG_RETRIEVAL_FACTOR", "4")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "5")),
            rerank_cap_multiplier=float(os.getenv("RAG_RERANK_CAP_MULTIPLIER", "2.0")),
            semantic_ratio=float(os.getenv("RAG_SEMANTIC_RATIO", "0.5")),
            fusion=fusion,  # type: ignore[arg-type]
            hyde_min_words=int(os.getenv("RAG_HYDE_MIN_WORDS", "8")),
            short_query_threshold=int(os.getenv("RAG_SHORT_QUERY_THRESHOLD", "6")),
            short_query_sort_tokens=bool(int(os.getenv("RAG_SHORT_QUERY_SORT_TOKENS", "1"))),
            bm25_fallback_threshold=float(os.getenv("RAG_BM25_FALLBACK_THRESHOLD", "0.4")),
            bm25_fallback_semantic_ratio=float(
                os.getenv("RAG_BM25_FALLBACK_SEMANTIC_RATIO", "0.9")
            ),
            expert_top_n=int(os.getenv("RAG_EXPERT_TOP_N", "10")),
            expert_threshold=float(os.getenv("RAG_EXPERT_THRESHOLD", "0.15")),
            max_iter=int(os.getenv("RAG_MAX_ITER", "3")),
            n_swarm_queries=int(os.getenv("RAG_N_SWARM_QUERIES", "4")),
            rerank_chars=int(os.getenv("RAG_RERANK_CHARS", "2048")),
        )

    @classmethod
    def from_toml(cls, path: str | Path = "rag7.config.toml") -> Self:
        """Load config from a TOML file. Expects a `[rag7]` table."""
        data = tomllib.loads(Path(path).read_text())
        section = data.get("rag7", data)
        return cls(**section)

    @classmethod
    def from_pyproject(cls, path: str | Path = "pyproject.toml") -> Self | None:
        """Load config from `[tool.rag7]` in pyproject.toml.

        Returns None if pyproject.toml is missing or has no [tool.rag7] section.
        Matches the standard Python tooling convention (ruff, black, mypy).
        """
        p = Path(path)
        if not p.is_file():
            return None
        data = tomllib.loads(p.read_text())
        section = data.get("tool", {}).get("rag7")
        if section is None:
            return None
        return cls(**section)

    @classmethod
    def auto(
        cls,
        pyproject_path: str | Path = "pyproject.toml",
        toml_path: str | Path = "rag7.config.toml",
    ) -> Self:
        """Discover config in priority order:

        1. ``[tool.rag7]`` in pyproject.toml (standard Python convention)
        2. rag7.config.toml (dedicated file)
        3. RAG_* env vars
        4. defaults
        """
        pp = cls.from_pyproject(pyproject_path)
        if pp is not None:
            return pp
        if Path(toml_path).is_file():
            return cls.from_toml(toml_path)
        return cls.from_env()

    def _toml_body(self, table_header: str) -> str:
        lines = [table_header]
        for name, value in self.model_dump().items():
            if isinstance(value, bool):
                lines.append(f"{name} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{name} = {value}")
            else:
                lines.append(f'{name} = "{value}"')
        return "\n".join(lines) + "\n"

    def save_toml(self, path: str | Path = "rag7.config.toml") -> None:
        """Persist config to a dedicated TOML file under `[rag7]`."""
        Path(path).write_text(self._toml_body("[rag7]"))

    def save_pyproject(self, path: str | Path = "pyproject.toml") -> None:
        """Write/overwrite the ``[tool.rag7]`` section of pyproject.toml in place.

        Preserves everything else in the file; replaces only the rag7 table.
        """
        p = Path(path)
        text = p.read_text() if p.is_file() else ""
        new_body = self._toml_body("[tool.rag7]")
        # Strip any existing [tool.rag7] block (header + following contiguous
        # key=value lines, stopping at next header or blank-line boundary).
        lines = text.splitlines()
        out: list[str] = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if stripped == "[tool.rag7]":
                skip = True
                continue
            if skip:
                if stripped.startswith("[") and stripped.endswith("]"):
                    skip = False
                else:
                    continue
            out.append(line)
        cleaned = "\n".join(out).rstrip() + ("\n\n" if out else "")
        p.write_text(cleaned + new_body)

    def overrides(self) -> dict[str, Any]:
        """Return the subset of fields that differ from defaults (for logging)."""
        defaults = type(self)()
        current = self.model_dump()
        return {k: v for k, v in current.items() if v != getattr(defaults, k)}
