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
    # None → disable HyDE entirely.
    hyde_min_words: int | None = Field(default=8, ge=2, le=20)

    # ── Short-query handling ─────────────────────────────────────────────────
    short_query_threshold: int = Field(default=6, ge=1, le=10)
    short_query_sort_tokens: bool = True
    # None → disable BM25 semantic fallback; always use configured ratio.
    bm25_fallback_threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    bm25_fallback_semantic_ratio: float = Field(default=0.9, ge=0.5, le=1.0)

    # ── Fast-path acceptance thresholds ──────────────────────────────────────
    # Score from BM25 keyword search above which we skip the slow-path
    # (preprocess + HyDE + filter-intent). None → never take fast path.
    fast_accept_score: float | None = Field(default=0.85, ge=0.0, le=1.0)
    # LLM relevance-check confidence required to accept the fast path when
    # the BM25 score isn't dominant. None → skip the LLM confirmation call.
    fast_accept_confidence: float | None = Field(default=0.9, ge=0.0, le=1.0)

    # ── Rerank-skip confidence (avoid reranker call on obvious hits) ─────────
    # None → never skip reranker even on obviously-matching queries.
    rerank_skip_dominance: float | None = Field(default=0.85, ge=0.0, le=1.0)
    rerank_skip_gap: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Name-field match boost ───────────────────────────────────────────────
    # After reranking, docs whose primary name_field matches query tokens
    # get a small multiplier. 0.0 disables; higher values prioritize
    # name_field matches over rerank scores. Tunable — high values lift
    # precise lookups ("trockenbeton von fixit") but can destabilize
    # paraphrase consistency.
    name_field_boost_max: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Reranker expert escalation ───────────────────────────────────────────
    expert_top_n: int = Field(default=10, ge=2, le=50)
    # None → never fire expert reranker.
    expert_threshold: float | None = Field(default=0.15, ge=0.0, le=1.0)

    # ── Pipeline stage toggles (set False to bypass the stage entirely) ──────
    enable_hyde: bool = True
    enable_filter_intent: bool = True
    enable_reasoning: bool = True
    enable_quality_gate: bool = True
    enable_preprocess_llm: bool = True

    # ── Timeouts (seconds) ───────────────────────────────────────────────────
    rerank_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    llm_timeout_s: float = Field(default=60.0, ge=1.0, le=300.0)

    # ── Agentic loop ─────────────────────────────────────────────────────────
    max_iter: int = Field(default=3, ge=1, le=20)
    n_swarm_queries: int = Field(default=4, ge=1, le=12)

    # ── Reranker doc truncation ──────────────────────────────────────────────
    rerank_chars: int = Field(default=2048, ge=256, le=16384)

    # ── LLM model tiers (``provider:model`` spec) ────────────────────────────
    # ``None`` means "use whatever is passed via kwargs or env var defaults".
    # Example specs: ``"azure:gpt-5.4"``, ``"azure:gpt-5.4-mini"``,
    # ``"azure:gpt-5.4-nano"``, ``"openai:gpt-4o-mini"``,
    # ``"anthropic:claude-haiku-4-5"``. All resolved via langchain's
    # ``init_chat_model`` (supports every langchain chat provider).
    #
    # Three tiers — tuner can mix/match them to trade cost vs. quality:
    # - ``strong_model``: quality-critical calls — final answer generation,
    #   rewrite. Use your best model here.
    # - ``weak_model``: cheap high-frequency calls — filter-intent,
    #   relevance-check, preprocess, quality-gate. Smaller/faster wins.
    # - ``thinking_model``: per-document reasoning, multi-step critique.
    #   Benefits from reasoning models (o1, sonnet-thinking) when available.
    strong_model: str | None = None
    weak_model: str | None = None
    thinking_model: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        """Build config from `RAG_*` env vars, falling back to defaults.

        Any env var set to the literal string ``none`` (case-insensitive) is
        treated as Python ``None`` for fields that support it, letting you
        disable optional stages from the environment.
        """
        def _env_float_or_none(key: str, default: str) -> float | None:
            raw = os.getenv(key, default)
            return None if raw.lower() == "none" else float(raw)

        def _env_int_or_none(key: str, default: str) -> int | None:
            raw = os.getenv(key, default)
            return None if raw.lower() == "none" else int(raw)

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
            hyde_min_words=_env_int_or_none("RAG_HYDE_MIN_WORDS", "8"),
            short_query_threshold=int(os.getenv("RAG_SHORT_QUERY_THRESHOLD", "6")),
            short_query_sort_tokens=bool(int(os.getenv("RAG_SHORT_QUERY_SORT_TOKENS", "1"))),
            bm25_fallback_threshold=_env_float_or_none("RAG_BM25_FALLBACK_THRESHOLD", "0.4"),
            bm25_fallback_semantic_ratio=float(
                os.getenv("RAG_BM25_FALLBACK_SEMANTIC_RATIO", "0.9")
            ),
            fast_accept_score=_env_float_or_none("RAG_FAST_ACCEPT_SCORE", "0.85"),
            fast_accept_confidence=_env_float_or_none(
                "RAG_FAST_ACCEPT_CONFIDENCE", "0.9"
            ),
            rerank_skip_dominance=_env_float_or_none(
                "RAG_RERANK_SKIP_DOMINANCE", "0.85"
            ),
            rerank_skip_gap=float(os.getenv("RAG_RERANK_SKIP_GAP", "0.1")),
            name_field_boost_max=float(os.getenv("RAG_NAME_FIELD_BOOST_MAX", "0.1")),
            expert_top_n=int(os.getenv("RAG_EXPERT_TOP_N", "10")),
            expert_threshold=_env_float_or_none("RAG_EXPERT_THRESHOLD", "0.15"),
            enable_hyde=bool(int(os.getenv("RAG_ENABLE_HYDE", "1"))),
            enable_filter_intent=bool(int(os.getenv("RAG_ENABLE_FILTER_INTENT", "1"))),
            enable_reasoning=bool(int(os.getenv("RAG_ENABLE_REASONING", "1"))),
            enable_quality_gate=bool(int(os.getenv("RAG_ENABLE_QUALITY_GATE", "1"))),
            enable_preprocess_llm=bool(
                int(os.getenv("RAG_ENABLE_PREPROCESS_LLM", "1"))
            ),
            rerank_timeout_s=float(os.getenv("RAG_RERANK_TIMEOUT_S", "30.0")),
            llm_timeout_s=float(os.getenv("RAG_LLM_TIMEOUT_S", "60.0")),
            max_iter=int(os.getenv("RAG_MAX_ITER", "3")),
            n_swarm_queries=int(os.getenv("RAG_N_SWARM_QUERIES", "4")),
            rerank_chars=int(os.getenv("RAG_RERANK_CHARS", "2048")),
            strong_model=os.getenv("RAG_STRONG_MODEL") or None,
            weak_model=os.getenv("RAG_WEAK_MODEL") or None,
            thinking_model=os.getenv("RAG_THINKING_MODEL") or None,
        )

    @classmethod
    def _apply_disable(cls, section: dict[str, Any]) -> dict[str, Any]:
        """Move ``disable = [...]`` entries into explicit None values."""
        disabled = section.pop("disable", None) or []
        for name in disabled:
            section[name] = None
        return section

    @classmethod
    def from_toml(cls, path: str | Path = "rag7.config.toml") -> Self:
        """Load config from a TOML file. Expects a `[rag7]` table.

        Fields listed in ``disable = [...]`` are set to ``None`` after
        loading — this is how TOML (which has no null) expresses
        "disable this stage".
        """
        data = tomllib.loads(Path(path).read_text())
        section = dict(data.get("rag7", data))
        return cls(**cls._apply_disable(section))

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
        return cls(**cls._apply_disable(dict(section)))

    @classmethod
    def auto(
        cls,
        pyproject_path: str | Path = "pyproject.toml",
        toml_path: str | Path = "rag7.config.toml",
    ) -> Self:
        """Discover config in priority order:

        1. ``rag7.config.toml`` (local / per-deployment override — gitignored)
        2. ``[tool.rag7]`` in pyproject.toml (shared / shipped default,
           committed so teams & library users get sensible tuned values)
        3. ``RAG_*`` env vars
        4. Library defaults

        The local TOML file wins over the shipped pyproject defaults so users
        can override without touching the committed config.
        """
        if Path(toml_path).is_file():
            return cls.from_toml(toml_path)
        pp = cls.from_pyproject(pyproject_path)
        if pp is not None:
            return pp
        return cls.from_env()

    def _toml_body(self, table_header: str) -> str:
        """Serialize to TOML. ``None`` is represented two ways:

        - If the field's default is also ``None``, the field is **omitted**
          (it's effectively unset — matches the default).
        - If the field's default is a concrete value but the current value
          is ``None``, the field name is added to a ``disable = [...]``
          array — this means "explicitly disable this stage on load".
        """
        defaults = type(self)().model_dump()
        lines = [table_header]
        disabled: list[str] = []
        for name, value in self.model_dump().items():
            if value is None:
                if defaults.get(name) is not None:
                    disabled.append(name)
                continue
            if isinstance(value, bool):
                lines.append(f"{name} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{name} = {value}")
            else:
                lines.append(f'{name} = "{value}"')
        if disabled:
            quoted = ", ".join(f'"{d}"' for d in disabled)
            lines.append("# Explicitly disabled stages (would default to a value):")
            lines.append(f"disable = [{quoted}]")
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
