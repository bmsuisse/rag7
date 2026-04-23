from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field


class RAGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, ge=1, le=50)
    retrieval_factor: int = Field(default=4, ge=1, le=16)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    rerank_cap_multiplier: float = Field(default=2.0, ge=1.0, le=8.0)
    semantic_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion: Literal["rrf", "dbsf"] = "rrf"

    hyde_min_words: int | None = Field(default=8, ge=2, le=20)

    short_query_threshold: int = Field(default=6, ge=1, le=10)
    short_query_sort_tokens: bool = True

    bm25_fallback_threshold: float | None = Field(default=0.4, ge=0.0, le=1.0)
    bm25_fallback_semantic_ratio: float = Field(default=0.9, ge=0.5, le=1.0)

    fast_accept_score: float | None = Field(default=0.85, ge=0.0, le=1.0)

    fast_accept_confidence: float | None = Field(default=0.9, ge=0.0, le=1.0)

    rerank_min_score: float | None = Field(default=0.2, ge=0.0, le=1.0)

    query_languages: list[str] = Field(default_factory=lambda: ["de", "fr", "it", "en"])

    name_field_boost_max: float = Field(default=0.1, ge=0.0, le=1.0)

    boost_decay_sigma: float = Field(default=0.05, ge=0.001, le=0.5)

    enable_close_match_grader: bool = True
    close_match_strictness: Literal["loose", "balanced", "strict"] = "loose"

    expert_top_n: int = Field(default=10, ge=2, le=50)

    expert_threshold: float | None = Field(default=0.15, ge=0.0, le=1.0)

    enable_hyde: bool = True
    enable_filter_intent: bool = True
    enable_reasoning: bool = True
    enable_quality_gate: bool = True
    enable_preprocess_llm: bool = True
    enable_final_grade: bool = False
    final_grade_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    enable_swarm_grade: bool = False

    rerank_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    llm_timeout_s: float = Field(default=60.0, ge=1.0, le=300.0)

    max_iter: int = Field(default=3, ge=1, le=20)
    n_swarm_queries: int = Field(default=4, ge=1, le=12)

    rerank_chars: int = Field(default=8192, ge=256, le=32768)

    preview_chars: int = Field(default=1500, ge=200, le=4000)

    strong_model: str | None = None
    weak_model: str | None = None
    thinking_model: str | None = None
    grader_model: str | None = None

    # Free-text rules appended to preprocess / filter-intent / close-match
    # prompts. Admins can set this globally under `[rag7]` or per-collection
    # under `[rag7.collections.<name>]` to further tune retrieval.
    custom_instructions: str = ""

    @classmethod
    def from_env(cls) -> Self:

        def _env_float_or_none(key: str, default: str) -> float | None:
            raw = os.getenv(key, default)
            return None if raw.lower() == "none" else float(raw)

        def _env_int_or_none(key: str, default: str) -> int | None:
            raw = os.getenv(key, default)
            return None if raw.lower() == "none" else int(raw)

        def _env_strictness(raw: str) -> Literal["loose", "balanced", "strict"]:
            v = raw.strip().lower()
            if v == "balanced":
                return "balanced"
            if v == "strict":
                return "strict"
            return "loose"

        fusion = os.getenv("RAG_FUSION", "rrf")
        if fusion not in ("rrf", "dbsf"):
            fusion = "rrf"
        return cls(
            top_k=int(os.getenv("RAG_TOP_K", "10")),
            retrieval_factor=int(os.getenv("RAG_RETRIEVAL_FACTOR", "4")),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "5")),
            rerank_cap_multiplier=float(os.getenv("RAG_RERANK_CAP_MULTIPLIER", "2.0")),
            semantic_ratio=float(os.getenv("RAG_SEMANTIC_RATIO", "0.5")),
            fusion=fusion,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            hyde_min_words=_env_int_or_none("RAG_HYDE_MIN_WORDS", "8"),
            short_query_threshold=int(os.getenv("RAG_SHORT_QUERY_THRESHOLD", "6")),
            short_query_sort_tokens=bool(
                int(os.getenv("RAG_SHORT_QUERY_SORT_TOKENS", "1"))
            ),
            bm25_fallback_threshold=_env_float_or_none(
                "RAG_BM25_FALLBACK_THRESHOLD", "0.4"
            ),
            bm25_fallback_semantic_ratio=float(
                os.getenv("RAG_BM25_FALLBACK_SEMANTIC_RATIO", "0.9")
            ),
            fast_accept_score=_env_float_or_none("RAG_FAST_ACCEPT_SCORE", "0.85"),
            fast_accept_confidence=_env_float_or_none(
                "RAG_FAST_ACCEPT_CONFIDENCE", "0.9"
            ),
            rerank_min_score=_env_float_or_none("RAG_RERANK_MIN_SCORE", "0.2"),
            name_field_boost_max=float(os.getenv("RAG_NAME_FIELD_BOOST_MAX", "0.1")),
            boost_decay_sigma=float(os.getenv("RAG_BOOST_DECAY_SIGMA", "0.05")),
            enable_close_match_grader=bool(
                int(os.getenv("RAG_ENABLE_CLOSE_MATCH_GRADER", "1"))
            ),
            close_match_strictness=_env_strictness(
                os.getenv("RAG_CLOSE_MATCH_STRICTNESS", "loose")
            ),
            expert_top_n=int(os.getenv("RAG_EXPERT_TOP_N", "10")),
            expert_threshold=_env_float_or_none("RAG_EXPERT_THRESHOLD", "0.15"),
            enable_hyde=bool(int(os.getenv("RAG_ENABLE_HYDE", "1"))),
            enable_filter_intent=bool(int(os.getenv("RAG_ENABLE_FILTER_INTENT", "1"))),
            enable_reasoning=bool(int(os.getenv("RAG_ENABLE_REASONING", "1"))),
            enable_quality_gate=bool(int(os.getenv("RAG_ENABLE_QUALITY_GATE", "1"))),
            enable_preprocess_llm=bool(
                int(os.getenv("RAG_ENABLE_PREPROCESS_LLM", "1"))
            ),
            enable_final_grade=bool(int(os.getenv("RAG_ENABLE_FINAL_GRADE", "0"))),
            final_grade_threshold=float(os.getenv("RAG_FINAL_GRADE_THRESHOLD", "0.9")),
            enable_swarm_grade=bool(int(os.getenv("RAG_ENABLE_SWARM_GRADE", "0"))),
            rerank_timeout_s=float(os.getenv("RAG_RERANK_TIMEOUT_S", "30.0")),
            llm_timeout_s=float(os.getenv("RAG_LLM_TIMEOUT_S", "60.0")),
            max_iter=int(os.getenv("RAG_MAX_ITER", "3")),
            n_swarm_queries=int(os.getenv("RAG_N_SWARM_QUERIES", "4")),
            rerank_chars=int(os.getenv("RAG_RERANK_CHARS", "8192")),
            preview_chars=int(os.getenv("RAG_PREVIEW_CHARS", "1500")),
            strong_model=os.getenv("RAG_STRONG_MODEL") or None,
            weak_model=os.getenv("RAG_WEAK_MODEL") or None,
            thinking_model=os.getenv("RAG_THINKING_MODEL") or None,
            grader_model=os.getenv("RAG_GRADER_MODEL") or None,
            custom_instructions=os.getenv("RAG_CUSTOM_INSTRUCTIONS", ""),
            query_languages=[
                lang.strip()
                for lang in os.getenv("RAG_QUERY_LANGUAGES", "de,fr,it,en").split(",")
                if lang.strip()
            ],
        )

    @classmethod
    def _apply_disable(cls, section: dict[str, Any]) -> dict[str, Any]:
        disabled = section.pop("disable", None) or []
        for name in disabled:
            section[name] = None
        return section

    @classmethod
    def from_toml(
        cls,
        path: str | Path = "rag7.config.toml",
        collection: str | None = None,
    ) -> Self:
        data = tomllib.loads(Path(path).read_text())
        section = dict(data.get("rag7", data))
        collections_data: dict[str, Any] = section.pop("collections", {})
        global_cfg = cls(**cls._apply_disable(section))
        if collection is None:
            return global_cfg
        col_section = dict(collections_data.get(collection, {}))
        if not col_section:
            return global_cfg
        merged = global_cfg.model_dump()
        merged.update(cls._apply_disable(col_section))
        return cls(**merged)

    @classmethod
    def collection_configs_from_toml(
        cls, path: str | Path = "rag7.config.toml"
    ) -> dict[str, "RAGConfig"]:
        """Return per-collection RAGConfig instances (global config merged with per-collection overrides)."""
        data = tomllib.loads(Path(path).read_text())
        section = dict(data.get("rag7", data))
        collections_data: dict[str, Any] = section.pop("collections", {})
        global_cfg = cls(**cls._apply_disable(section))
        result: dict[str, RAGConfig] = {}
        for col_name, col_section in collections_data.items():
            merged = global_cfg.model_dump()
            merged.update(cls._apply_disable(dict(col_section)))
            result[col_name] = cls(**merged)
        return result

    @classmethod
    def from_pyproject(cls, path: str | Path = "pyproject.toml") -> Self | None:
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
        collection: str | None = None,
    ) -> Self:
        if Path(toml_path).is_file():
            return cls.from_toml(toml_path, collection=collection)
        pp = cls.from_pyproject(pyproject_path)
        if pp is not None:
            return pp
        return cls.from_env()

    def _toml_body(
        self, table_header: str, reference: "RAGConfig | None" = None
    ) -> str:
        ref = (
            reference.model_dump()
            if reference is not None
            else type(self)().model_dump()
        )
        lines = [table_header]
        disabled: list[str] = []
        for name, value in self.model_dump().items():
            if reference is not None and value == ref.get(name):
                continue  # skip fields identical to reference (for collection diffs)
            if value is None:
                if ref.get(name) is not None:
                    disabled.append(name)
                continue
            if isinstance(value, bool):
                lines.append(f"{name} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{name} = {value}")
            elif isinstance(value, str) and ("\n" in value or '"' in value):
                # Use TOML triple-quoted literal string for multiline / quoted text
                lines.append(f"{name} = '''\n{value}\n'''")
            else:
                lines.append(f'{name} = "{value}"')
        if disabled:
            quoted = ", ".join(f'"{d}"' for d in disabled)
            lines.append("# Explicitly disabled stages (would default to a value):")
            lines.append(f"disable = [{quoted}]")
        return "\n".join(lines) + "\n"

    def save_toml(self, path: str | Path = "rag7.config.toml") -> None:
        p = Path(path)
        new_global = self._toml_body("[rag7]")
        if not p.is_file():
            p.write_text(new_global)
            return
        # Preserve any [rag7.collections.*] sections that follow the global block
        lines = p.read_text().splitlines(keepends=True)
        col_start = next(
            (
                i
                for i, ln in enumerate(lines)
                if ln.strip().startswith("[rag7.collections.")
            ),
            None,
        )
        if col_start is None:
            p.write_text(new_global)
        else:
            p.write_text(new_global + "\n" + "".join(lines[col_start:]))

    def save_collection_toml(
        self, collection: str, path: str | Path = "rag7.config.toml"
    ) -> None:
        """Upsert a [rag7.collections.COLLECTION] section with only the fields that differ from the global config."""
        p = Path(path)
        global_cfg = type(self).from_toml(p) if p.is_file() else type(self)()
        header = f"[rag7.collections.{collection}]"
        new_section = self._toml_body(header, reference=global_cfg)
        if not p.is_file():
            p.write_text(new_section)
            return
        text = p.read_text()
        lines = text.splitlines(keepends=True)
        # Find existing section for this collection
        start = next(
            (i for i, ln in enumerate(lines) if ln.strip() == header),
            None,
        )
        if start is None:
            p.write_text(text.rstrip() + "\n\n" + new_section)
            return
        # Find end of this section (next section header or EOF)
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].strip().startswith("[")
            ),
            len(lines),
        )
        p.write_text("".join(lines[:start]) + new_section + "\n" + "".join(lines[end:]))

    def save_pyproject(self, path: str | Path = "pyproject.toml") -> None:
        p = Path(path)
        text = p.read_text() if p.is_file() else ""
        new_body = self._toml_body("[tool.rag7]")

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
        defaults = type(self)()
        current = self.model_dump()
        return {k: v for k, v in current.items() if v != getattr(defaults, k)}
