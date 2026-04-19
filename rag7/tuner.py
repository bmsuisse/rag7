"""Optuna-based tuner for `RAGConfig`.

Users bring: a list of (query, expected_ids, id_field) hit cases,
a backend factory, an embed_fn. The tuner searches the config space
with optuna's TPE sampler and returns the best config for their corpus.

Install: `pip install rag7[tune]` (adds optuna).

Example:
    from rag7 import AgenticRAG, MeilisearchBackend
    from rag7.tuner import RAGTuner

    tuner = RAGTuner(
        backend_factory=lambda: MeilisearchBackend("my_index"),
        embed_fn=my_embed_fn,
        hit_cases=[("kettle 1.7L", ["SKU-123"], "sku"), ...],
    )
    best_config = tuner.tune(n_trials=50)
    best_config.save_toml("rag7.config.toml")
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RAGConfig


HitCase = tuple[str, list[str], str]  # (query, expected_ids, id_field)


def load_testset(path: str | Path) -> list[HitCase]:
    """Load a testset from JSON or JSONL.

    Expected format (either):
        [{"query": "...", "expected_ids": ["id1"], "id_field": "id"}, ...]
        OR one object per line (JSONL).

    Returns a list of `(query, expected_ids, id_field)` tuples.
    """
    p = Path(path)
    text = p.read_text()
    entries: list[dict[str, Any]]
    if p.suffix == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        entries = json.loads(text)
    return [
        (e["query"], [str(x) for x in e["expected_ids"]], e.get("id_field", "id"))
        for e in entries
    ]


@dataclass
class TrialResult:
    config: RAGConfig
    score: float
    hit_rate: float
    mean_latency: float
    n_hits: int
    n_total: int


@dataclass
class RAGTuner:
    """Tune `RAGConfig` on your hit cases with optuna.

    Parameters
    ----------
    backend_factory:
        Zero-arg callable returning a fresh `SearchBackend` per trial.
    embed_fn:
        Embedding function for the agent.
    hit_cases:
        List of `(query, expected_ids, id_field)` — hit@k is computed as
        the fraction of queries where any `expected_id` appears in top-k.
    eval_k:
        Top-k cutoff for hit rate computation (default 5).
    latency_weight:
        0–1 blend for latency in the objective. 0 = ignore latency,
        1 = latency dominates. Default 0.15.
    latency_budget_ms:
        Queries slower than this (in ms) are penalized linearly.
    llm / gen_llm / reranker:
        Forwarded to `AgenticRAG` unchanged. Use small/cheap models during
        tuning to keep cost down.
    """

    backend_factory: Callable[[], Any]
    embed_fn: Callable[[str], list[float]]
    hit_cases: Sequence[HitCase]
    eval_k: int = 5
    latency_weight: float = 0.15
    latency_budget_ms: float = 2000.0
    llm: Any = None
    gen_llm: Any = None
    reranker: Any = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_agent(self, config: RAGConfig) -> Any:
        from .core import AgenticRAG

        backend = self.backend_factory()
        index_name = getattr(backend, "index", None) or "tuned"
        return AgenticRAG(
            index=index_name,
            backend=backend,
            embed_fn=self.embed_fn,
            config=config,
            llm=self.llm,
            gen_llm=self.gen_llm,
            reranker=self.reranker,
            auto_strategy=False,
            **self.extra_kwargs,
        )

    async def _score_config(self, config: RAGConfig) -> TrialResult:
        import time

        rag = self._build_agent(config)
        sem = asyncio.Semaphore(10)
        hits = 0
        latencies: list[float] = []

        async def one(case: HitCase) -> bool:
            query, expected, field_name = case
            async with sem:
                t0 = time.perf_counter()
                _, docs = await rag._aretrieve_documents(query, top_k=self.eval_k)
                latencies.append((time.perf_counter() - t0) * 1000)
            retrieved = [str(d.metadata.get(field_name, "")) for d in docs]
            return any(str(e) in retrieved for e in expected)

        results = await asyncio.gather(*(one(c) for c in self.hit_cases))
        hits = sum(results)
        total = len(self.hit_cases)
        hit_rate = hits / total if total else 0.0

        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        latency_penalty = max(0.0, mean_latency / self.latency_budget_ms - 1.0)
        score = hit_rate * (1.0 - self.latency_weight) + (
            (1.0 - min(latency_penalty, 1.0)) * self.latency_weight
        )

        return TrialResult(
            config=config,
            score=score,
            hit_rate=hit_rate,
            mean_latency=mean_latency,
            n_hits=hits,
            n_total=total,
        )

    def tune(
        self,
        *,
        n_trials: int = 50,
        seed: int = 42,
        show_progress: bool = True,
        baseline_config: RAGConfig | None = None,
    ) -> RAGConfig:
        """Run optuna search and return the best config.

        Always enqueues the baseline (default or provided) as trial 0 so
        tuning can only improve over the starting point.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Install optuna to use RAGTuner: `pip install rag7[tune]` "
                "or `pip install optuna`."
            ) from exc

        baseline = baseline_config or RAGConfig()

        def objective(trial: "optuna.Trial") -> float:
            config = RAGConfig(
                top_k=trial.suggest_int("top_k", 5, 20),
                retrieval_factor=trial.suggest_int("retrieval_factor", 2, 8),
                rerank_top_n=trial.suggest_int("rerank_top_n", 3, 10),
                rerank_cap_multiplier=trial.suggest_float(
                    "rerank_cap_multiplier", 1.5, 4.0
                ),
                semantic_ratio=trial.suggest_float("semantic_ratio", 0.3, 0.9),
                fusion=trial.suggest_categorical("fusion", ["rrf", "dbsf"]),
                hyde_min_words=trial.suggest_int("hyde_min_words", 4, 12),
                short_query_threshold=trial.suggest_int("short_query_threshold", 3, 8),
                short_query_sort_tokens=trial.suggest_categorical(
                    "short_query_sort_tokens", [True, False]
                ),
                bm25_fallback_threshold=trial.suggest_float(
                    "bm25_fallback_threshold", 0.2, 0.6
                ),
                bm25_fallback_semantic_ratio=trial.suggest_float(
                    "bm25_fallback_semantic_ratio", 0.7, 1.0
                ),
                expert_threshold=trial.suggest_float("expert_threshold", 0.05, 0.3),
            )
            result = asyncio.run(self._score_config(config))
            trial.set_user_attr("hit_rate", result.hit_rate)
            trial.set_user_attr("mean_latency_ms", result.mean_latency)
            trial.set_user_attr("n_hits", result.n_hits)
            return result.score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Seed with baseline so tuning dominates it
        study.enqueue_trial(
            {
                k: v
                for k, v in baseline.model_dump().items()
                if k
                in {
                    "top_k",
                    "retrieval_factor",
                    "rerank_top_n",
                    "rerank_cap_multiplier",
                    "semantic_ratio",
                    "fusion",
                    "hyde_min_words",
                    "short_query_threshold",
                    "short_query_sort_tokens",
                    "bm25_fallback_threshold",
                    "bm25_fallback_semantic_ratio",
                    "expert_threshold",
                }
            }
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

        best_params = study.best_params
        # Merge best params into a full RAGConfig (non-tuned fields from baseline)
        merged = baseline.model_dump() | best_params
        return RAGConfig(**merged)
