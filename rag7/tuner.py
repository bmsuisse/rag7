from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RAGConfig

HitCase = tuple[str, list[str], str]


def load_testset(path: str | Path) -> list[HitCase]:
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


class _EarlyStopCallback:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best: float | None = None
        self._no_improve = 0

    def __call__(self, study: Any, trial: Any) -> None:
        current = study.best_value
        if self._best is None or current > self._best:
            self._best = current
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            study.stop()


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

    candidate_models: Sequence[str] = field(default_factory=list)

    tune_strong_model: bool = False

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

    _NONEABLE_FIELDS = frozenset(
        {
            "bm25_fallback_threshold",
            "fast_accept_score",
            "fast_accept_confidence",
            "rerank_min_score",
            "expert_threshold",
            "hyde_min_words",
        }
    )

    def tune(
        self,
        *,
        n_trials: int = 50,
        seed: int = 42,
        show_progress: bool = True,
        baseline_config: RAGConfig | None = None,
        patience: int | None = 10,
        trial_timeout_s: float = 180.0,
    ) -> RAGConfig:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Install optuna to use RAGTuner: `pip install rag7[tune]` "
                "or `pip install optuna`."
            ) from exc

        baseline = baseline_config or RAGConfig()

        def _opt_float(
            trial: "optuna.Trial", name: str, low: float, high: float
        ) -> float | None:
            if trial.suggest_categorical(f"{name}_enabled", [True, False]):
                return trial.suggest_float(name, low, high)
            return None

        def _opt_int(
            trial: "optuna.Trial", name: str, low: int, high: int
        ) -> int | None:
            if trial.suggest_categorical(f"{name}_enabled", [True, False]):
                return trial.suggest_int(name, low, high)
            return None

        def objective(trial: "optuna.Trial") -> float:
            config = RAGConfig(
                top_k=trial.suggest_int("top_k", 5, 20),
                retrieval_factor=trial.suggest_int("retrieval_factor", 2, 8),
                rerank_top_n=trial.suggest_int("rerank_top_n", 3, 10),
                rerank_cap_multiplier=trial.suggest_float(
                    "rerank_cap_multiplier", 1.5, 4.0
                ),
                semantic_ratio=trial.suggest_float("semantic_ratio", 0.3, 0.9),
                fusion=trial.suggest_categorical("fusion", ["rrf", "dbsf"]),  # ty: ignore[invalid-argument-type]
                hyde_min_words=_opt_int(trial, "hyde_min_words", 4, 12),
                short_query_threshold=trial.suggest_int("short_query_threshold", 3, 8),
                short_query_sort_tokens=trial.suggest_categorical(
                    "short_query_sort_tokens", [True, False]
                ),
                bm25_fallback_threshold=_opt_float(
                    trial, "bm25_fallback_threshold", 0.2, 0.6
                ),
                bm25_fallback_semantic_ratio=trial.suggest_float(
                    "bm25_fallback_semantic_ratio", 0.7, 1.0
                ),
                fast_accept_score=_opt_float(trial, "fast_accept_score", 0.5, 0.95),
                fast_accept_confidence=_opt_float(
                    trial, "fast_accept_confidence", 0.6, 0.95
                ),
                rerank_min_score=_opt_float(trial, "rerank_min_score", 0.05, 0.5),
                name_field_boost_max=trial.suggest_float(
                    "name_field_boost_max", 0.0, 0.5
                ),
                boost_decay_sigma=trial.suggest_float(
                    "boost_decay_sigma", 0.01, 0.15
                ),
                expert_threshold=_opt_float(trial, "expert_threshold", 0.05, 0.3),
                enable_hyde=trial.suggest_categorical("enable_hyde", [True, False]),
                enable_filter_intent=trial.suggest_categorical(
                    "enable_filter_intent", [True, False]
                ),
                enable_quality_gate=trial.suggest_categorical(
                    "enable_quality_gate", [True, False]
                ),
                enable_preprocess_llm=trial.suggest_categorical(
                    "enable_preprocess_llm", [True, False]
                ),
                strong_model=(
                    trial.suggest_categorical(
                        "strong_model", list(self.candidate_models)
                    )
                    if self.candidate_models and self.tune_strong_model
                    else baseline.strong_model
                ),
                weak_model=(
                    trial.suggest_categorical("weak_model", list(self.candidate_models))
                    if self.candidate_models
                    else baseline.weak_model
                ),
                thinking_model=(
                    trial.suggest_categorical(
                        "thinking_model", list(self.candidate_models)
                    )
                    if self.candidate_models
                    else baseline.thinking_model
                ),
            )
            try:
                result = asyncio.run(
                    asyncio.wait_for(
                        self._score_config(config), timeout=trial_timeout_s
                    )
                )
            except Exception as e:
                trial.set_user_attr("error", type(e).__name__)
                return 0.0
            trial.set_user_attr("hit_rate", result.hit_rate)
            trial.set_user_attr("mean_latency_ms", result.mean_latency)
            trial.set_user_attr("n_hits", result.n_hits)
            return result.score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        baseline_dump = baseline.model_dump()
        seed_params: dict[str, Any] = {}
        for k in (
            "top_k",
            "retrieval_factor",
            "rerank_top_n",
            "rerank_cap_multiplier",
            "semantic_ratio",
            "fusion",
            "short_query_threshold",
            "short_query_sort_tokens",
            "bm25_fallback_semantic_ratio",
            "name_field_boost_max",
            "boost_decay_sigma",
            "enable_hyde",
            "enable_filter_intent",
            "enable_quality_gate",
            "enable_preprocess_llm",
        ):
            seed_params[k] = baseline_dump[k]
        if self.candidate_models:
            default_model = self.candidate_models[0]
            if self.tune_strong_model:
                seed_params["strong_model"] = baseline.strong_model or default_model
            seed_params["weak_model"] = baseline.weak_model or default_model
            seed_params["thinking_model"] = baseline.thinking_model or default_model
        for k in self._NONEABLE_FIELDS:
            val = baseline_dump[k]
            seed_params[f"{k}_enabled"] = val is not None
            if val is not None:
                seed_params[k] = val
        study.enqueue_trial(seed_params)

        callbacks: list[Any] = []
        if patience is not None and patience > 0:
            callbacks.append(_EarlyStopCallback(patience=patience))

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
            callbacks=callbacks or None,
        )

        return self._decode_best(study.best_params, baseline)

    # Fields that are worth varying per-collection (retrieval shape + pipeline toggles).
    # Model names, timeouts, and rerank internals stay global.
    _COLLECTION_TUNABLE = (
        "semantic_ratio",
        "retrieval_factor",
        "rerank_top_n",
        "fusion",
        "enable_filter_intent",
        "enable_preprocess_llm",
        "enable_hyde",
        "bm25_fallback_semantic_ratio",
        "short_query_threshold",
        "short_query_sort_tokens",
        "fast_accept_score",
        "fast_accept_confidence",
        "rerank_min_score",
    )

    def tune_collection(
        self,
        collection: str,
        global_config: RAGConfig,
        *,
        n_trials: int = 30,
        seed: int = 42,
        show_progress: bool = True,
        patience: int | None = 10,
        trial_timeout_s: float = 180.0,
    ) -> RAGConfig:
        """Tune collection-specific overrides on top of global_config.

        Returns a RAGConfig representing the merged (global + collection) best config.
        Use save_collection_toml() to persist only the overrides.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Install optuna: pip install rag7[tune]") from exc

        baseline = global_config

        def objective(trial: "optuna.Trial") -> float:
            overrides: dict[str, Any] = {
                    **baseline.model_dump(),
                    "semantic_ratio": trial.suggest_float("semantic_ratio", 0.2, 0.9),
                    "retrieval_factor": trial.suggest_int("retrieval_factor", 2, 8),
                    "rerank_top_n": trial.suggest_int("rerank_top_n", 3, 10),
                    "fusion": trial.suggest_categorical("fusion", ["rrf", "dbsf"]),
                    "enable_filter_intent": trial.suggest_categorical(
                        "enable_filter_intent", [True, False]
                    ),
                    "enable_preprocess_llm": trial.suggest_categorical(
                        "enable_preprocess_llm", [True, False]
                    ),
                    "enable_hyde": trial.suggest_categorical(
                        "enable_hyde", [True, False]
                    ),
                    "bm25_fallback_semantic_ratio": trial.suggest_float(
                        "bm25_fallback_semantic_ratio", 0.7, 1.0
                    ),
                    "short_query_threshold": trial.suggest_int(
                        "short_query_threshold", 3, 8
                    ),
                    "short_query_sort_tokens": trial.suggest_categorical(
                        "short_query_sort_tokens", [True, False]
                    ),
                    "fast_accept_score": (
                        trial.suggest_float("fast_accept_score", 0.5, 0.95)
                        if trial.suggest_categorical(
                            "fast_accept_score_enabled", [True, False]
                        )
                        else None
                    ),
                    "fast_accept_confidence": (
                        trial.suggest_float("fast_accept_confidence", 0.6, 0.95)
                        if trial.suggest_categorical(
                            "fast_accept_confidence_enabled", [True, False]
                        )
                        else None
                    ),
                    "rerank_min_score": (
                        trial.suggest_float("rerank_min_score", 0.05, 0.5)
                        if trial.suggest_categorical(
                            "rerank_min_score_enabled", [True, False]
                        )
                        else None
                    ),
                }
            config = RAGConfig(**overrides)
            try:
                result = asyncio.run(
                    asyncio.wait_for(
                        self._score_config(config), timeout=trial_timeout_s
                    )
                )
            except Exception as e:
                trial.set_user_attr("error", type(e).__name__)
                return 0.0
            trial.set_user_attr("hit_rate", result.hit_rate)
            trial.set_user_attr("mean_latency_ms", result.mean_latency)
            trial.set_user_attr("n_hits", result.n_hits)
            return result.score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"collection-{collection}",
        )

        # Seed with global baseline
        baseline_dump = baseline.model_dump()
        seed_params: dict[str, Any] = {
            k: baseline_dump[k]
            for k in (
                "semantic_ratio",
                "retrieval_factor",
                "rerank_top_n",
                "fusion",
                "enable_filter_intent",
                "enable_preprocess_llm",
                "enable_hyde",
                "bm25_fallback_semantic_ratio",
                "short_query_threshold",
                "short_query_sort_tokens",
            )
        }
        for key in ("fast_accept_score", "fast_accept_confidence", "rerank_min_score"):
            val = baseline_dump[key]
            seed_params[f"{key}_enabled"] = val is not None
            if val is not None:
                seed_params[key] = val
        study.enqueue_trial(seed_params)

        callbacks: list[Any] = []
        if patience is not None and patience > 0:
            callbacks.append(_EarlyStopCallback(patience=patience))
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
            callbacks=callbacks or None,
        )

        best_params = study.best_params
        decoded: dict[str, Any] = {}
        for key, value in best_params.items():
            if key.endswith("_enabled"):
                continue
            decoded[key] = value
        for key in ("fast_accept_score", "fast_accept_confidence", "rerank_min_score"):
            if best_params.get(f"{key}_enabled") is False:
                decoded[key] = None
        return RAGConfig(**{**baseline.model_dump(), **decoded})

    @classmethod
    def _decode_best(
        cls, best_params: dict[str, Any], baseline: RAGConfig
    ) -> RAGConfig:
        decoded: dict[str, Any] = {}
        for key, value in best_params.items():
            if key.endswith("_enabled"):
                continue
            decoded[key] = value
        for key in cls._NONEABLE_FIELDS:
            enabled = best_params.get(f"{key}_enabled")
            if enabled is False:
                decoded[key] = None
            elif enabled is True and key not in decoded:
                decoded[key] = baseline.model_dump().get(key)
        merged = baseline.model_dump() | decoded
        return RAGConfig(**merged)


def _cli_main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m rag7.tuner",
        description="Tune RAGConfig on your testset using Optuna.",
    )
    parser.add_argument(
        "--index", required=True, help="Meilisearch index name (uses MEILI_URL/KEY env)"
    )
    parser.add_argument(
        "--hits",
        required=True,
        help="Path to testset JSON/JSONL with query/expected_ids/id_field entries",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="rag7.config.toml",
        help="Where to save the tuned config (default: rag7.config.toml)",
    )
    parser.add_argument(
        "--pyproject",
        action="store_true",
        help="Save to pyproject.toml [tool.rag7] instead of --output",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        metavar="PROVIDER:MODEL",
        help=(
            "Candidate LLM specs the tuner may pick between for the weak / "
            "thinking (and optionally strong) tiers. Example: "
            "--models azure:gpt-5.4 azure:gpt-5.4-mini azure:gpt-5.4-nano"
        ),
    )
    args = parser.parse_args()

    from .backend import MeilisearchBackend
    from .utils import _make_azure_embed_fn

    hits = load_testset(args.hits)
    print(f"Loaded {len(hits)} hit cases from {args.hits}")

    embed_fn = _make_azure_embed_fn()
    if embed_fn is None:
        print(
            "Azure OpenAI env vars missing (AZURE_OPENAI_ENDPOINT + "
            "AZURE_OPENAI_API_KEY). Configure them in your env or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    tuner = RAGTuner(
        backend_factory=lambda: MeilisearchBackend(index=args.index),
        embed_fn=embed_fn,
        hit_cases=hits,
        candidate_models=args.models,
    )
    best = tuner.tune(n_trials=args.trials, seed=args.seed, show_progress=True)

    if args.pyproject:
        best.save_pyproject()
        print("Saved tuned config to pyproject.toml [tool.rag7]")
    else:
        best.save_toml(args.output)
        print(f"Saved tuned config to {args.output}")

    overrides = best.overrides()
    print(f"\nOverrides from defaults ({len(overrides)}):")
    for k, v in overrides.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    _cli_main()
