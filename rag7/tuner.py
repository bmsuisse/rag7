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


class _EarlyStopCallback:
    """Optuna callback: stop the study if no trial improves for ``patience``
    consecutive attempts. Mirrors early-stopping patterns from sklearn.
    """

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
    # Optional pool of ``provider:model`` specs the tuner can pick between
    # for each of the three model tiers (strong / weak / thinking). If empty,
    # LLMs passed via ``llm=`` / ``gen_llm=`` or env defaults are used and
    # not varied. Example:
    #   ``["azure:gpt-5.4", "azure:gpt-5.4-mini", "azure:gpt-5.4-nano"]``
    candidate_models: Sequence[str] = field(default_factory=list)
    # By default, ``strong_model`` (= gen_llm, the final-answer model) is
    # held constant at whatever the baseline config specifies. Generation
    # quality is usually not where you want to save a few cents. Set to
    # ``True`` to include it in the search if you really want to.
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

    # Fields whose "None" value is a first-class tuning option
    # (semantically = disable this stage).
    _NONEABLE_FIELDS = frozenset(
        {
            "bm25_fallback_threshold",
            "fast_accept_score",
            "fast_accept_confidence",
            "rerank_skip_dominance",
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
        """Run optuna search and return the best config.

        Explores both scalar knobs and stage-enable toggles. Optional fields
        (``bm25_fallback_threshold``, ``expert_threshold``, etc.) are explored
        with a categorical "enabled?" gate so ``None`` (disable the stage) is
        a first-class hypothesis alongside any numeric value.

        Always enqueues the baseline (default or provided) as trial 0 so
        tuning can only improve over the starting point.

        Parameters
        ----------
        patience:
            Early-stop the study when no trial improves the best value in
            this many consecutive attempts. ``None`` disables early stopping.
        trial_timeout_s:
            Per-trial hard timeout. Slow or hung trials (typically network
            stalls) score 0 and don't block the study.
        """
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
            """Float or None — TPE can explore disabling the stage."""
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
                fusion=trial.suggest_categorical("fusion", ["rrf", "dbsf"]),
                hyde_min_words=_opt_int(trial, "hyde_min_words", 4, 12),
                short_query_threshold=trial.suggest_int(
                    "short_query_threshold", 3, 8
                ),
                short_query_sort_tokens=trial.suggest_categorical(
                    "short_query_sort_tokens", [True, False]
                ),
                bm25_fallback_threshold=_opt_float(
                    trial, "bm25_fallback_threshold", 0.2, 0.6
                ),
                bm25_fallback_semantic_ratio=trial.suggest_float(
                    "bm25_fallback_semantic_ratio", 0.7, 1.0
                ),
                fast_accept_score=_opt_float(
                    trial, "fast_accept_score", 0.5, 0.95
                ),
                fast_accept_confidence=_opt_float(
                    trial, "fast_accept_confidence", 0.6, 0.95
                ),
                rerank_skip_dominance=_opt_float(
                    trial, "rerank_skip_dominance", 0.6, 0.95
                ),
                rerank_skip_gap=trial.suggest_float("rerank_skip_gap", 0.05, 0.3),
                name_field_boost_max=trial.suggest_float(
                    "name_field_boost_max", 0.0, 0.5
                ),
                expert_threshold=_opt_float(
                    trial, "expert_threshold", 0.05, 0.3
                ),
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
                    trial.suggest_categorical(
                        "weak_model", list(self.candidate_models)
                    )
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
            except (asyncio.TimeoutError, Exception) as e:
                trial.set_user_attr("error", type(e).__name__)
                return 0.0
            trial.set_user_attr("hit_rate", result.hit_rate)
            trial.set_user_attr("mean_latency_ms", result.mean_latency)
            trial.set_user_attr("n_hits", result.n_hits)
            return result.score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Seed trial 0 with the shipped baseline so TPE only improves on it.
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
            "rerank_skip_gap",
            "name_field_boost_max",
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

    @classmethod
    def _decode_best(
        cls, best_params: dict[str, Any], baseline: RAGConfig
    ) -> RAGConfig:
        """Decode optuna search-space params back into RAGConfig fields.

        Collapses ``foo_enabled`` + ``foo`` pairs into either the numeric
        value or ``None``.
        """
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
    """CLI: ``python -m rag7.tuner --hits cases.json --trials 50``.

    Requires an embed_fn and backend factory. This minimal CLI only supports
    Meilisearch backends (``--index NAME``) and Azure OpenAI embeddings. For
    other backends, use the Python API directly.
    """
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
            "Candidate LLM specs the tuner may choose between for both "
            "gen_model and llm_model. Example: "
            "--models azure:gpt-5.4 azure:gpt-5.4-mini azure:gpt-5.4-nano"
        ),
    )
    args = parser.parse_args()

    try:
        from .backend import MeilisearchBackend
        from .utils import _make_azure_embed_fn
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        sys.exit(1)

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
