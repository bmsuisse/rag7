"""Run RAGTuner on the eval_v2 adversarial+paraphrase testset.

Usage: uv run --env-file .env python -m tests.eval_v2.run_tuner [--trials N]
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from rag7.backend import MeilisearchBackend
from rag7.config import RAGConfig
from rag7.utils import _make_azure_embed_fn
from tests.eval_v2.adversarial import HIT_CASE, build_adversarial_cases
from tests.eval_v2.runner import paraphrase_groups
from tests.eval_v2.__main__ import (
    ARTICLE_HITS,
    ONETRADE_DE_HITS,
    SUPPLIER_CATALOGS_DE_HITS,
)


def build_testset() -> list[tuple[str, list[HIT_CASE], list[HIT_CASE]]]:
    """Return [(index_name, hit_cases, base_cases), ...]."""
    suites = [
        ("onetrade_articles_de", ONETRADE_DE_HITS),
        ("article", ARTICLE_HITS),
        ("supplier_catalogs_de", SUPPLIER_CATALOGS_DE_HITS),
    ]
    return [
        (index, list(base) + build_adversarial_cases(list(base)), list(base))
        for index, base in suites
    ]


async def _retrieve_ids(rag: Any, query: str, id_field: str, k: int = 5) -> list[str]:
    _, docs = await rag._aretrieve_documents(query, top_k=k)
    return [str(d.metadata.get(id_field, "")) for d in docs]


async def evaluate_config(
    config: RAGConfig,
    testset: list[tuple[str, list[HIT_CASE], list[HIT_CASE]]],
    embed_fn: Any,
) -> dict[str, float]:
    """Compute hit@5 + paraphrase consistency + stable_top1 across indexes."""
    from rag7 import AgenticRAG

    total_hits = 0
    total = 0
    consistencies: list[float] = []
    stable_count = 0
    group_count = 0
    sem = asyncio.Semaphore(10)

    for index, hit_cases, base_cases in testset:
        rag = AgenticRAG(
            index=index,
            backend=MeilisearchBackend(index=index),
            embed_fn=embed_fn,
            config=config,
            auto_strategy=False,
        )

        # Hit@5 over adversarial cases
        async def _hit(case: HIT_CASE) -> bool:
            q, expected, field_name = case
            async with sem:
                retrieved = await _retrieve_ids(rag, q, field_name)
            return any(str(e) in retrieved for e in expected)

        total_hits += sum(await asyncio.gather(*(_hit(c) for c in hit_cases)))
        total += len(hit_cases)

        # Paraphrase consistency + stable_top1
        groups = paraphrase_groups(base_cases)
        for (seed_q, expected, field_name), variants in groups:
            async with sem:
                seed_ids = await _retrieve_ids(rag, seed_q, field_name)
            seed_top1 = seed_ids[0] if seed_ids else None
            if not variants:
                continue
            variant_tops = await asyncio.gather(
                *(_retrieve_ids(rag, v, field_name) for v in variants)
            )
            expected_set = {str(e) for e in expected}
            hit_count = sum(
                1 for ids in variant_tops if any(i in expected_set for i in ids)
            )
            consistencies.append(hit_count / len(variants))
            if seed_top1 and all(ids and ids[0] == seed_top1 for ids in variant_tops):
                stable_count += 1
            group_count += 1

    hit_rate = total_hits / total if total else 0.0
    consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0
    stable = stable_count / group_count if group_count else 0.0
    return {
        "hit@5": hit_rate,
        "consistency": consistency,
        "stable_top1": stable,
        "combined": hit_rate * 0.4 + consistency * 0.35 + stable * 0.25,
    }


CANDIDATE_MODELS = ["azure:gpt-5.4-mini", "azure:gpt-5.4-nano"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="rag7.config.toml")
    parser.add_argument(
        "--tune-models",
        action="store_true",
        help="Also tune weak/thinking model selection from CANDIDATE_MODELS",
    )
    parser.add_argument(
        "--tune-strong-model",
        action="store_true",
        help=(
            "Include strong_model (gen_llm) in the tuning search space. By "
            "default strong_model is held fixed — generation quality is "
            "rarely worth optimizing away."
        ),
    )
    parser.add_argument(
        "--strong-model",
        default=None,
        help="Fix strong_model to this spec (e.g. 'azure:brain').",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stop when no improvement for N trials (0 = no early stop)",
    )
    parser.add_argument(
        "--trial-timeout-s",
        type=float,
        default=120.0,
        help="Per-trial timeout (seconds) — slow/hung trials score 0",
    )
    args = parser.parse_args()

    testset = build_testset()
    total_cases = sum(len(cases) for _, cases, _ in testset)
    print(f"Testset: {len(testset)} indexes, {total_cases} total cases")

    embed_fn = _make_azure_embed_fn()

    baseline = RAGConfig(strong_model=args.strong_model) if args.strong_model else RAGConfig()
    baseline_metrics = asyncio.run(evaluate_config(baseline, testset, embed_fn))
    print(
        f"Baseline: hit@5={baseline_metrics['hit@5']:.4f} "
        f"consistency={baseline_metrics['consistency']:.4f} "
        f"stable={baseline_metrics['stable_top1']:.4f} "
        f"combined={baseline_metrics['combined']:.4f}"
    )

    # Use the multi-index evaluator as the objective by flattening to one RAGTuner
    # per index isn't ideal; build a custom optuna loop here.
    import optuna

    def _suggest_float_or_none(
        trial: optuna.Trial, name: str, low: float, high: float
    ) -> float | None:
        """Categorical 'enabled?' gate around a float range — lets TPE explore
        'disable this stage' as a first-class hypothesis."""
        if trial.suggest_categorical(f"{name}_enabled", [True, False]):
            return trial.suggest_float(name, low, high)
        return None

    def objective(trial: optuna.Trial) -> float:
        cfg = RAGConfig(
            retrieval_factor=trial.suggest_int("retrieval_factor", 2, 8),
            rerank_top_n=trial.suggest_int("rerank_top_n", 3, 10),
            rerank_cap_multiplier=trial.suggest_float("rerank_cap_multiplier", 1.5, 4.0),
            semantic_ratio=trial.suggest_float("semantic_ratio", 0.3, 0.9),
            fusion=trial.suggest_categorical("fusion", ["rrf", "dbsf"]),
            short_query_threshold=trial.suggest_int("short_query_threshold", 3, 8),
            short_query_sort_tokens=trial.suggest_categorical(
                "short_query_sort_tokens", [True, False]
            ),
            bm25_fallback_threshold=_suggest_float_or_none(
                trial, "bm25_fallback_threshold", 0.2, 0.6
            ),
            bm25_fallback_semantic_ratio=trial.suggest_float(
                "bm25_fallback_semantic_ratio", 0.7, 1.0
            ),
            fast_accept_score=_suggest_float_or_none(
                trial, "fast_accept_score", 0.5, 0.95
            ),
            fast_accept_confidence=_suggest_float_or_none(
                trial, "fast_accept_confidence", 0.6, 0.95
            ),
            rerank_skip_dominance=_suggest_float_or_none(
                trial, "rerank_skip_dominance", 0.6, 0.95
            ),
            rerank_skip_gap=trial.suggest_float("rerank_skip_gap", 0.05, 0.3),
            expert_threshold=_suggest_float_or_none(
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
                trial.suggest_categorical("strong_model", CANDIDATE_MODELS)
                if args.tune_models and args.tune_strong_model
                else (args.strong_model or baseline.strong_model)
            ),
            weak_model=(
                trial.suggest_categorical("weak_model", CANDIDATE_MODELS)
                if args.tune_models
                else None
            ),
            thinking_model=(
                trial.suggest_categorical("thinking_model", CANDIDATE_MODELS)
                if args.tune_models
                else None
            ),
        )
        try:
            metrics = asyncio.run(
                asyncio.wait_for(
                    evaluate_config(cfg, testset, embed_fn),
                    timeout=args.trial_timeout_s,
                )
            )
        except (asyncio.TimeoutError, Exception) as e:
            trial.set_user_attr("error", type(e).__name__)
            return 0.0
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        return metrics["combined"]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed trial 0 with the shipped baseline so TPE only improves on it.
    def _enc(val: float | None, key: str) -> dict[str, Any]:
        if val is None:
            return {f"{key}_enabled": False}
        return {f"{key}_enabled": True, key: val}

    seed_params: dict[str, Any] = {
        "retrieval_factor": baseline.retrieval_factor,
        "rerank_top_n": baseline.rerank_top_n,
        "rerank_cap_multiplier": baseline.rerank_cap_multiplier,
        "semantic_ratio": baseline.semantic_ratio,
        "fusion": baseline.fusion,
        "short_query_threshold": baseline.short_query_threshold,
        "short_query_sort_tokens": baseline.short_query_sort_tokens,
        "bm25_fallback_semantic_ratio": baseline.bm25_fallback_semantic_ratio,
        "rerank_skip_gap": baseline.rerank_skip_gap,
        "enable_hyde": baseline.enable_hyde,
        "enable_filter_intent": baseline.enable_filter_intent,
        "enable_quality_gate": baseline.enable_quality_gate,
        "enable_preprocess_llm": baseline.enable_preprocess_llm,
    }
    for key, val in [
        ("bm25_fallback_threshold", baseline.bm25_fallback_threshold),
        ("fast_accept_score", baseline.fast_accept_score),
        ("fast_accept_confidence", baseline.fast_accept_confidence),
        ("rerank_skip_dominance", baseline.rerank_skip_dominance),
        ("expert_threshold", baseline.expert_threshold),
    ]:
        seed_params.update(_enc(val, key))
    if args.tune_models:
        # Seed trial 0 with production-grade tiers.
        if args.tune_strong_model:
            seed_params["strong_model"] = args.strong_model or "azure:brain"
        seed_params["weak_model"] = "azure:gpt-5.4-mini"
        seed_params["thinking_model"] = "azure:gpt-5.4-mini"
    study.enqueue_trial(seed_params)

    from rag7.tuner import _EarlyStopCallback

    callbacks = [_EarlyStopCallback(args.patience)] if args.patience > 0 else None
    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=False,
        callbacks=callbacks,
    )

    best_trial = study.best_trial
    print(
        f"\nBest: combined={best_trial.value:.4f} "
        f"hit@5={best_trial.user_attrs.get('hit@5', 0):.4f} "
        f"consistency={best_trial.user_attrs.get('consistency', 0):.4f} "
        f"stable={best_trial.user_attrs.get('stable_top1', 0):.4f}"
    )
    print(f"Baseline combined: {baseline_metrics['combined']:.4f}")
    print(f"Best params: {best_trial.params}")

    # Decode optuna search-space params back into RAGConfig fields:
    # collapse "foo_enabled + foo" pairs into None or foo.
    NONEABLE = {
        "bm25_fallback_threshold",
        "fast_accept_score",
        "fast_accept_confidence",
        "rerank_skip_dominance",
        "expert_threshold",
    }
    decoded: dict[str, Any] = {}
    for key, value in study.best_params.items():
        if key.endswith("_enabled"):
            base = key[: -len("_enabled")]
            if base in NONEABLE and value is False:
                decoded[base] = None
            continue
        if key in NONEABLE and decoded.get(key) is None and key not in decoded:
            # may already be None from _enabled=False in a previous key; skip
            pass
        decoded[key] = value
    # Re-apply None for _enabled=False even if the base key came after
    for key in list(decoded.keys()):
        if key in NONEABLE and study.best_params.get(f"{key}_enabled") is False:
            decoded[key] = None
    merged = baseline.model_dump() | decoded
    best = RAGConfig(**merged)
    best.save_toml(args.output)
    print(f"Saved best config -> {args.output}")


if __name__ == "__main__":
    main()
