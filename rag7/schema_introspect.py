from __future__ import annotations

import math
import re
import warnings
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from .backend import SearchBackend


_ID_SUFFIXES = ("_id", "_code", "_key", "_num")

_CATEGORY_FIELD_PATTERN = re.compile(
    r"(group|category|kategorie|categorie|categoria|class|klass|type|typ|section|department|rubrik|family)",
    re.IGNORECASE,
)


def _has_is_own_brand_field(backend: SearchBackend) -> bool:
    try:
        return "is_own_brand" in backend.get_index_config().filterable_attributes
    except Exception:
        return False


def _detect_embedder_name(backend: SearchBackend, *, fallback: str) -> str:
    try:
        names = backend.get_index_config().embedders
    except Exception:
        return fallback
    if not names:
        return fallback
    if "default" in names:
        return "default"
    return names[0]


def _validate_embedder_name(backend: SearchBackend, name: str) -> None:
    try:
        names = backend.get_index_config().embedders
    except Exception:
        return
    if names and name not in names:
        warnings.warn(
            f"Configured embedder_name={name!r} is not declared on the index "
            f"(available: {names}). Vector search will silently fall back to "
            f"BM25-only. Set RAG_EMBEDDER_NAME or pass embedder_name= to one "
            f"of the available names.",
            RuntimeWarning,
            stacklevel=3,
        )


def _align_embed_fn_with_backend(
    embed_fn: Callable[[str], list[float]] | None,
    backend: SearchBackend,
) -> Callable[[str], list[float]] | None:
    if embed_fn is None:
        return None
    try:
        cfg = backend.get_index_config()
    except Exception:
        return embed_fn
    dims = set(cfg.embedder_dims.values())
    if not dims:
        return embed_fn
    try:
        probe = embed_fn("probe")
    except Exception:
        return embed_fn
    probe_dim = len(probe)
    if probe_dim in dims:
        return embed_fn
    if len(dims) != 1:
        warnings.warn(
            f"Embed fn returns {probe_dim}-d vectors but index declares "
            f"multiple dims {sorted(dims)}; vector search may miss. "
            "Provide an embed_fn with a matching dimension.",
            RuntimeWarning,
            stacklevel=3,
        )
        return embed_fn
    target = next(iter(dims))
    if probe_dim < target:
        warnings.warn(
            f"Embed fn returns {probe_dim}-d vectors but index expects "
            f"{target}-d — can't up-project. Use a larger embedder.",
            RuntimeWarning,
            stacklevel=3,
        )
        return embed_fn

    rebuild = getattr(embed_fn, "_rag7_azure_rebuild", None)
    if callable(rebuild):
        rebuilt = cast(Callable[[str], list[float]] | None, rebuild(target))
        if rebuilt is not None:
            return rebuilt
    from .utils import _adapt_embed_fn_to_dim

    warnings.warn(
        f"Embed fn returns {probe_dim}-d but index expects {target}-d; "
        "auto-adapting via slice+L2 renorm (Matryoshka-style).",
        RuntimeWarning,
        stacklevel=3,
    )
    return _adapt_embed_fn_to_dim(embed_fn, target)


def _detect_category_fields(backend: SearchBackend) -> list[str]:
    try:
        samples = backend.sample_documents(limit=20)
    except Exception:
        return []
    if not samples:
        return []
    seen: dict[str, int] = {}
    for doc in samples:
        for k, v in doc.items():
            if not isinstance(v, str) or not v:
                continue
            if _CATEGORY_FIELD_PATTERN.search(k):
                seen[k] = seen.get(k, 0) + 1
    if not seen:
        return []

    def rank(k: str) -> tuple[int, int]:
        depth = 0
        for suffix, d in (("_l3", 3), ("_l2", 2), ("_l1", 1)):
            if k.lower().endswith(suffix):
                depth = d
                break
        return (-depth, -seen[k])

    return sorted(seen, key=rank)


def _detect_index_signals(
    backend: SearchBackend,
) -> tuple[list[str] | None, Callable[[dict], float] | None, list[str]]:
    config = backend.get_index_config()
    sortable = config.sortable_attributes
    if not sortable:
        return None, None, []

    samples = backend.sample_documents(limit=20)
    if not samples:
        return None, None, []

    field_sample: dict[str, Any] = {}
    for doc in samples:
        for f in sortable:
            if f not in field_sample and doc.get(f) is not None:
                field_sample[f] = doc[f]
        if len(field_sample) == len(sortable):
            break

    bool_fields: list[str] = []
    num_fields: list[str] = []
    for f in sortable:
        value = field_sample.get(f)
        if isinstance(value, bool):
            bool_fields.append(f)
        elif (
            isinstance(value, (int, float))
            and value >= 0
            and not any(f.endswith(s) for s in _ID_SUFFIXES)
        ):
            num_fields.append(f)

    if not bool_fields and not num_fields:
        return None, None, []

    sort_fields = [f"{f}:desc" for f in bool_fields + num_fields]

    def boost_fn(meta: dict) -> float:
        binary_boost = 1.05 if any(meta.get(f) for f in bool_fields) else 1.0
        top_num = max((float(meta.get(f, 0) or 0) for f in num_fields), default=0.0)
        if top_num < 150_000:
            top_num = 0.0
        return binary_boost * (1.0 + math.log1p(top_num) / 50)

    return sort_fields, boost_fn, num_fields
