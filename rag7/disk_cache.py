from __future__ import annotations

import json
from pathlib import Path


def _cache_dir() -> Path:
    base = Path.home() / ".cache" / "rag7"
    base.mkdir(parents=True, exist_ok=True)
    return base


def filters_cache_path(schema_sig: str) -> Path:
    return _cache_dir() / f"filters-v1_{schema_sig}.json"


def filters_cache_load(schema_sig: str) -> dict[str, list[str]] | None:
    try:
        p = filters_cache_path(schema_sig)
        if not p.is_file():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def filters_cache_save(schema_sig: str, values: dict[str, list[str]]) -> None:
    try:
        filters_cache_path(schema_sig).write_text(json.dumps(values, indent=2))
    except Exception:
        pass


def field_rank_cache_path(schema_sig: str) -> Path:
    return _cache_dir() / f"field-rank-v1_{schema_sig}.json"


def field_rank_cache_load(schema_sig: str) -> dict[str, int] | None:
    try:
        p = field_rank_cache_path(schema_sig)
        if not p.is_file():
            return None
        raw = json.loads(p.read_text())
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        return None


def field_rank_cache_save(schema_sig: str, ranks: dict[str, int]) -> None:
    try:
        field_rank_cache_path(schema_sig).write_text(json.dumps(ranks, indent=2))
    except Exception:
        pass
