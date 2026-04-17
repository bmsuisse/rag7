# Changelog

## 0.6.5

- `mem0_memory=` parameter for LLM-based fact extraction with deduplication and conflict resolution
- Supports both `Memory` (sync via thread-pool) and `AsyncMemory` (native async)
- Strips retrieved documents from checkpointed state to keep context lean

## 0.6.4

- `auto_strategy=True` is now the default — rag7 is agentic out of the box
- Added multi-collection routing (`collections=` parameter) with LLM-based selection
- Added `_MultiBackend` with `_ACTIVE_COLLECTIONS` context-variable scoping
- Extended filter coverage: NOT CONTAINS (ILIKE) for LanceDB, DuckDB, pgvector; Qdrant server MatchText; Chroma AND filters
- Added `% ` to `_SAFE_FILTER_RE` to allow ILIKE patterns
- Cleaned repo of all internal product references

## 0.6.3

- Backend-aware filter translator (Meili, SQL ILIKE, OData, Chroma dict, Qdrant native)
- `build_filter_expr` per-backend helper

## 0.6.1 — 0.6.2

- LanceDB filter coverage and test suite
- Embedding API call cache (disk-based)

## 0.6.0

- Multi-query swarm retrieval
- LangGraph state machine refactor
- Async-native pipeline with `_run_sync` for sync wrappers

## Earlier

See [GitHub releases](https://github.com/bmsuisse/rag7/releases) for full history.
