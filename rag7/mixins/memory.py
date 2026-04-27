from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

    from ..models import RAGState


class MemoryMixin:
    """Conversation-memory side of AgenticRAG: optional LangGraph store
    or mem0 backend, used by the read_memory / write_memory graph nodes.

    The read path filters recalled memories by relevance score
    (``memory_relevance_threshold``) before they reach the rest of the
    graph. Only memories that clear the bar are placed in
    ``state.memory_context`` — the field that retrieval and generation
    consume. Below-threshold memories are still recorded in
    ``state.trace`` for diagnostics but do not influence the answer.
    """

    _memory_store: Any
    _mem0_memory: Any
    memory_relevance_threshold: float

    async def _aread_memory(
        self,
        state: RAGState,
        *,
        store: Any = None,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self._mem0_memory is not None:
            return await self._aread_mem0(state, config=config)
        if store is None:
            return {}
        try:
            user_id = ((config or {}).get("configurable") or {}).get(
                "user_id", "default"
            )
            memories = await store.asearch(
                ("memories", user_id), query=state.question, limit=5
            )
            if not memories:
                return {}
            mem_text = "\n".join(f"- {m.value['text']}" for m in memories)
            return {
                "trace": state.trace + [{"node": "read_memory", "memories": mem_text}]
            }
        except Exception:
            return {}

    async def _awrite_memory(
        self,
        state: RAGState,
        *,
        store: Any = None,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self._mem0_memory is not None:
            return await self._awrite_mem0(state, config=config)
        if store is None or not state.answer:
            return {}
        try:
            user_id = ((config or {}).get("configurable") or {}).get(
                "user_id", "default"
            )
            import time as _time
            import uuid

            key = str(uuid.uuid4())
            await store.aput(
                ("memories", user_id),
                key,
                {
                    "text": f"Q: {state.question}\nA: {state.answer[:300]}",
                    "ts": _time.time(),
                },
            )
        except Exception:
            pass
        return {}

    async def _aread_mem0(
        self, state: RAGState, *, config: RunnableConfig | None = None
    ) -> dict:
        user_id = ((config or {}).get("configurable") or {}).get("user_id", "default")
        try:
            import asyncio
            import inspect

            m = self._mem0_memory
            search_fn = getattr(m, "asearch", None) or m.search
            filters = {"user_id": user_id}
            if inspect.iscoroutinefunction(search_fn):
                results = await search_fn(state.question, filters=filters)
            else:
                results = await asyncio.to_thread(
                    search_fn, state.question, filters=filters
                )
            entries = (
                results.get("results", results)
                if isinstance(results, dict)
                else results
            )
            if not entries:
                return {}
            threshold = self.memory_relevance_threshold
            relevant: list[str] = []
            scanned: list[tuple[str, float | None]] = []
            for r in entries[:10]:
                if not isinstance(r, dict):
                    continue
                text = str(r.get("memory") or r)
                raw_score = r.get("score")
                try:
                    score = float(raw_score) if raw_score is not None else None
                except (TypeError, ValueError):
                    score = None
                scanned.append((text, score))
                if score is not None and score >= threshold:
                    relevant.append(text)
            if not relevant:
                return {
                    "trace": state.trace
                    + [
                        {
                            "node": "read_memory",
                            "skipped": "below_threshold",
                            "threshold": threshold,
                            "best_score": max(
                                (s for _, s in scanned if s is not None),
                                default=None,
                            ),
                            "n_scanned": len(scanned),
                        }
                    ],
                }
            mem_text = "\n".join(f"- {m}" for m in relevant[:5])
            return {
                "trace": state.trace
                + [
                    {
                        "node": "read_memory",
                        "memories": mem_text,
                        "n_kept": len(relevant[:5]),
                        "n_scanned": len(scanned),
                        "threshold": threshold,
                    }
                ],
                "memory_context": mem_text,
            }
        except Exception:
            return {}

    async def _awrite_mem0(
        self, state: RAGState, *, config: RunnableConfig | None = None
    ) -> dict:
        if not state.answer:
            return {}
        user_id = ((config or {}).get("configurable") or {}).get("user_id", "default")
        try:
            import asyncio
            import inspect

            m = self._mem0_memory
            add_fn = getattr(m, "aadd", None) or m.add
            messages = [
                {"role": "user", "content": state.question},
                {"role": "assistant", "content": state.answer[:500]},
            ]
            if inspect.iscoroutinefunction(add_fn):
                await add_fn(messages, user_id=user_id)
            else:
                await asyncio.to_thread(add_fn, messages, user_id=user_id)
        except Exception as e:
            logging.getLogger(__name__).warning("mem0 write failed: %s", e)
        return {}
