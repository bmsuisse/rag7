from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

    from ..models import RAGState


class MemoryMixin:
    """Conversation-memory side of AgenticRAG: optional LangGraph store
    or mem0 backend, used by the read_memory / write_memory graph nodes.
    """

    _memory_store: Any
    _mem0_memory: Any

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
            mem_text = "\n".join(f"- {r.get('memory', r)}" for r in entries[:5])
            return {
                "trace": state.trace + [{"node": "read_memory", "memories": mem_text}]
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
