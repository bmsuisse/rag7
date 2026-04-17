"""Regression tests for utils._run_sync (Databricks/Jupyter event-loop reentry).

The bug: sync shims like AgenticRAG.invoke() used asyncio.run() directly,
which raises RuntimeError when called from inside a running loop — the
default state in Databricks notebooks, Jupyter, FastAPI handlers, etc.

The fix: _run_sync() detects a running loop and falls back to running the
coroutine in a worker-thread's own loop.
"""

from __future__ import annotations

import asyncio

import pytest

from rag7.utils import _run_sync


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def test_run_sync_no_running_loop():
    """Plain sync caller — no loop active, should use asyncio.run."""
    assert _run_sync(_answer()) == 42


def test_run_sync_from_within_running_loop():
    """The Databricks/Jupyter scenario: a loop is already running."""

    async def outer():
        return _run_sync(_answer())

    assert asyncio.run(outer()) == 42


def test_run_sync_propagates_exceptions():
    async def boom():
        raise RuntimeError("deliberate")

    with pytest.raises(RuntimeError, match="deliberate"):
        _run_sync(boom())

    async def outer():
        return _run_sync(boom())

    with pytest.raises(RuntimeError, match="deliberate"):
        asyncio.run(outer())


def test_run_sync_nested_gather():
    """Worker-thread loop must handle gather + real awaits."""

    async def work(n: int) -> int:
        await asyncio.sleep(0)
        return n * 2

    async def combined() -> list[int]:
        return list(await asyncio.gather(work(1), work(2), work(3)))

    async def outer():
        return _run_sync(combined())

    assert asyncio.run(outer()) == [2, 4, 6]
