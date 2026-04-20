"""Integration test: mem0 + LangGraph checkpointer on Postgres + Azure OpenAI."""

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# langchain Azure requires OPENAI_API_VERSION
os.environ.setdefault(
    "OPENAI_API_VERSION",
    os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

# Skip the whole module on CI / anywhere Azure env vars are not provided.
pytestmark = pytest.mark.skipif(
    not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
    reason="Azure OpenAI env vars (AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY) not set",
)

PG = os.getenv("TEST_PG_URL", "postgresql://postgres:postgres@localhost:5432/rag7test")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
AZURE_EMBED = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_KWARGS = {
    "api_key": AZURE_KEY,
    "azure_endpoint": AZURE_ENDPOINT,
    "api_version": AZURE_API_VERSION,
}


def test_mem0_checkpointer_postgres():
    asyncio.run(_run())


async def _run():
    from mem0 import AsyncMemory
    from mem0.configs.base import (
        EmbedderConfig,
        LlmConfig,
        MemoryConfig,
        VectorStoreConfig,
    )
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from rag7 import init_agent
    from rag7.backend import InMemoryBackend

    memory = AsyncMemory(
        config=MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="pgvector",
                config={"connection_string": PG},
            ),
            llm=LlmConfig(
                provider="azure_openai",
                config={
                    "azure_kwargs": {
                        **AZURE_KWARGS,
                        "azure_deployment": AZURE_DEPLOYMENT,
                    },
                },
            ),
            embedder=EmbedderConfig(
                provider="azure_openai",
                config={
                    "azure_kwargs": {**AZURE_KWARGS, "azure_deployment": AZURE_EMBED},
                },
            ),
        )
    )

    async with AsyncPostgresSaver.from_conn_string(PG) as checkpointer:
        await checkpointer.setup()

        rag = init_agent(
            "test",
            model=f"azure_openai:{AZURE_DEPLOYMENT}",
            backend=InMemoryBackend(),
            checkpointer=checkpointer,
            mem0_memory=memory,
        )

        config = {
            "configurable": {"thread_id": "test-thread-1", "user_id": "test-user"}
        }

        # Turn 1 — plant a preference
        state1 = await rag.ainvoke("I prefer very short answers.", config=config)
        assert state1.answer
        print("\nTurn 1:", state1.answer[:100])

        # Turn 2 — same thread, checkpointer resumes
        state2 = await rag.ainvoke("What is RAG?", config=config)
        assert state2.answer
        print("Turn 2:", state2.answer[:100])

        # Turn 3 — new thread, same user — mem0 should recall preference
        config2 = {
            "configurable": {"thread_id": "test-thread-2", "user_id": "test-user"}
        }
        state3 = await rag.ainvoke("Explain vector search.", config=config2)
        assert state3.answer
        print("Turn 3 (new thread):", state3.answer[:100])

        read_steps = [s for s in state3.trace if s.get("node") == "read_memory"]
        print("mem0 recalled:", read_steps)
        assert read_steps, "mem0 should have recalled something for the new thread"
