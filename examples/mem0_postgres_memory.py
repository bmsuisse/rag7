"""
mem0 + LangGraph checkpointer on Postgres + Azure OpenAI

Requirements:
    pip install rag7 mem0ai langgraph-checkpoint-postgres "psycopg[binary]"

Environment variables (or .env file):
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT         (e.g. gpt-5.4-mini)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT (e.g. text-embedding-3-small)
    AZURE_OPENAI_API_VERSION        (e.g. 2024-12-01-preview)
    OPENAI_API_VERSION              (same as above — required by langchain)

Postgres:
    docker run -d --name rag7-pg \\
      -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag7db \\
      -p 5432:5432 pgvector/pgvector:pg16
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault(
    "OPENAI_API_VERSION",
    os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

PG = os.getenv("PG_URL", "postgresql://postgres:postgres@localhost:5432/rag7db")

AZURE_KWARGS = {
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
}


async def main():
    from mem0 import AsyncMemory
    from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from rag7 import init_agent
    from rag7.backend import MeilisearchBackend  # swap for your backend

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    embed_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    # mem0 — LLM-based fact extraction stored in pgvector
    memory = AsyncMemory(
        config=MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="pgvector",
                config={"connection_string": PG},
            ),
            llm=LlmConfig(
                provider="azure_openai",
                config={"azure_kwargs": {**AZURE_KWARGS, "azure_deployment": deployment}},
            ),
            embedder=EmbedderConfig(
                provider="azure_openai",
                config={"azure_kwargs": {**AZURE_KWARGS, "azure_deployment": embed_deployment}},
            ),
        )
    )

    # LangGraph checkpointer — resumes conversation threads from Postgres
    async with AsyncPostgresSaver.from_conn_string(PG) as checkpointer:
        await checkpointer.setup()

        rag = init_agent(
            "my-index",
            model=f"azure_openai:{deployment}",
            backend=MeilisearchBackend(
                url=os.environ["MEILI_URL"],
                api_key=os.environ["MEILI_KEY"],
                index_name="my-index",
            ),
            checkpointer=checkpointer,   # per-thread conversation memory
            mem0_memory=memory,          # cross-session fact memory
        )

        # Session 1 — Alice plants a preference
        config = {"configurable": {"thread_id": "alice-session-1", "user_id": "alice"}}
        state = await rag.ainvoke("I prefer answers in German.", config=config)
        print("Session 1:", state.answer)

        # Session 1 — follow-up in same thread (checkpointer resumes)
        state = await rag.ainvoke("What is hybrid search?", config=config)
        print("Session 1 follow-up:", state.answer)

        # Session 2 — new thread, same user (mem0 recalls preference)
        config2 = {"configurable": {"thread_id": "alice-session-2", "user_id": "alice"}}
        state = await rag.ainvoke("Explain vector search.", config=config2)
        print("Session 2 (new thread):", state.answer)

        # Show what mem0 recalled
        for step in state.trace:
            if step.get("node") == "read_memory":
                print("\nmem0 recalled:\n", step.get("memories"))


if __name__ == "__main__":
    asyncio.run(main())
