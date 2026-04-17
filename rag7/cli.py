"""
rag7 — Interactive RAG CLI

Usage:
    rag7                    # guided setup wizard
    rag7 --chat             # chat mode (default after setup)
    rag7 --retriever        # retriever-only mode (no LLM answer)
    rag7 -c my_index        # specify collection/index name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Callable

warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1 functionality.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

# Optional: python-dotenv for .env loading
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# Optional: rich for pretty output — falls back to plain print
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.table import Table

    _console = Console()
    _HAS_RICH = True
except ModuleNotFoundError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]

from langchain_core.documents import Document  # noqa: E402

from . import AgenticRAG, ConversationTurn  # noqa: E402


# ── Output helpers (rich or plain) ────────────────────────────────────────────


def _print(msg: str = "", **kwargs: Any) -> None:
    if _HAS_RICH:
        _console.print(msg, **kwargs)
    else:
        # Strip basic rich markup for plain output
        import re

        plain = re.sub(r"\[/?[^\]]*\]", "", msg)
        print(plain)


def _prompt(label: str, default: str = "", choices: list[str] | None = None) -> str:
    if _HAS_RICH:
        if default:
            return Prompt.ask(label, default=default, console=_console)
        return Prompt.ask(label, console=_console, show_default=False)
    display = f"{label}"
    if default:
        display += f" [{default}]"
    display += ": "
    ans = input(display).strip()
    if not ans:
        return default
    if choices and ans not in choices:
        print(f"  Invalid choice: {ans}. Options: {', '.join(choices)}")
        return _prompt(label, default, choices)
    return ans


def _banner(text: str) -> None:
    if _HAS_RICH:
        _console.print(Panel.fit(text, border_style="bold blue"))
    else:
        width = len(text) + 4
        print("=" * width)
        print(f"  {text}")
        print("=" * width)


def _section(title: str) -> None:
    if _HAS_RICH:
        _console.print(f"\n[bold cyan]{title}[/bold cyan]")
        _console.print(Rule(style="dim"))
    else:
        print(f"\n--- {title} ---")


def _option_table(options: list[tuple[str, str, str]]) -> None:
    """Show options as table: (key, description, extras_needed)."""
    if _HAS_RICH:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Choice", style="bold green", no_wrap=True)
        table.add_column("Description")
        table.add_column("Extra", style="dim")
        for key, desc, extra in options:
            table.add_row(key, desc, extra or "base")
        _console.print(table)
    else:
        for key, desc, extra in options:
            tag = f" (requires: pip install rag7[{extra}])" if extra else ""
            print(f"  {key:<14} {desc}{tag}")


# ── Dependency checker ────────────────────────────────────────────────────────


def _check_import(module: str, pip_extra: str) -> bool:
    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        _print(
            f"[bold red]Missing:[/bold red] {module}. "
            f"Install with: [bold]pip install rag7[{pip_extra}][/bold]"
        )
        return False


# ── Embedder builders ─────────────────────────────────────────────────────────


def _build_openai_embed() -> Callable[[str], list[float]] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _print(
            "[yellow]OPENAI_API_KEY not set. Set it or choose another embedder.[/yellow]"
        )
        return None
    model = _prompt("  Embedding model", default="text-embedding-3-small")
    import requests

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )

    def _embed(text: str) -> list[float]:
        resp = session.post(
            "https://api.openai.com/v1/embeddings",
            json={"input": [text], "model": model},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    return _embed


def _build_azure_embed() -> Callable[[str], list[float]] | None:
    from .utils import _make_azure_embed_fn

    fn = _make_azure_embed_fn()
    if fn is None:
        _print(
            "[yellow]Azure OpenAI env vars not set "
            "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY).[/yellow]"
        )
    return fn


def _build_ollama_embed() -> Callable[[str], list[float]] | None:
    url = _prompt("  Ollama URL", default="http://localhost:11434")
    model = _prompt("  Embedding model", default="nomic-embed-text")
    import requests

    def _embed(text: str) -> list[float]:
        resp = requests.post(
            f"{url}/api/embed",
            json={"model": model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    # Test connection
    try:
        _embed("test")
        return _embed
    except Exception as exc:
        _print(f"[yellow]Ollama connection failed: {exc}[/yellow]")
        return None


_EMBEDDER_OPTIONS: list[tuple[str, str, str]] = [
    ("openai", "OpenAI API (text-embedding-3-small)", ""),
    ("azure", "Azure OpenAI (from env vars)", ""),
    ("ollama", "Ollama local (nomic-embed-text, etc.)", ""),
    ("none", "No embeddings (BM25/keyword only)", ""),
]

_EMBEDDER_BUILDERS: dict[str, Callable[[], Callable[[str], list[float]] | None]] = {
    "openai": _build_openai_embed,
    "azure": _build_azure_embed,
    "ollama": _build_ollama_embed,
}

# ── Backend options ───────────────────────────────────────────────────────────

_BACKEND_OPTIONS: list[tuple[str, str, str]] = [
    ("memory", "In-memory (no setup, great for testing)", ""),
    ("duckdb", "DuckDB (local file, fast)", "duckdb"),
    ("chromadb", "ChromaDB (local or server)", "chromadb"),
    ("meilisearch", "Meilisearch (full-text + vector)", "meilisearch"),
    ("qdrant", "Qdrant (vector DB, local or cloud)", "qdrant"),
    ("pgvector", "PostgreSQL + pgvector", "pgvector"),
    ("lancedb", "LanceDB (columnar vector DB)", "lancedb"),
    ("azure", "Azure AI Search", "azure"),
]

_BACKEND_MODULES: dict[str, str] = {
    "duckdb": "duckdb",
    "chromadb": "chromadb",
    "meilisearch": "meilisearch",
    "qdrant": "qdrant_client",
    "pgvector": "psycopg",
    "lancedb": "lancedb",
    "azure": "azure.search.documents",
}

# ── Reranker options ──────────────────────────────────────────────────────────

_RERANKER_OPTIONS: list[tuple[str, str, str]] = [
    ("none", "No reranker (skip reranking step)", ""),
    ("cohere", "Cohere Rerank (API, best quality)", "cohere"),
    ("jina", "Jina Reranker (API)", "jina"),
    ("huggingface", "HuggingFace cross-encoder (local)", "huggingface"),
    ("llm", "LLM-based reranker (uses your chat model)", ""),
]

# ── LLM options ───────────────────────────────────────────────────────────────

_LLM_OPTIONS: list[tuple[str, str, str]] = [
    ("openai", "OpenAI (gpt-4o, gpt-4o-mini, ...)", ""),
    ("anthropic", "Anthropic (claude-sonnet-4-6, ...)", ""),
    ("ollama", "Ollama local (llama3, mistral, ...)", ""),
    ("env", "Use default from env vars", ""),
]


# ── Wizard ────────────────────────────────────────────────────────────────────


def _env_defaults() -> dict[str, str]:
    """Infer wizard defaults from env vars. Each key maps to a default choice
    we'll pre-select in the relevant prompt. Missing keys fall back to hardcoded
    defaults in the wizard."""
    d: dict[str, str] = {}

    # LLM provider
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        d["llm"] = "env"  # env-driven azure via init_agent
    elif os.getenv("OPENAI_API_KEY"):
        d["llm"] = "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        d["llm"] = "anthropic"
    elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL"):
        d["llm"] = "ollama"

    # Embedder
    if os.getenv("AZURE_OPENAI_ENDPOINT") and (
        os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    ):
        d["embedder"] = "azure"
    elif os.getenv("OPENAI_API_KEY"):
        d["embedder"] = "openai"
    elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL"):
        d["embedder"] = "ollama"

    # Backend
    if os.getenv("MEILI_URL"):
        d["backend"] = "meilisearch"
        d["meili_url"] = os.getenv("MEILI_URL", "")
        d["meili_key"] = os.getenv("MEILI_KEY", "") or os.getenv("MEILI_MASTER_KEY", "")
    elif os.getenv("QDRANT_URL"):
        d["backend"] = "qdrant"
        d["qdrant_url"] = os.getenv("QDRANT_URL", "")
    elif os.getenv("DATABASE_URL") or os.getenv("PGVECTOR_DSN"):
        d["backend"] = "pgvector"
        d["pg_dsn"] = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL") or ""
    elif os.getenv("AZURE_SEARCH_ENDPOINT"):
        d["backend"] = "azure"
        d["azure_search_endpoint"] = os.getenv("AZURE_SEARCH_ENDPOINT", "")

    # Reranker
    if os.getenv("COHERE_API_KEY"):
        d["reranker"] = "cohere"
    elif os.getenv("JINA_API_KEY"):
        d["reranker"] = "jina"

    # Index
    if os.getenv("MS_INDEX"):
        d["index"] = os.getenv("MS_INDEX", "")

    return d


def _prompt_with_env_hint(
    label: str,
    default: str,
    env_default: str | None,
    secret: bool = False,
) -> str:
    """Prompt with '(from env)' hint. When secret=True the env value is not
    shown on screen — user presses Enter to accept the hidden default."""
    if env_default:
        if secret:
            label = f"{label} [dim](from env — hidden)[/dim]"
            if _HAS_RICH:
                ans = Prompt.ask(
                    label, default="", console=_console, show_default=False
                )
            else:
                ans = input(f"{label}: ").strip()
            return ans or env_default
        label = f"{label} [dim](from env)[/dim]"
        default = env_default
    return _prompt(label, default=default)


def _wizard() -> dict[str, Any]:
    """Interactive setup wizard. Returns kwargs for init_agent.

    Env-driven defaults: when an env var implies a provider/backend/index,
    the wizard pre-selects it so pressing Enter accepts the env-configured
    choice.
    """
    _banner("rag7 — Setup Wizard")
    envd = _env_defaults()
    if envd:
        detected = ", ".join(sorted(envd.keys()))
        _print(f"[dim]Detected from env: {detected}[/dim]")

    # ── 1. LLM ────────────────────────────────────────────────────────────────
    _section("1. Chat / Generation LLM")
    _option_table(_LLM_OPTIONS)
    llm_choice = _prompt_with_env_hint(
        "  Choose LLM provider", default="openai", env_default=envd.get("llm")
    )

    model: str | None = None
    if llm_choice == "openai":
        model_name = _prompt("  Model name", default="gpt-4o-mini")
        model = f"openai:{model_name}"
    elif llm_choice == "anthropic":
        model_name = _prompt("  Model name", default="claude-sonnet-4-6")
        model = f"anthropic:{model_name}"
    elif llm_choice == "ollama":
        model_name = _prompt("  Model name", default="llama3")
        model = f"ollama:{model_name}"
    # else: env → model=None, uses env default

    # ── 2. Embedder ───────────────────────────────────────────────────────────
    _section("2. Embedding Model")
    _option_table(_EMBEDDER_OPTIONS)
    embed_choice = _prompt("  Choose embedder", default="openai")

    embed_fn: Callable[[str], list[float]] | None = None
    if embed_choice in _EMBEDDER_BUILDERS:
        embed_fn = _EMBEDDER_BUILDERS[embed_choice]()
        if embed_fn is None and embed_choice != "none":
            _print("[dim]Falling back to no embeddings (BM25 only).[/dim]")

    # ── 3. Backend ────────────────────────────────────────────────────────────
    _section("3. Vector Store / Backend")
    _option_table(_BACKEND_OPTIONS)
    backend_choice = _prompt_with_env_hint(
        "  Choose backend", default="memory", env_default=envd.get("backend")
    )

    backend_url: str | None = None
    backend_kwargs: dict[str, Any] = {}
    index = "default"

    # Check dependency
    if backend_choice in _BACKEND_MODULES:
        mod = _BACKEND_MODULES[backend_choice]
        if not _check_import(mod, backend_choice):
            _print("[dim]Falling back to in-memory backend.[/dim]")
            backend_choice = "memory"

    if backend_choice == "memory":
        index = "in_memory"
        _print("  [dim]In-memory backend selected. Load data after setup.[/dim]")
    elif backend_choice == "meilisearch":
        backend_url = _prompt_with_env_hint(
            "  Meilisearch URL",
            default="http://localhost:7700",
            env_default=envd.get("meili_url"),
        )
        api_key = _prompt_with_env_hint(
            "  API key",
            default="masterKey",
            env_default=envd.get("meili_key"),
            secret=True,
        )
        backend_kwargs["api_key"] = api_key
        index = _prompt_with_env_hint(
            "  Index name", default="documents", env_default=envd.get("index")
        )
    elif backend_choice == "qdrant":
        backend_url = _prompt_with_env_hint(
            "  Qdrant URL",
            default="http://localhost:6333",
            env_default=envd.get("qdrant_url"),
        )
        index = _prompt_with_env_hint(
            "  Collection name", default="documents", env_default=envd.get("index")
        )
    elif backend_choice == "pgvector":
        dsn = _prompt_with_env_hint(
            "  PostgreSQL DSN",
            default="postgresql://localhost:5432/ragdb",
            env_default=envd.get("pg_dsn"),
            secret=True,
        )
        backend_kwargs["dsn"] = dsn
        index = _prompt_with_env_hint(
            "  Table name", default="documents", env_default=envd.get("index")
        )
    elif backend_choice == "chromadb":
        index = _prompt("  Collection name", default="documents")
    elif backend_choice == "duckdb":
        db_path = _prompt("  DB path", default=":memory:")
        if db_path != ":memory:":
            backend_kwargs["db_path"] = db_path
        index = _prompt("  Table name", default="documents")
    elif backend_choice == "lancedb":
        db_path = _prompt("  DB path", default="./lancedb")
        backend_kwargs["db_path"] = db_path
        index = _prompt("  Table name", default="documents")
    elif backend_choice == "azure":
        backend_url = _prompt(
            "  Azure Search endpoint",
            default=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        )
        index = _prompt("  Index name", default="documents")

    # ── 3b. Multi-collection (optional) ───────────────────────────────────────
    collections: list[str] | None = None
    _section("3b. Multi-collection (optional)")
    _print(
        "  [dim]Query multiple indexes/collections at once. Leave blank for single.[/dim]"
    )
    multi = _prompt(
        "  Extra collections (comma-separated) or blank", default=""
    ).strip()
    if multi:
        extras = [c.strip() for c in multi.split(",") if c.strip()]
        collections = [index, *extras] if index not in extras else extras

    # ── 4. Reranker ───────────────────────────────────────────────────────────
    _section("4. Reranker")
    _option_table(_RERANKER_OPTIONS)
    reranker_choice = _prompt_with_env_hint(
        "  Choose reranker", default="none", env_default=envd.get("reranker")
    )

    reranker: str | None = None
    reranker_model: str | None = None
    if reranker_choice != "none":
        reranker = reranker_choice
        if reranker_choice == "huggingface":
            reranker_model = _prompt(
                "  Model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        elif reranker_choice == "cohere":
            if not os.getenv("COHERE_API_KEY"):
                _print("[yellow]COHERE_API_KEY not set.[/yellow]")

    return {
        "index": index,
        "collections": collections,
        "model": model,
        "backend": backend_choice,
        "backend_url": backend_url,
        "backend_kwargs": backend_kwargs or None,
        "embed_fn": embed_fn,
        "reranker": reranker,
        "reranker_model": reranker_model,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_documents(rag: AgenticRAG) -> None:
    """Optionally load documents from a JSON file into the backend."""
    _section("Load Documents")
    _print("  Load documents from a JSON file?")
    _print("  Format: list of objects, each with text fields.")
    path = _prompt("  File path (or skip)", default="skip")
    if path == "skip":
        return

    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        _print(f"[red]File not found: {path}[/red]")
        return

    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            _print("[red]Expected a JSON array of objects.[/red]")
            return
        _print(f"  Loading {len(data)} documents...")
        rag.backend.add_documents(data)
        _print(f"  [bold green]Loaded {len(data)} documents.[/bold green]")
    except Exception as exc:
        _print(f"[red]Error loading: {exc}[/red]")


# ── Citation formatting ───────────────────────────────────────────────────────


def format_citation_line(i: int, doc: Document) -> str:
    title = doc.metadata.get("title") or doc.metadata.get("name") or ""
    url = doc.metadata.get("url") or doc.metadata.get("link") or ""
    source = doc.metadata.get("source") or ""
    label = title or source or doc.page_content[:60].replace("\n", " ") + "..."
    suffix = f" -- {url}" if url else ""
    return f"[{i}] {label}{suffix}"


# ── Rendering -----------------------------------------------------------------


def render_sources(docs: list[Document]) -> None:
    if not docs:
        return
    if _HAS_RICH:
        _console.print(Rule("[dim]Sources[/dim]"))
        for i, doc in enumerate(docs, 1):
            line = format_citation_line(i, doc)
            _console.print(f"  [dim cyan]{line}[/dim cyan]")
    else:
        print("--- Sources ---")
        for i, doc in enumerate(docs, 1):
            line = format_citation_line(i, doc)
            print(f"  {line}")


def render_answer(answer: str, docs: list[Document]) -> None:
    if _HAS_RICH:
        _console.print()
        _console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
                expand=True,
            )
        )
    else:
        print(f"\n=== Answer ===\n{answer}\n")
    render_sources(docs)


# ── Retriever mode ────────────────────────────────────────────────────────────


def run_retriever(rag: AgenticRAG, question: str, top_k: int | None = None) -> None:
    if _HAS_RICH:
        with _console.status(
            "[bold yellow]Retrieving...[/bold yellow]", spinner="dots"
        ):
            query, docs = rag.retrieve_documents(question, top_k=top_k)
    else:
        print("Retrieving...")
        query, docs = rag.retrieve_documents(question, top_k=top_k)

    _print(f"\n  Search query: {query}\n")

    if not docs:
        _print("  No documents found.")
        return

    for i, doc in enumerate(docs, 1):
        title = (
            doc.metadata.get("name")
            or doc.metadata.get("title")
            or doc.page_content[:80].replace("\n", " ")
        )
        if _HAS_RICH:
            table = Table(
                title=f"#{i}  {title}",
                show_header=True,
                header_style="bold cyan",
                title_style="bold",
                title_justify="left",
                box=None,
                pad_edge=False,
            )
            table.add_column("Field", style="dim", no_wrap=True)
            table.add_column("Value", overflow="fold")
            for k, v in doc.metadata.items():
                if v in (None, "", []):
                    continue
                sv = str(v)
                if len(sv) > 200:
                    sv = sv[:200] + "…"
                table.add_row(str(k), sv)
            if doc.page_content:
                snippet = doc.page_content[:300].replace("\n", " ")
                if len(doc.page_content) > 300:
                    snippet += "…"
                table.add_row("[bold]content[/bold]", snippet)
            _console.print(table)
            _console.print()
        else:
            print(f"\n#{i} {title}")
            for k, v in doc.metadata.items():
                if v in (None, "", []):
                    continue
                print(f"  {k}: {v}")
            if doc.page_content:
                print(f"  content: {doc.page_content[:300]}")


# ── Chat runner ───────────────────────────────────────────────────────────────


def run_chat(
    rag: AgenticRAG,
    question: str,
    history: list[ConversationTurn],
) -> ConversationTurn | None:
    if _HAS_RICH:
        with _console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            state = rag.chat(question, history)
    else:
        print("Thinking...")
        state = rag.chat(question, history)

    # LangGraph ainvoke may return a dict at runtime despite the type hint.
    get = (
        state.get if isinstance(state, dict) else lambda k, d=None: getattr(state, k, d)
    )
    answer = get("answer") or ""
    docs = get("documents") or []
    query = get("query") or question

    _print(f"\n  Search query: {query}")

    if not answer:
        _print("  No answer generated.")
        return None

    render_answer(answer, docs)
    return ConversationTurn(question=question, answer=answer)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="rag7 — Agentic RAG CLI")
    parser.add_argument(
        "--collection", "-c", default=None, help="Index/collection name"
    )
    parser.add_argument(
        "--collections",
        default=None,
        help="Comma-separated collection names for multi-index search "
        "(e.g. 'docs,faq'). Each may be 'name:description' for LLM routing.",
    )
    parser.add_argument("--top-k", "-k", type=int, default=None, help="Max documents")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--chat", dest="mode", action="store_const", const="chat", help="Chat mode"
    )
    mode_group.add_argument(
        "--retriever",
        dest="mode",
        action="store_const",
        const="retriever",
        help="Retriever mode",
    )
    mode_group.add_argument("--mode", "-m", choices=["chat", "retriever"], default=None)

    parser.add_argument(
        "--skip-wizard",
        action="store_true",
        help="Skip setup wizard (use env defaults). Implied when env is sufficient.",
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Force interactive wizard even if env vars are set.",
    )

    args = parser.parse_args()

    # ── Wizard or env defaults ────────────────────────────────────────────────
    # Default: skip wizard when env provides enough to build an agent.
    # Force wizard only with --wizard. --skip-wizard kept for back-compat.
    envd = _env_defaults()
    env_sufficient = bool(envd.get("backend") or envd.get("llm") or envd.get("index"))
    use_wizard = args.wizard or (not env_sufficient and not args.skip_wizard)

    if not use_wizard:
        from .utils import _make_azure_embed_fn

        # Ask only for what env/flags don't provide.
        index = args.collection or envd.get("index")
        if not index:
            index = _prompt("  Index / collection name", default="documents")

        collections_cfg: list[str] | dict[str, str] | None = None
        if not args.collections:
            multi = _prompt(
                "  Extra collections (comma-sep, blank=single)", default=""
            ).strip()
            if multi:
                extras = [c.strip() for c in multi.split(",") if c.strip()]
                collections_cfg = [index, *extras] if index not in extras else extras

        config: dict[str, Any] = {
            "index": index,
            "embed_fn": _make_azure_embed_fn(),
        }
        if collections_cfg:
            config["collections"] = collections_cfg
        if envd.get("backend") == "meilisearch":
            config["backend"] = "meilisearch"
            config["backend_url"] = envd.get("meili_url")
            if envd.get("meili_key"):
                config["backend_kwargs"] = {"api_key": envd["meili_key"]}
        elif envd.get("backend") == "qdrant":
            config["backend"] = "qdrant"
            config["backend_url"] = envd.get("qdrant_url")
        elif envd.get("backend") == "pgvector":
            config["backend"] = "pgvector"
            config["backend_kwargs"] = {"dsn": envd["pg_dsn"]}
        elif envd.get("backend") == "azure":
            config["backend"] = "azure"
            config["backend_url"] = envd.get("azure_search_endpoint")
        if envd.get("reranker"):
            config["reranker"] = envd["reranker"]
        detected = ", ".join(sorted(envd.keys())) if envd else "none"
        _print(f"[dim]Using env config ({detected}). --wizard to override.[/dim]")
    else:
        config = _wizard()

    if args.collection:
        # Accept comma-separated list in -c as a convenience alias for --collections
        if "," in args.collection:
            parts = [p.strip() for p in args.collection.split(",") if p.strip()]
            config["index"] = parts[0]
            config["collections"] = parts
        else:
            config["index"] = args.collection

    if args.collections:
        parts = [p.strip() for p in args.collections.split(",") if p.strip()]
        if any(":" in p for p in parts):
            config["collections"] = {
                (p.split(":", 1)[0].strip()): (
                    p.split(":", 1)[1].strip() if ":" in p else p.strip()
                )
                for p in parts
            }
        else:
            config["collections"] = parts

    # ── Build agent ───────────────────────────────────────────────────────────
    _section("Initialising")
    _print("  Building RAG agent...")

    from .factory import init_agent

    try:
        rag = init_agent(**{k: v for k, v in config.items() if v is not None})
    except Exception as exc:
        _print(f"[bold red]Failed to initialise: {exc}[/bold red]")
        sys.exit(1)

    # ── Load data for in-memory backend ───────────────────────────────────────
    if config.get("backend") == "memory":
        _load_documents(rag)

    # ── Mode selection ────────────────────────────────────────────────────────
    mode: str = args.mode or "chat"
    if args.mode is None and use_wizard:
        _section("5. Mode")
        _print("  [bold]chat[/bold]       — Ask questions, get answers with citations")
        _print("  [bold]retriever[/bold]  — Search only, no LLM generation")
        mode = _prompt("  Choose mode", default="chat")

    cols = config.get("collections")
    if cols:
        cols_str = ", ".join(cols if isinstance(cols, list) else cols.keys())
        coll_line = f"Collections: {cols_str}"
        prompt_coll = cols_str if len(cols_str) <= 40 else f"{len(cols)} collections"
    else:
        prompt_coll = str(config.get("index", "?"))
        coll_line = f"Collection: {prompt_coll}"
    _print(f"\n  [bold green]Ready![/bold green] Mode: {mode}  |  {coll_line}")
    _print(
        "  Type your question. [bold]q[/bold] to quit, [bold]clear[/bold] to reset history.\n"
    )

    # ── REPL ──────────────────────────────────────────────────────────────────
    history: list[ConversationTurn] = []

    while True:
        if mode == "chat" and history:
            _print(
                f"  [dim]({len(history)} turn{'s' if len(history) != 1 else ''} in context)[/dim]"
            )

        try:
            question = _prompt(f"[dim]{prompt_coll}[/dim] [bold blue]>[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            _print("\nBye.")
            sys.exit(0)

        q = question.strip()
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            _print("Bye.")
            break
        if mode == "chat" and q.lower() == "clear":
            history.clear()
            _print("  History cleared.\n")
            continue

        try:
            if mode == "retriever":
                run_retriever(rag, q, top_k=args.top_k)
            else:
                turn = run_chat(rag, q, history)
                if turn:
                    history.append(turn)
        except Exception as exc:
            _print(f"[bold red]Error:[/bold red] {exc}")
        _print()


if __name__ == "__main__":
    main()
