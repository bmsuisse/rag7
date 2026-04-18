from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Collection names active for the current search context. Set by AgenticRAG
# after LLM-routing selects a subset; read by _MultiBackend to fan out only
# to those backends. None / empty list = use all registered collections.
_ACTIVE_COLLECTIONS: ContextVar[list[str] | None] = ContextVar(
    "rag_active_collections", default=None
)

# ── SQL injection protection ──────────────────────────────────────────────────

_SAFE_FILTER_RE = re.compile(
    r"^[\w\s.,=<>!'\"\-()%]+$",
    re.IGNORECASE,
)

_DANGEROUS_RE = re.compile(
    r"\b(DROP|ALTER|INSERT|UPDATE|DELETE|EXEC|UNION)\b|;|--",
    re.IGNORECASE,
)

_IDENT_RE = re.compile(r"^[a-zA-Z_]\w*$")


def _validate_filter_expr(expr: str | dict[str, Any] | None) -> str | None:
    """Reject filter expressions containing SQL injection patterns.

    Dict-form filters are only accepted by backends with native dict APIs
    (Chroma/Qdrant). SQL/Meili/OData backends require string form.
    """
    if not expr:
        return None
    if isinstance(expr, dict):
        raise ValueError(
            "dict filter_expr is not supported by this backend; pass a string"
        )
    if not _SAFE_FILTER_RE.match(expr):
        raise ValueError(f"Unsafe filter expression rejected: {expr!r}")
    if _DANGEROUS_RE.search(expr):
        raise ValueError(f"Unsafe filter expression rejected: {expr!r}")
    return expr


def _validate_identifiers(names: list[str]) -> list[str]:
    """Validate that column/table names are safe SQL identifiers."""
    for name in names:
        if not _IDENT_RE.match(name):
            raise ValueError(f"Unsafe identifier rejected: {name!r}")
    return names


@dataclass
class SearchRequest:
    """Backend-agnostic search request."""

    query: str
    limit: int
    vector: list[float] | None = None
    semantic_ratio: float = 0.0
    # String form for Meili/SQL/OData dialects, dict form for Chroma/Qdrant
    # native filters. Each backend normalizes in its own search().
    filter_expr: str | dict[str, Any] | None = None
    sort_fields: list[str] | None = None
    show_ranking_score: bool = False
    matching_strategy: str = "frequency"
    embedder_name: str = ""
    index_uid: str | None = None


@dataclass
class IndexConfig:
    """Backend-agnostic index metadata."""

    filterable_attributes: list[str] = field(default_factory=list)
    searchable_attributes: list[str] = field(default_factory=list)
    sortable_attributes: list[str] = field(default_factory=list)
    ranking_rules: list[str] = field(default_factory=list)
    embedders: list[str] = field(default_factory=list)


def _distance_rows_to_hits(
    cols: list[str],
    rows: list[Any],
    vector_col: str,
) -> list[dict]:
    """Convert SQL rows with _distance column to hit dicts with _rankingScore."""
    hits: list[dict] = []
    for row in rows:
        hit = dict(zip(cols, row))
        dist = hit.pop("_distance", None)
        if dist is not None:
            hit["_rankingScore"] = 1.0 / (1.0 + float(dist))
        hit.pop(vector_col, None)
        hits.append(hit)
    return hits


class SearchBackend(ABC):
    """Abstract search backend."""

    _embed_fn: Any = None

    def _resolve_vector(self, request: SearchRequest) -> list[float] | None:
        """Return vector from request or by calling embed_fn, or None."""
        return request.vector or (
            self._embed_fn(request.query) if self._embed_fn else None
        )

    @abstractmethod
    def search(self, request: SearchRequest) -> list[dict]: ...

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        """Default: sequential search. Override for native batch support."""
        return [self.search(r) for r in requests]

    @abstractmethod
    def get_index_config(self) -> IndexConfig: ...

    @abstractmethod
    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]: ...

    # ── Filter translation ────────────────────────────────────────────────
    # Default dialect: Meilisearch (`=`, `!=`, `NOT field CONTAINS "x"`).
    # SQL backends and Azure override build_filter_expr with their syntax.
    #
    # Kept narrow on purpose: takes the small set of operators that
    # core._build_filter_expr previously emitted inline. Value escaping is
    # applied per dialect to avoid injection of quote characters.

    def build_filter_expr(self, intent: Any) -> str:
        """Meili-dialect filter builder (default).

        `intent` is a FilterIntent-like object with `field`, `value`,
        `operator`, `extra_excludes` attributes.
        """
        op = intent.operator
        field_name = intent.field
        value = _escape_meili_value(intent.value)
        if op == "NOT_CONTAINS":
            parts = [f'NOT {field_name} CONTAINS "{value}"']
            for extra in intent.extra_excludes:
                parts.append(
                    f'NOT {field_name} CONTAINS "{_escape_meili_value(extra)}"'
                )
            return " AND ".join(parts)
        return f'{field_name} {op} "{value}"'


def _escape_meili_value(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"')


def _escape_sql_value(v: str) -> str:
    return v.replace("'", "''")


def _build_sql_filter(intent: Any) -> str:
    """SQL-dialect filter (LanceDB/DataFusion, DuckDB, pgvector).

    Maps Meili CONTAINS → ILIKE '%v%' (case-insensitive LIKE).
    """
    op = intent.operator
    field_name = intent.field
    value = _escape_sql_value(intent.value)
    if op == "NOT_CONTAINS":
        parts = [f"{field_name} NOT ILIKE '%{value}%'"]
        for extra in intent.extra_excludes:
            parts.append(f"{field_name} NOT ILIKE '%{_escape_sql_value(extra)}%'")
        return " AND ".join(parts)
    return f"{field_name} {op} '{value}'"


def _build_odata_filter(intent: Any) -> str:
    """Azure AI Search OData filter dialect."""
    op = intent.operator
    op_map = {"=": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
    field_name = intent.field
    value = intent.value.replace("'", "''")
    if op == "NOT_CONTAINS":
        parts = [f"not search.ismatch('{value}', '{field_name}')"]
        for extra in intent.extra_excludes:
            esc = extra.replace("'", "''")
            parts.append(f"not search.ismatch('{esc}', '{field_name}')")
        return " and ".join(parts)
    return f"{field_name} {op_map.get(op, op)} '{value}'"


class MeilisearchBackend(SearchBackend):
    def __init__(
        self,
        index: str,
        url: str | None = None,
        api_key: str | None = None,
    ):
        import meilisearch

        self.index = index
        self._client = meilisearch.Client(
            url or os.getenv("MEILI_URL", "http://localhost:7700"),
            api_key or os.getenv("MEILI_KEY", "masterKey"),
        )

    @property
    def client(self) -> Any:
        return self._client

    def _to_native_params(self, req: SearchRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": req.query,
            "limit": req.limit,
            "matchingStrategy": req.matching_strategy,
        }
        if req.index_uid:
            params["indexUid"] = req.index_uid
        if req.filter_expr:
            params["filter"] = req.filter_expr
        if req.sort_fields:
            params["sort"] = req.sort_fields
        if req.vector and req.embedder_name:
            params["vector"] = req.vector
            params["hybrid"] = {
                "embedder": req.embedder_name,
                "semanticRatio": req.semantic_ratio,
            }
        if req.show_ranking_score:
            params["showRankingScore"] = True
        return params

    def search(self, request: SearchRequest) -> list[dict]:
        params = self._to_native_params(request)
        params.pop("indexUid", None)
        try:
            return self._client.index(self.index).search(request.query, params)["hits"]
        except Exception:
            params.pop("vector", None)
            params.pop("hybrid", None)
            try:
                return self._client.index(self.index).search(request.query, params)[
                    "hits"
                ]
            except Exception:
                return []

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        queries = [self._to_native_params(r) for r in requests]
        for q in queries:
            if "indexUid" not in q:
                q["indexUid"] = self.index
        try:
            results = self._client.multi_search(queries)["results"]
            return [r["hits"] for r in results]
        except Exception:
            for q in queries:
                q.pop("vector", None)
                q.pop("hybrid", None)
                q["matchingStrategy"] = "last"
            try:
                results = self._client.multi_search(queries)["results"]
                return [r["hits"] for r in results]
            except Exception:
                return []

    def get_index_config(self) -> IndexConfig:
        try:
            settings = self._client.index(self.index).get_settings()
            return IndexConfig(
                filterable_attributes=settings.get("filterableAttributes", []),
                searchable_attributes=settings.get("searchableAttributes", []),
                sortable_attributes=settings.get("sortableAttributes", []),
                ranking_rules=settings.get("rankingRules", []),
                embedders=list(settings.get("embedders", {}).keys()),
            )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if filter_expr:
            params["filter"] = filter_expr
        if attributes_to_retrieve:
            params["attributesToRetrieve"] = attributes_to_retrieve
        try:
            return self._client.index(self.index).search("", params)["hits"] or []
        except Exception:
            return []


def _wrap_embed_fn(embed_fn: Any) -> Any:
    """Wrap a ``str → list[float]`` callable into ChromaDB's EmbeddingFunction protocol."""
    from chromadb import Documents, EmbeddingFunction, Embeddings

    class _WrappedEF(EmbeddingFunction):  # type: ignore[type-arg]
        def __init__(self, fn: Any = None) -> None:
            self._fn = fn or embed_fn

        def __call__(self, input: Documents) -> Embeddings:
            return [self._fn(doc) for doc in input]

        @staticmethod
        def name() -> str:
            return "custom"

        def get_config(self) -> dict[str, Any]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> "_WrappedEF":
            return _WrappedEF()

    return _WrappedEF()


_CHROMA_EQ_RE = re.compile(r'^\s*(\w+)\s*(=|!=)\s*"([^"]*)"\s*$')
_CHROMA_NOT_CONTAINS_RE = re.compile(r'^\s*NOT\s+(\w+)\s+CONTAINS\s+"([^"]*)"\s*$')


def _chroma_filter_from_str(expr: str) -> dict[str, Any] | None:
    """Parse the Meili-dialect filter string into a ChromaDB `where` dict.

    Supports: `field = "value"`, `field != "value"`, clauses joined by
    ` AND `. `NOT field CONTAINS "x"` has no metadata equivalent in
    Chroma — the clause is dropped and the caller gets a best-effort
    filter built from the remaining clauses.

    Returns None when nothing could be parsed.
    """
    clauses: list[dict[str, Any]] = []
    for part in re.split(r"\s+AND\s+", expr):
        m = _CHROMA_EQ_RE.match(part)
        if m:
            field, op, value = m.group(1), m.group(2), m.group(3)
            clauses.append({field: value} if op == "=" else {field: {"$ne": value}})
            continue
        if _CHROMA_NOT_CONTAINS_RE.match(part):
            # Chroma metadata filters have no substring operator; dropping.
            continue
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaDBBackend(SearchBackend):
    """ChromaDB backend. Requires `pip install chromadb`."""

    def __init__(
        self,
        collection: str,
        embed_fn: Any = None,
        *,
        host: str | None = None,
        port: int = 8000,
        path: str | None = None,
    ):
        import chromadb

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.Client()

        ef = _wrap_embed_fn(embed_fn) if embed_fn else None
        self._collection = self._client.get_or_create_collection(
            collection,
            embedding_function=ef,
        )
        self._embed_fn = embed_fn
        self.index = collection

    def search(self, request: SearchRequest) -> list[dict]:
        kwargs: dict[str, Any] = {"n_results": request.limit}

        if request.vector:
            kwargs["query_embeddings"] = [request.vector]
        else:
            kwargs["query_texts"] = [request.query]

        where = self._prepare_where(request.filter_expr)
        if where is not None:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return self._to_hits(results)

    @staticmethod
    def _prepare_where(filter_expr: Any) -> dict[str, Any] | None:
        """Chroma expects dict filters; parse string filters we emit."""
        if filter_expr is None:
            return None
        if isinstance(filter_expr, dict):
            return filter_expr
        if isinstance(filter_expr, str):
            return _chroma_filter_from_str(filter_expr)
        return None

    def get_index_config(self) -> IndexConfig:
        metadata = self._collection.metadata or {}
        return IndexConfig(
            filterable_attributes=list(metadata.keys()),
            searchable_attributes=["documents"],
            embedders=["default"] if self._embed_fn else [],
        )

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        where = self._prepare_where(filter_expr)
        try:
            if where is not None:
                results = self._collection.get(where=where, limit=limit)
                return self._to_hits(results, nested=False)
            results = self._collection.peek(limit=limit)
            return self._to_hits(results, nested=False)
        except Exception:
            return []

    @staticmethod
    def _to_hits(results: Any, *, nested: bool = True) -> list[dict]:
        ids = results.get("ids", [[]])[0] if nested else results.get("ids", [])
        docs = (
            results.get("documents", [[]])[0]
            if nested
            else results.get("documents", [])
        )
        metadatas = (
            results.get("metadatas", [[]])[0]
            if nested
            else results.get("metadatas", [])
        )
        distances = results.get("distances", [[]])[0] if nested else []
        hits: list[dict] = []
        for i, doc_id in enumerate(ids):
            hit: dict[str, Any] = {"id": doc_id, "content": docs[i] if docs else ""}
            if metadatas and metadatas[i]:
                hit.update(metadatas[i])
            if distances:
                hit["_rankingScore"] = 1.0 / (1.0 + distances[i])
            hits.append(hit)
        return hits


class LanceDBBackend(SearchBackend):
    """LanceDB backend. Requires `pip install lancedb`."""

    def build_filter_expr(self, intent: Any) -> str:
        return _build_sql_filter(intent)

    def __init__(
        self,
        table: str,
        db_uri: str = "~/.lancedb",
        embed_fn: Any = None,
        text_column: str = "content",
        vector_column: str = "vector",
    ):
        import lancedb

        self._db = lancedb.connect(db_uri)
        self._table = self._db.open_table(table)
        self._embed_fn = embed_fn
        self._text_col = text_column
        self._vector_col = vector_column
        self.index = table

    def search(self, request: SearchRequest) -> list[dict]:
        try:
            if request.vector:
                results = self._table.search(
                    request.vector, vector_column_name=self._vector_col
                ).limit(request.limit)
            elif self._embed_fn:
                vec = self._embed_fn(request.query)
                results = self._table.search(
                    vec, vector_column_name=self._vector_col
                ).limit(request.limit)
            else:
                results = self._table.search(request.query).limit(request.limit)

            if request.filter_expr:
                results = results.where(request.filter_expr)

            # Avoid to_pandas() — pandas is not a hard dep.
            rows = results.to_arrow().to_pylist()
            return self._rows_to_hits(rows)
        except Exception:
            return []

    def get_index_config(self) -> IndexConfig:
        try:
            schema = self._table.schema
            columns = [f.name for f in schema]
            return IndexConfig(
                filterable_attributes=[c for c in columns if c != self._vector_col],
                searchable_attributes=[self._text_col]
                if self._text_col in columns
                else columns,
                embedders=[self._vector_col] if self._vector_col in columns else [],
            )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        try:
            query = self._table.search().limit(limit)
            if filter_expr:
                query = query.where(filter_expr)
            rows = query.to_arrow().to_pylist()
            if attributes_to_retrieve:
                rows = [
                    {k: v for k, v in r.items() if k in attributes_to_retrieve}
                    for r in rows
                ]
            return rows
        except Exception:
            return []

    def _rows_to_hits(self, rows: list[dict]) -> list[dict]:
        hits: list[dict] = []
        for hit in rows:
            if "_distance" in hit:
                hit["_rankingScore"] = 1.0 / (1.0 + hit.pop("_distance"))
            hit.pop(self._vector_col, None)
            hits.append(hit)
        return hits


class AzureAISearchBackend(SearchBackend):
    """Azure AI Search backend. Requires `pip install azure-search-documents`."""

    def build_filter_expr(self, intent: Any) -> str:
        return _build_odata_filter(intent)

    def __init__(
        self,
        index: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        vector_field: str = "contentVector",
        embed_fn: Any = None,
        semantic_config: str | None = None,
    ):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT", "")
        api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY", "")
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index,
            credential=AzureKeyCredential(api_key),
        )
        self._vector_field = vector_field
        self._embed_fn = embed_fn
        self._semantic_config = semantic_config
        self.index = index

    def search(self, request: SearchRequest) -> list[dict]:
        from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery

        kwargs: dict[str, Any] = {
            "search_text": request.query or None,
            "top": request.limit,
        }

        if self._embed_fn is None and request.query:
            # No client-side embedder — use native integrated vectorization
            kwargs["vector_queries"] = [
                VectorizableTextQuery(
                    text=request.query,
                    k_nearest_neighbors=request.limit,
                    fields=self._vector_field,
                ),
            ]
        else:
            vector = self._resolve_vector(request)
            if vector:
                kwargs["vector_queries"] = [
                    VectorizedQuery(
                        vector=vector,
                        k_nearest_neighbors=request.limit,
                        fields=self._vector_field,
                    ),
                ]

        if self._semantic_config:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self._semantic_config

        if request.filter_expr:
            kwargs["filter"] = request.filter_expr  # OData filter syntax

        if request.sort_fields:
            kwargs["order_by"] = request.sort_fields

        try:
            results = self._client.search(**kwargs)
            return self._results_to_hits(results)
        except Exception:
            return []

    def get_index_config(self) -> IndexConfig:
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.indexes import SearchIndexClient

            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
            api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
            index_client = SearchIndexClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            index_def = index_client.get_index(self.index)
            filterable = [f.name for f in index_def.fields if f.filterable]
            searchable = [f.name for f in index_def.fields if f.searchable]
            sortable = [f.name for f in index_def.fields if f.sortable]
            return IndexConfig(
                filterable_attributes=filterable,
                searchable_attributes=searchable,
                sortable_attributes=sortable,
            )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        kwargs: dict[str, Any] = {"search_text": "*", "top": limit}
        if filter_expr:
            kwargs["filter"] = filter_expr
        if attributes_to_retrieve:
            kwargs["select"] = attributes_to_retrieve
        try:
            results = self._client.search(**kwargs)
            return self._results_to_hits(results)
        except Exception:
            return []

    @staticmethod
    def _results_to_hits(results: Any) -> list[dict]:
        hits: list[dict] = []
        for r in results:
            hit = dict(r)
            score = hit.pop("@search.score", None)
            if score is not None:
                hit["_rankingScore"] = float(score)
            hit.pop("@search.reranker_score", None)
            hits.append(hit)
        return hits


class PgvectorBackend(SearchBackend):
    def build_filter_expr(self, intent: Any) -> str:
        return _build_sql_filter(intent)

    """PostgreSQL + pgvector backend. Requires `pip install psycopg[binary] pgvector`."""

    def __init__(
        self,
        table: str,
        dsn: str | None = None,
        embed_fn: Any = None,
        vector_column: str = "embedding",
        content_column: str = "content",
    ):
        import psycopg
        from pgvector.psycopg import register_vector

        self._dsn = dsn or os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/postgres"
        )
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        register_vector(self._conn)
        self._table = table
        self._embed_fn = embed_fn
        self._vector_col = vector_column
        self._content_col = content_column
        self.index = table

    def search(self, request: SearchRequest) -> list[dict]:
        vector = self._resolve_vector(request)
        if not vector:
            return self._text_search(request)

        safe_filter = _validate_filter_expr(request.filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        sql = f"""
            SELECT *, ({self._vector_col} <=> %s::vector) AS _distance
            FROM {self._table}
            {where}
            ORDER BY _distance
            LIMIT %s
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (vector, request.limit))  # type: ignore[arg-type]
                cols = [desc[0] for desc in cur.description or []]
                rows = cur.fetchall()
                return self._rows_to_hits(cols, rows)
        except Exception:
            return []

    def _text_search(self, request: SearchRequest) -> list[dict]:
        safe_filter = _validate_filter_expr(request.filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        if request.query:
            ts_where = f"{'AND' if where else 'WHERE'} to_tsvector({self._content_col}) @@ plainto_tsquery(%s)"
        else:
            ts_where = ""
        sql = f"""
            SELECT * FROM {self._table}
            {where} {ts_where}
            LIMIT %s
        """
        params: list[Any] = []
        if request.query:
            params.append(request.query)
        params.append(request.limit)
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)  # type: ignore[arg-type]
                cols = [desc[0] for desc in cur.description or []]
                rows = cur.fetchall()
                return self._rows_to_hits(cols, rows)
        except Exception:
            return []

    def get_index_config(self) -> IndexConfig:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                    (self._table,),
                )
                cols = [row[0] for row in cur.fetchall()]
                return IndexConfig(
                    filterable_attributes=cols,
                    searchable_attributes=[self._content_col]
                    if self._content_col in cols
                    else cols,
                    embedders=[self._vector_col] if self._vector_col in cols else [],
                )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        if attributes_to_retrieve:
            _validate_identifiers(attributes_to_retrieve)
            cols = ", ".join(attributes_to_retrieve)
        else:
            cols = "*"
        safe_filter = _validate_filter_expr(filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        sql = f"SELECT {cols} FROM {self._table} {where} LIMIT %s"
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (limit,))  # type: ignore[arg-type]
                col_names = [desc[0] for desc in cur.description or []]
                rows = cur.fetchall()
                return [dict(zip(col_names, row)) for row in rows]
        except Exception:
            return []

    def _rows_to_hits(self, cols: list[str], rows: list[Any]) -> list[dict]:
        return _distance_rows_to_hits(cols, rows, self._vector_col)


class QdrantBackend(SearchBackend):
    """Qdrant backend. Requires `pip install qdrant-client`."""

    def __init__(
        self,
        collection: str,
        embed_fn: Any = None,
        *,
        url: str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = False,
    ):
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
            prefer_grpc=prefer_grpc,
        )
        self._collection = collection
        self._embed_fn = embed_fn
        self.index = collection

    def search(self, request: SearchRequest) -> list[dict]:
        from qdrant_client.models import Filter

        vector = self._resolve_vector(request)
        if not vector:
            return self._scroll(request)

        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "query": vector,
            "limit": request.limit,
            "with_payload": True,
        }
        if request.filter_expr:
            if isinstance(request.filter_expr, Filter):
                kwargs["query_filter"] = request.filter_expr
            elif isinstance(request.filter_expr, dict):
                kwargs["query_filter"] = Filter(**request.filter_expr)  # type: ignore[arg-type]

        try:
            results = self._client.query_points(**kwargs).points
            return self._scored_points_to_hits(results)
        except Exception:
            return []

    def _scroll(self, request: SearchRequest) -> list[dict]:
        from qdrant_client.models import Filter

        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "limit": request.limit,
            "with_payload": True,
        }
        if request.filter_expr:
            if isinstance(request.filter_expr, Filter):
                kwargs["scroll_filter"] = request.filter_expr
            elif isinstance(request.filter_expr, dict):
                kwargs["scroll_filter"] = Filter(**request.filter_expr)  # type: ignore[arg-type]
        try:
            points, _ = self._client.scroll(**kwargs)
            return [{"id": str(p.id), **(p.payload or {})} for p in points]
        except Exception:
            return []

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        from qdrant_client.models import QueryRequest

        batch = []
        for req in requests:
            vector = self._resolve_vector(req)
            if not vector:
                continue
            batch.append(
                QueryRequest(query=vector, limit=req.limit, with_payload=True),
            )
        if not batch:
            return [self.search(r) for r in requests]
        try:
            results = self._client.query_batch_points(
                collection_name=self._collection,
                requests=batch,
            )
            return [self._scored_points_to_hits(r.points) for r in results]
        except Exception:
            return [self.search(r) for r in requests]

    def get_index_config(self) -> IndexConfig:
        try:
            info = self._client.get_collection(self._collection)
            payload_schema = info.payload_schema or {}
            return IndexConfig(
                filterable_attributes=list(payload_schema.keys()),
                searchable_attributes=list(payload_schema.keys()),
                embedders=["default"],
            )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
            )
            hits = [{"id": str(p.id), **(p.payload or {})} for p in points]
            if attributes_to_retrieve:
                hits = [
                    {k: h[k] for k in attributes_to_retrieve if k in h} for h in hits
                ]
            return hits
        except Exception:
            return []

    @staticmethod
    def _scored_points_to_hits(points: list[Any]) -> list[dict]:
        hits: list[dict] = []
        for p in points:
            hit: dict[str, Any] = {"id": str(p.id), **(p.payload or {})}
            if p.score is not None:
                hit["_rankingScore"] = float(p.score)
            hits.append(hit)
        return hits


class DuckDBBackend(SearchBackend):
    def build_filter_expr(self, intent: Any) -> str:
        return _build_sql_filter(intent)

    """DuckDB + VSS extension backend. Requires `pip install duckdb`."""

    def __init__(
        self,
        table: str,
        db_path: str = ":memory:",
        embed_fn: Any = None,
        vector_column: str = "embedding",
        content_column: str = "content",
    ):
        import duckdb

        self._conn = duckdb.connect(db_path)
        try:
            self._conn.execute("INSTALL vss; LOAD vss;")
        except Exception:
            pass  # VSS may already be loaded or unavailable
        try:
            self._conn.execute("INSTALL fts; LOAD fts;")
        except Exception:
            pass
        self._table = table
        self._embed_fn = embed_fn
        self._vector_col = vector_column
        self._content_col = content_column
        self.index = table

    def search(self, request: SearchRequest) -> list[dict]:
        vector = self._resolve_vector(request)
        if vector:
            return self._vector_search(vector, request)
        return self._text_search(request)

    def _vector_search(self, vector: list[float], request: SearchRequest) -> list[dict]:
        safe_filter = _validate_filter_expr(request.filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        sql = f"""
            SELECT *, array_distance({self._vector_col}, ?::FLOAT[{len(vector)}]) AS _distance
            FROM {self._table}
            {where}
            ORDER BY _distance
            LIMIT ?
        """
        try:
            result = self._conn.execute(sql, [vector, request.limit])
            cols = [desc[0] for desc in result.description or []]
            rows = result.fetchall()
            return self._rows_to_hits(cols, rows)
        except Exception:
            return []

    def _text_search(self, request: SearchRequest) -> list[dict]:
        safe_filter = _validate_filter_expr(request.filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        if request.query:
            fts_where = f"{'AND' if where else 'WHERE'} fts_main_{self._table}.match_bm25(id, ?) IS NOT NULL"
        else:
            fts_where = ""
        sql = f"SELECT * FROM {self._table} {where} {fts_where} LIMIT ?"
        params: list[Any] = []
        if request.query:
            params.append(request.query)
        params.append(request.limit)
        try:
            result = self._conn.execute(sql, params)
            cols = [desc[0] for desc in result.description or []]
            rows = result.fetchall()
            return [dict(zip(cols, row)) for row in rows]
        except Exception:
            return []

    def get_index_config(self) -> IndexConfig:
        try:
            result = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                [self._table],
            )
            cols = [row[0] for row in result.fetchall()]
            return IndexConfig(
                filterable_attributes=[c for c in cols if c != self._vector_col],
                searchable_attributes=[self._content_col]
                if self._content_col in cols
                else cols,
                embedders=[self._vector_col] if self._vector_col in cols else [],
            )
        except Exception:
            return IndexConfig()

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        if attributes_to_retrieve:
            _validate_identifiers(attributes_to_retrieve)
            cols = ", ".join(attributes_to_retrieve)
        else:
            cols = "*"
        safe_filter = _validate_filter_expr(filter_expr)
        where = f"WHERE {safe_filter}" if safe_filter else ""
        sql = f"SELECT {cols} FROM {self._table} {where} LIMIT ?"
        try:
            result = self._conn.execute(sql, [limit])
            col_names = [desc[0] for desc in result.description or []]
            rows = result.fetchall()
            return [dict(zip(col_names, row)) for row in rows]
        except Exception:
            return []

    def _rows_to_hits(self, cols: list[str], rows: list[Any]) -> list[dict]:
        return _distance_rows_to_hits(cols, rows, self._vector_col)


class InMemoryBackend(SearchBackend):
    """In-memory vector store using langchain-core's InMemoryVectorStore.

    Zero external dependencies — works out of the box for quick prototyping.
    """

    def __init__(
        self,
        embed_fn: Any = None,
        documents: list[dict] | None = None,
    ):
        self._embed_fn = embed_fn
        self._documents: list[dict] = documents or []
        self._doc_text_lower: list[str] = [
            " ".join(str(v) for v in d.values()).lower() for d in self._documents
        ]
        self._store: Any = None
        self.index = "in_memory"

        if embed_fn:
            self._build_store()
            if documents:
                self.add_documents(documents)

    def _build_store(self) -> None:
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        embed_fn = self._embed_fn

        class _WrapEmbed(Embeddings):
            def __init__(self) -> None:
                pass

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [embed_fn(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return embed_fn(text)

        self._store = InMemoryVectorStore(embedding=_WrapEmbed())

    def add_documents(self, documents: list[dict], text_key: str = "content") -> None:
        """Add documents to the in-memory store."""
        from langchain_core.documents import Document

        self._documents.extend(documents)
        self._doc_text_lower.extend(
            " ".join(str(v) for v in d.values()).lower() for d in documents
        )
        if self._store is None and self._embed_fn:
            self._build_store()
        if self._store:
            docs = [
                Document(page_content=d.get(text_key, ""), metadata=d)
                for d in documents
            ]
            self._store.add_documents(docs)

    def search(self, request: SearchRequest) -> list[dict]:
        if self._store and (request.vector or self._embed_fn):
            try:
                results = self._store.similarity_search(
                    request.query,
                    k=request.limit,
                )
                return [
                    {
                        "content": r.page_content,
                        **r.metadata,
                        "_rankingScore": 1.0 / (i + 1),
                    }
                    for i, r in enumerate(results)
                ]
            except Exception:
                pass

        # Fallback: naive substring match over pre-lowercased doc text.
        query_lower = request.query.lower()
        hits: list[dict] = []
        for doc, text in zip(self._documents, self._doc_text_lower):
            if query_lower in text:
                hits.append(doc)
                if len(hits) >= request.limit:
                    break
        return hits

    def get_index_config(self) -> IndexConfig:
        if not self._documents:
            return IndexConfig()
        sample = self._documents[0]
        keys = [k for k in sample if k != "embedding"]
        return IndexConfig(
            filterable_attributes=keys,
            searchable_attributes=keys,
            embedders=["default"] if self._embed_fn else [],
        )

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        docs = self._documents[:limit]
        if attributes_to_retrieve:
            docs = [{k: d[k] for k in attributes_to_retrieve if k in d} for d in docs]
        return docs


class _MultiBackend(SearchBackend):
    """Wraps N backends and fans out searches to the active subset.

    Active subset is taken from the `_ACTIVE_COLLECTIONS` ContextVar; an empty
    value or None means all backends. Each returned hit is tagged with the
    originating collection name via ``_collection`` so callers can merge,
    dedupe, or filter downstream.
    """

    def __init__(self, backends: dict[str, SearchBackend]):
        if not backends:
            raise ValueError("_MultiBackend requires at least one backend")
        self._backends = backends
        self.names: list[str] = list(backends.keys())
        self.index = ",".join(self.names)

    def _active_names(self) -> list[str]:
        active = _ACTIVE_COLLECTIONS.get()
        if not active:
            return self.names
        filtered = [n for n in active if n in self._backends]
        return filtered or self.names

    @staticmethod
    def _tag(hits: list[dict], name: str) -> list[dict]:
        return [{**h, "_collection": name} for h in hits]

    def search(self, request: SearchRequest) -> list[dict]:
        merged: list[dict] = []
        for name in self._active_names():
            merged.extend(self._tag(self._backends[name].search(request), name))
        merged.sort(key=lambda h: h.get("_rankingScore", 0.0), reverse=True)
        return merged[: request.limit]

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        from dataclasses import replace

        def _rebind(reqs: list[SearchRequest], name: str) -> list[SearchRequest]:
            out = []
            for r in reqs:
                if r.index_uid and r.index_uid != name:
                    out.append(replace(r, index_uid=None))
                else:
                    out.append(r)
            return out

        per_backend: list[tuple[str, list[list[dict]]]] = [
            (n, self._backends[n].batch_search(_rebind(requests, n)))
            for n in self._active_names()
        ]
        outputs: list[list[dict]] = []
        for i, req in enumerate(requests):
            merged: list[dict] = []
            for name, res in per_backend:
                if i < len(res):
                    merged.extend(self._tag(res[i], name))
            merged.sort(key=lambda h: h.get("_rankingScore", 0.0), reverse=True)
            outputs.append(merged[: req.limit])
        return outputs

    def get_index_config(self) -> IndexConfig:
        def _union(attr: str) -> list[str]:
            seen: list[str] = []
            for b in self._backends.values():
                for v in getattr(b.get_index_config(), attr):
                    if v not in seen:
                        seen.append(v)
            return seen

        return IndexConfig(
            filterable_attributes=_union("filterable_attributes"),
            searchable_attributes=_union("searchable_attributes"),
            sortable_attributes=_union("sortable_attributes"),
            ranking_rules=_union("ranking_rules"),
            embedders=_union("embedders"),
        )

    def sample_documents(
        self,
        limit: int = 20,
        filter_expr: str | None = None,
        attributes_to_retrieve: list[str] | None = None,
    ) -> list[dict]:
        names = self._active_names()
        per = max(1, limit // max(len(names), 1))
        samples: list[dict] = []
        for name in names:
            samples.extend(
                self._backends[name].sample_documents(
                    per, filter_expr, attributes_to_retrieve
                )
            )
        return samples[:limit]

    def add_documents(self, documents: list[dict], *args, **kwargs) -> None:
        """Route documents to their collection by `_collection` field.

        Documents without `_collection` are rejected: with N backends the
        target is ambiguous.
        """
        buckets: dict[str, list[dict]] = {n: [] for n in self._backends}
        missing: list[dict] = []
        for doc in documents:
            name = doc.get("_collection")
            if name in buckets:
                buckets[name].append(
                    {k: v for k, v in doc.items() if k != "_collection"}
                )
            else:
                missing.append(doc)
        if missing:
            raise ValueError(
                f"{len(missing)} document(s) missing/invalid `_collection` field. "
                f"Valid collections: {list(self._backends)}"
            )
        for name, docs in buckets.items():
            if not docs:
                continue
            backend = self._backends[name]
            add = getattr(backend, "add_documents", None)
            if add is not None:
                add(docs, *args, **kwargs)
