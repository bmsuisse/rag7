"""Create Azure AI Search index and populate with test data from local parquet files."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
INDEX_NAME = "benchmark-docs"


def create_index() -> None:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchableField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
    )

    client = SearchIndexClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(API_KEY),
    )

    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="url", type=SearchFieldDataType.String),
    ]

    index = SearchIndex(name=INDEX_NAME, fields=fields)

    # Delete if exists, then create
    try:
        client.delete_index(INDEX_NAME)
        print(f"Deleted existing index '{INDEX_NAME}'")
    except Exception:
        pass

    client.create_index(index)
    print(f"Created index '{INDEX_NAME}'")


def upload_docs() -> None:
    import pandas as pd
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        endpoint=ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(API_KEY),
    )

    # Load subset from each source — 50 docs per file to stay within free tier limits
    from pathlib import Path

    data_dir = Path("data")
    total = 0

    for parquet_file in sorted(data_dir.glob("**/*.parquet")):
        df = pd.read_parquet(parquet_file).head(50)
        docs = df.to_dict("records")

        ALLOWED_FIELDS = {"id", "title", "content", "language", "source", "url"}
        clean_docs = []
        for doc in docs:
            clean = {
                "id": str(doc["id"]),
                "content": str(doc.get("content", ""))[:32000],
                "title": str(doc.get("title", "")),
                "language": str(doc.get("language", "")),
                "source": str(doc.get("source", "")),
                "url": str(doc.get("url", "")),
            }
            clean_docs.append(clean)
        docs = clean_docs

        if docs:
            result = client.upload_documents(docs)
            succeeded = sum(1 for r in result if r.succeeded)
            print(f"  {parquet_file}: {succeeded}/{len(docs)} uploaded")
            total += succeeded

    print(f"\nTotal: {total} docs uploaded to '{INDEX_NAME}'")


if __name__ == "__main__":
    print("Creating Azure AI Search index...")
    create_index()
    print("\nUploading documents...")
    upload_docs()
    print("\nDone. Add AZURE_SEARCH_INDEX=benchmark-docs to .env")
