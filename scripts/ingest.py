"""Load the USDA pages, split them with LangChain, and store vectors in Chroma.

Same shape as Week 5 implementation/ingest.py. Embeddings stay on this
machine with all-MiniLM-L6-v2. OpenAI embeddings are not used.

The knowledge folder is flat. Each Markdown file becomes a LangChain
Document. Three collections are written:

- character-500: RecursiveCharacterTextSplitter, 500 characters, 200 overlap
- character-1000: the same splitter at the Day 2 size, 1000 / 200
- section: MarkdownHeaderTextSplitter, then the 500-character splitter
  on any section that is still long

    uv add sentence-transformers
    uv run python scripts/ingest.py
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = str(ROOT / "knowledge")
DB_NAME = str(ROOT / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def fetch_documents() -> list[Document]:
    documents = []
    for path in sorted(Path(KNOWLEDGE_BASE).glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path), "source_file": path.name},
            )
        )
    return documents


def character_chunks(documents: list[Document], size: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=200)
    return splitter.split_documents(documents)


def section_chunks(documents: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    pieces: list[Document] = []
    for doc in documents:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata.update(doc.metadata)
        pieces.extend(size_splitter.split_documents(sections))
    return pieces


def create_embeddings(chunks: list[Document], collection_name: str) -> Chroma:
    client = chromadb.PersistentClient(path=DB_NAME)
    existing = [collection.name for collection in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
        collection_name=collection_name,
    )
    count = vectorstore._collection.count()
    sample = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    print(f"{collection_name}: {count:,} vectors, {len(sample):,} dimensions")
    return vectorstore


def main() -> None:
    documents = fetch_documents()
    print(f"Loaded {len(documents)} documents")
    pipelines = {
        "character-500": character_chunks(documents, 500),
        "character-1000": character_chunks(documents, 1000),
        "section": section_chunks(documents),
    }
    for name, chunks in pipelines.items():
        create_embeddings(chunks, name)
    print("Ingestion complete")


if __name__ == "__main__":
    main()
