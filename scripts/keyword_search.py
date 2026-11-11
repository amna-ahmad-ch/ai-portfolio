"""BM25, vector, and hybrid search over the chunks stored by scripts/ingest.py.

ingest.py writes three Chroma collections with LangChain splitters:

- character-500
- character-1000
- section

This script does not split the pages again. BM25 ranks the stored chunk
text by shared words. Vector search calls Chroma, which uses the
all-MiniLM-L6-v2 embeddings saved at ingest time. Hybrid search asks both
for their top 20 and merges the lists with reciprocal rank fusion.
hybrid-rerank scores those 20 question-and-chunk pairs with the local
cross-encoder ms-marco-MiniLM-L-6-v2 and keeps the best 5.

A question counts as found when every keyword appears in the top 5 chunks.
MRR is Ed's retrieval score: for each keyword, the reciprocal of the rank
of the first chunk that contains it. A keyword in chunk 1 scores 1, chunk 2
scores 1/2, and a keyword missing from the top 5 scores 0. The question
score is the average of its keywords, and the row score is the average of
the questions.

    uv run python scripts/keyword_search.py
    uv run python scripts/keyword_search.py "How long can leftovers stay in the refrigerator?"
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions.jsonl"
DB_NAME = str(ROOT / "vector_db")
COLLECTIONS = ("character-500", "character-1000", "section")
TOP_K = 5
CANDIDATES = 20
RRF_K = 60

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global reranker
    if reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    url: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def source_url(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Source: "):
            return line.removeprefix("Source: ").strip()
    return ""


def load_chunks(collection_name: str) -> tuple[Chroma, list[Chunk]]:
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DB_NAME,
    )
    stored = store.get(include=["documents", "metadatas"])
    chunks = []
    for text, metadata in zip(stored["documents"], stored["metadatas"]):
        metadata = metadata or {}
        chunks.append(
            Chunk(
                text=text,
                source=metadata.get("source_file", ""),
                url=source_url(text),
            )
        )
    return store, chunks


def bm25_search(chunks: list[Chunk], question: str, k: int = TOP_K) -> list[Chunk]:
    index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
    scores = index.get_scores(tokenize(question))
    ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [chunks[i] for i in ranked[:k]]


def vector_search(store: Chroma, question: str, k: int = TOP_K) -> list[Chunk]:
    docs = store.similarity_search(question, k=k)
    return [
        Chunk(
            text=doc.page_content,
            source=doc.metadata.get("source_file", ""),
            url=source_url(doc.page_content),
        )
        for doc in docs
    ]


def hybrid_search(
    chunks: list[Chunk],
    store: Chroma,
    question: str,
    k: int = TOP_K,
) -> list[Chunk]:
    """Merge BM25 and vector ranks. Rank 1 scores 1/61, rank 2 scores 1/62, and so on."""
    rankings = (
        bm25_search(chunks, question, CANDIDATES),
        vector_search(store, question, CANDIDATES),
    )
    scores: dict[Chunk, float] = {}
    for hits in rankings:
        for rank, chunk in enumerate(hits, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda chunk: scores[chunk], reverse=True)
    return ordered[:k]


def rerank(question: str, chunks: list[Chunk], k: int = TOP_K) -> list[Chunk]:
    """Score each question and chunk together. The highest score is the most relevant chunk."""
    if not chunks:
        return []
    scores = get_reranker().predict(
        [(question, chunk.text) for chunk in chunks],
        show_progress_bar=False,
    )
    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [chunks[i] for i in order[:k]]


def hybrid_rerank(
    chunks: list[Chunk],
    store: Chroma,
    question: str,
    k: int = TOP_K,
) -> list[Chunk]:
    pool = hybrid_search(chunks, store, question, CANDIDATES)
    return rerank(question, pool, k)


def reciprocal_rank(keyword: str, chunks: list[Chunk]) -> float:
    """Ed's calculate_mrr: 1/rank of the first chunk that contains this keyword."""
    needle = keyword.lower()
    for rank, chunk in enumerate(chunks, start=1):
        if needle in chunk.text.lower():
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(chunks: list[Chunk], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    return sum(reciprocal_rank(keyword, chunks) for keyword in keywords) / len(keywords)


def keywords_found(chunks: list[Chunk], keywords: list[str]) -> bool:
    haystack = "\n".join(chunk.text for chunk in chunks).lower()
    return all(keyword.lower() in haystack for keyword in keywords)


def load_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    return questions


def print_hits(question: str, chunks: list[Chunk]) -> None:
    print(question)
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ")[:160]
        print(f"  {rank}. {chunk.source}")
        print(f"     {chunk.url}")
        print(f"     {preview}")


def score(questions: list[dict], search_one) -> tuple[list[str], list[str], float]:
    hits = []
    misses = []
    total_mrr = 0.0
    for item in questions:
        found = search_one(item["question"])
        (hits if keywords_found(found, item["keywords"]) else misses).append(item["question"])
        total_mrr += mean_reciprocal_rank(found, item["keywords"])
    average_mrr = total_mrr / len(questions) if questions else 0.0
    return hits, misses, average_mrr


def evaluate() -> None:
    questions = [item for item in load_questions() if item["keywords"]]
    print(f"In-scope questions: {len(questions)}")
    for name in COLLECTIONS:
        store, chunks = load_chunks(name)
        print(f"\n{name}: {len(chunks)} chunks")
        for label, search_one in (
            ("bm25", lambda question, chunks=chunks: bm25_search(chunks, question)),
            ("vector", lambda question, store=store: vector_search(store, question)),
            ("hybrid", lambda question, chunks=chunks, store=store: hybrid_search(chunks, store, question)),
            ("hybrid-rerank", lambda question, chunks=chunks, store=store: hybrid_rerank(chunks, store, question)),
        ):
            hits, misses, average_mrr = score(questions, search_one)
            print(f"  {label}: {len(hits)}/{len(questions)}  MRR {average_mrr:.3f}")
            for question in misses:
                print(f"    miss: {question}")


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        for name in COLLECTIONS:
            store, chunks = load_chunks(name)
            print(f"\n## {name} / bm25")
            print_hits(question, bm25_search(chunks, question))
            print(f"\n## {name} / vector")
            print_hits(question, vector_search(store, question))
            print(f"\n## {name} / hybrid")
            print_hits(question, hybrid_search(chunks, store, question))
            print(f"\n## {name} / hybrid-rerank")
            print_hits(question, hybrid_rerank(chunks, store, question))
        return
    evaluate()


if __name__ == "__main__":
    main()
