"""Ed's Day 5 chunks: a chat model picks the boundaries, then reranks search results.

Same shape as week5/pro_implementation/ingest.py and answer.py. The model
returns a headline, a summary, and the original text for each chunk. Those
chunks are embedded locally with all-MiniLM-L6-v2 and stored in the Chroma
collection llm-chunks. OpenAI embeddings are not used.

Ed's file asks for one chunk per 100 characters. These USDA pages are long
enough that one reply would be cut off, so each page is sent in pieces of
at most 6,000 characters and the model aims for about one chunk per 1,000
characters inside each piece.

The chunks are saved in eval/llm_chunks.json. The scores are saved in
eval/llm_results.json. A later run reads those files and does not call the
model again.

Create a file named .env in the project folder with a free Groq key:

    GROQ_API_KEY=your-key

Then:

    uv run python scripts/llm_chunks.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
import litellm
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
DB_NAME = str(ROOT / "vector_db")
CHUNKS_PATH = ROOT / "eval" / "llm_chunks.json"
RESULTS_PATH = ROOT / "eval" / "llm_results.json"
COLLECTION = "llm-chunks"
MODEL = "groq/openai/gpt-oss-120b"
PIECE_SIZE = 6000
AVERAGE_CHUNK_SIZE = 1000
TOP_K = 5
CANDIDATES = 8
FUSION_CANDIDATES = 20
RRF_K = 60
PAUSE_SECONDS = 50

load_dotenv(ROOT / ".env")
litellm.suppress_debug_info = True

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

DOC_TYPES = {
    "refrigeration.md": "storage",
    "freezing-and-food-safety.md": "storage",
    "leftovers-and-food-safety.md": "storage",
    "big-thaw.md": "storage",
    "danger-zone.md": "temperature",
    "how-temperatures-affect-food.md": "temperature",
    "safe-temperature-chart.md": "temperature",
    "food-thermometers.md": "temperature",
    "grilling-and-food-safety.md": "cooking",
    "cooking-with-microwave-ovens.md": "cooking",
    "steps-to-keep-food-safe.md": "handling",
    "cutting-boards.md": "handling",
}


class Chunk(BaseModel):
    headline: str = Field(description="A few words that would match a question about this chunk")
    summary: str = Field(description="A few sentences that answer the questions this chunk is good for")
    original_text: str = Field(description="The original text of this chunk, copied exactly and not rewritten")


class Chunks(BaseModel):
    chunks: list[Chunk]


def require_key() -> None:
    if os.getenv("GROQ_API_KEY"):
        return
    raise SystemExit(
        "Add a free Groq key to .env in the project folder:\n\n"
        "    GROQ_API_KEY=your-key\n\n"
        "Create the key at https://console.groq.com. No credit card is required.\n"
        "Then run: uv run python scripts/llm_chunks.py"
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def source_url(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Source: "):
            return line.removeprefix("Source: ").strip()
    return ""


def fetch_documents() -> list[dict]:
    documents = []
    for path in sorted(KNOWLEDGE.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            {
                "source_file": path.name,
                "source": str(path),
                "doc_type": DOC_TYPES.get(path.name, "general"),
                "url": source_url(text),
                "text": text,
            }
        )
    return documents


def pieces_of(text: str) -> list[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    parts = [part.page_content.strip() for part in splitter.split_text(text)]
    parts = [part for part in parts if part]
    if not parts:
        parts = [text.strip()]

    pieces: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > PIECE_SIZE:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            start = 0
            while start < len(part):
                pieces.append(part[start : start + PIECE_SIZE])
                if start + PIECE_SIZE >= len(part):
                    break
                start += PIECE_SIZE - 500
            continue
        if buffer and len(buffer) + 2 + len(part) > PIECE_SIZE:
            pieces.append(buffer)
            buffer = part
        else:
            buffer = f"{buffer}\n\n{part}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def make_prompt(document: dict, piece: str) -> str:
    how_many = max(1, (len(piece) // AVERAGE_CHUNK_SIZE) + 1)
    return f"""
You take a document and you split it into overlapping chunks for a knowledge base.

The document is a public USDA food-safety page.
The document type is: {document["doc_type"]}
The file is: {document["source_file"]}

A chatbot will use these chunks to answer questions about food safety.
Divide this text so the whole piece is covered. Do not leave anything out.
Use about {how_many} chunks. Overlap them by about 25 percent, or about 50 words,
so the same sentences can appear in more than one chunk.

For each chunk provide a headline, a summary, and the original text.
The original text must be copied exactly from the document, not rewritten.

Here is the document piece:

{piece}
"""


def message_text(response) -> str:
    message = response.choices[0].message
    content = message.content or getattr(message, "reasoning_content", None) or ""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


@retry(wait=wait_exponential(multiplier=1, min=45, max=120), stop=stop_after_attempt(6), reraise=True)
def chunk_piece(document: dict, piece: str) -> list[dict]:
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": make_prompt(document, piece)}],
        response_format=Chunks,
        max_tokens=4096,
        extra_body={"reasoning_effort": "low"},
    )
    time.sleep(PAUSE_SECONDS)
    parsed = Chunks.model_validate_json(message_text(response))
    stored = []
    for chunk in parsed.chunks:
        if not chunk.original_text.strip():
            continue
        stored.append(
            {
                "headline": chunk.headline,
                "summary": chunk.summary,
                "original_text": chunk.original_text,
                "source_file": document["source_file"],
                "source": document["source"],
                "doc_type": document["doc_type"],
                "url": document["url"],
            }
        )
    if not stored:
        raise ValueError(f"no chunks returned for {document['source_file']}")
    return stored


def load_saved() -> dict:
    if not CHUNKS_PATH.exists():
        return {"files_done": [], "chunks": []}
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def save_chunks(payload: dict) -> None:
    CHUNKS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_chunks(documents: list[dict]) -> list[dict]:
    payload = load_saved() if "--refresh" not in sys.argv else {"files_done": [], "chunks": []}
    done = set(payload["files_done"])
    for document in documents:
        name = document["source_file"]
        if name in done:
            print(f"skip {name}")
            continue
        require_key()
        pieces = pieces_of(document["text"])
        print(f"{name}: {len(pieces)} pieces", flush=True)
        for index, piece in enumerate(pieces, start=1):
            print(f"  piece {index}/{len(pieces)}", flush=True)
            payload["chunks"].extend(chunk_piece(document, piece))
            save_chunks(payload)
        payload["files_done"].append(name)
        save_chunks(payload)
    print(f"LLM chunks: {len(payload['chunks'])}")
    return payload["chunks"]


def page_content(chunk: dict) -> str:
    return f"{chunk['headline']}\n\n{chunk['summary']}\n\n{chunk['original_text']}"


def embed(chunks: list[dict]) -> Chroma:
    documents = [
        Document(
            page_content=page_content(chunk),
            metadata={
                "source": chunk["source"],
                "source_file": chunk["source_file"],
                "doc_type": chunk["doc_type"],
            },
        )
        for chunk in chunks
    ]
    client = chromadb.PersistentClient(path=DB_NAME)
    existing = [collection.name for collection in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_NAME,
        collection_name=COLLECTION,
    )
    print(f"{COLLECTION}: {store._collection.count():,} vectors")
    return store


def load_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["keywords"]:
                questions.append(item)
    return questions


def bm25_search(chunks: list[dict], question: str, k: int) -> list[dict]:
    index = BM25Okapi([tokenize(page_content(chunk)) for chunk in chunks])
    scores = index.get_scores(tokenize(question))
    ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    return [chunks[i] for i in ranked[:k]]


def vector_search(store: Chroma, question: str, k: int) -> list[dict]:
    found = []
    for doc in store.similarity_search(question, k=k):
        found.append(
            {
                "headline": "",
                "summary": "",
                "original_text": doc.page_content,
                "text": doc.page_content,
                "source_file": doc.metadata.get("source_file", ""),
                "source": doc.metadata.get("source", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "url": source_url(doc.page_content),
            }
        )
    return found


def fusion_key(chunk: dict) -> str:
    return chunk.get("text") or page_content(chunk)


def hybrid_search(chunks: list[dict], store: Chroma, question: str, k: int = TOP_K) -> list[dict]:
    """Merge the top 20 BM25 and vector ranks, same reciprocal rank fusion as keyword_search.py."""
    rankings = (
        bm25_search(chunks, question, FUSION_CANDIDATES),
        vector_search(store, question, FUSION_CANDIDATES),
    )
    scores: dict[str, float] = {}
    chosen: dict[str, dict] = {}
    for hits in rankings:
        for rank, chunk in enumerate(hits, start=1):
            key = fusion_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chosen.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chosen[key] for key in ordered[:k]]


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids from most relevant to least relevant. Include every id you were given."
    )


@retry(wait=wait_exponential(multiplier=1, min=45, max=120), stop=stop_after_attempt(6), reraise=True)
def llm_rerank(question: str, chunks: list[dict], pause: bool = True) -> list[dict]:
    system_prompt = """
You are a document re-ranker.
You are given a question and chunks from a USDA food-safety knowledge base.
The chunks are in retrieval order. Rank them by relevance to the question, most relevant first.
Reply only with the list of ranked chunk ids. Include every chunk id you were given.
"""
    user_prompt = f"Question:\n\n{question}\n\n"
    for index, chunk in enumerate(chunks, start=1):
        user_prompt += f"# CHUNK ID: {index}:\n\n{page_content(chunk)}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids."
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RankOrder,
        max_tokens=400,
        extra_body={"reasoning_effort": "low"},
    )
    if pause:
        time.sleep(PAUSE_SECONDS)
    order = RankOrder.model_validate_json(message_text(response)).order
    ranked = []
    seen = set()
    for chunk_id in order:
        position = chunk_id - 1
        if position in seen or not 0 <= position < len(chunks):
            continue
        seen.add(position)
        ranked.append(chunks[position])
    for index, chunk in enumerate(chunks):
        if index not in seen:
            ranked.append(chunk)
    return ranked[:TOP_K]


def ranked_records(found: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "source_file": chunk.get("source_file", ""),
            "text": page_content(chunk),
        }
        for rank, chunk in enumerate(found, start=1)
    ]


def mrr_from_records(records: list[dict], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    total = 0.0
    for keyword in keywords:
        needle = keyword.lower()
        score = 0.0
        for record in records:
            if needle in record["text"].lower():
                score = 1.0 / record["rank"]
                break
        total += score
    return total / len(keywords)


def keywords_found(chunks: list[dict], keywords: list[str]) -> bool:
    haystack = "\n".join(page_content(chunk) for chunk in chunks).lower()
    return all(keyword.lower() in haystack for keyword in keywords)


def score_label(rows: dict, orders: list[dict], label: str, item: dict, found: list[dict]) -> None:
    records = ranked_records(found)
    orders[-1][label] = records
    rows[label]["mrr"] = rows[label].get("mrr", 0.0) + mrr_from_records(records, item["keywords"])
    if keywords_found(found, item["keywords"]):
        rows[label]["hits"] += 1
    else:
        rows[label]["misses"].append(item["question"])


def finish_mrr(rows: dict, question_count: int) -> None:
    for row in rows.values():
        row["mrr"] = round(row.get("mrr", 0.0) / question_count, 3) if question_count else 0.0


def score_rows(chunks: list[dict], store: Chroma) -> dict:
    questions = load_questions()
    rows = {
        "bm25": {"hits": 0, "misses": [], "mrr": 0.0},
        "vector": {"hits": 0, "misses": [], "mrr": 0.0},
        "hybrid": {"hits": 0, "misses": [], "mrr": 0.0},
        "llm-rerank": {"hits": 0, "misses": [], "mrr": 0.0},
    }
    orders: list[dict] = []
    for item in questions:
        question = item["question"]
        orders.append({"question": question, "keywords": item["keywords"]})
        checks = {
            "bm25": bm25_search(chunks, question, TOP_K),
            "vector": vector_search(store, question, TOP_K),
            "hybrid": hybrid_search(chunks, store, question),
            "llm-rerank": llm_rerank(question, vector_search(store, question, CANDIDATES)),
        }
        for label, found in checks.items():
            score_label(rows, orders, label, item, found)
        print(f"scored: {question}")
    finish_mrr(rows, len(questions))
    return {
        "model": MODEL,
        "chunks": len(chunks),
        "questions": len(questions),
        **rows,
        "orders": orders,
    }


def print_results(results: dict) -> None:
    print(f"\nllm-chunks: {results['chunks']} chunks, {results['questions']} questions")
    for label in ("bm25", "vector", "hybrid", "llm-rerank"):
        if label not in results:
            continue
        row = results[label]
        mrr = row.get("mrr")
        mrr_text = f"  MRR {mrr:.3f}" if isinstance(mrr, (int, float)) else ""
        print(f"  {label}: {row['hits']}/{results['questions']}{mrr_text}")
        for question in row["misses"]:
            print(f"    miss: {question}")


def open_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_NAME,
    )


def save_local_orders() -> None:
    """Store BM25, vector, and hybrid orders. The Groq rerank score is left as saved."""
    chunks = load_saved()["chunks"]
    store = open_store()
    questions = load_questions()
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    previous_orders = {row["question"]: row for row in saved.get("orders", [])}
    rows = {
        "bm25": {"hits": 0, "misses": [], "mrr": 0.0},
        "vector": {"hits": 0, "misses": [], "mrr": 0.0},
        "hybrid": {"hits": 0, "misses": [], "mrr": 0.0},
    }
    orders: list[dict] = []
    for item in questions:
        question = item["question"]
        orders.append({"question": question, "keywords": item["keywords"]})
        checks = {
            "bm25": bm25_search(chunks, question, TOP_K),
            "vector": vector_search(store, question, TOP_K),
            "hybrid": hybrid_search(chunks, store, question),
        }
        for label, found in checks.items():
            score_label(rows, orders, label, item, found)
        reranked = previous_orders.get(question, {}).get("llm-rerank")
        if reranked:
            orders[-1]["llm-rerank"] = reranked
    finish_mrr(rows, len(questions))
    saved.update(rows)
    saved["orders"] = orders
    saved["chunks"] = len(chunks)
    saved["questions"] = len(questions)
    RESULTS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print_results(saved)


def save_rerank_orders() -> None:
    """Call Groq once per question and store the ordered chunks with the score."""
    require_key()
    chunks = load_saved()["chunks"]
    store = open_store()
    questions = load_questions()
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    orders = {row["question"]: row for row in saved.get("orders", [])}
    row = {"hits": 0, "misses": [], "mrr": 0.0}
    for item in questions:
        question = item["question"]
        entry = orders.setdefault(question, {"question": question, "keywords": item["keywords"]})
        if entry.get("llm-rerank"):
            print(f"skip {question}", flush=True)
        else:
            print(f"rerank: {question}", flush=True)
            found = llm_rerank(question, vector_search(store, question, CANDIDATES))
            entry["llm-rerank"] = ranked_records(found)
            saved["orders"] = [orders[q["question"]] for q in questions if q["question"] in orders]
            RESULTS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        records = entry["llm-rerank"]
        row["mrr"] += mrr_from_records(records, item["keywords"])
        if keywords_found(
            [{"headline": "", "summary": "", "original_text": record["text"]} for record in records],
            item["keywords"],
        ):
            row["hits"] += 1
        else:
            row["misses"].append(question)
    row["mrr"] = round(row["mrr"] / len(questions), 3) if questions else 0.0
    saved["llm-rerank"] = row
    saved["orders"] = [orders[item["question"]] for item in questions]
    RESULTS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print_results(saved)


def main() -> None:
    documents = fetch_documents()
    if "--count" in sys.argv:
        for document in documents:
            print(f"{document['source_file']}: {len(pieces_of(document['text']))} pieces")
        return
    if "--save-orders" in sys.argv:
        save_local_orders()
        return
    if "--rerank" in sys.argv:
        save_rerank_orders()
        return

    chunks = create_chunks(documents)
    store = embed(chunks)
    if RESULTS_PATH.exists() and "--rescore" not in sys.argv:
        print_results(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
        print("Saved scores loaded from eval/llm_results.json")
        return
    require_key()
    results = score_rows(chunks, store)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_results(results)


if __name__ == "__main__":
    main()
