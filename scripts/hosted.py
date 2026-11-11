"""Ask path for the public server.

A question is searched with hybrid retrieval over the 109 LLM chunks.
Keyword rank and OpenAI text-embedding-3-small vector rank are fused,
and the top 5 chunks are given to Groq to write the reply. The citation
check keeps or replaces it.

No local embedding model is loaded. The comparison table is still the saved
MiniLM evaluation and is not rescored here.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "eval" / "llm_chunks.json"
VECTORS_PATH = ROOT / "eval" / "openai_vectors.json"
EMBED_MODEL = "text-embedding-3-small"
MODEL = "groq/openai/gpt-oss-120b"
TOP_K = 5
FUSION_CANDIDATES = 20
RRF_K = 60
REFUSAL = "I don't know based on these documents."

load_dotenv(ROOT / ".env")
litellm.suppress_debug_info = True


def require_key() -> None:
    missing = [name for name in ("OPENAI_API_KEY", "GROQ_API_KEY") if not os.getenv(name)]
    if not missing:
        return
    raise SystemExit(
        "Add these to .env locally, and as secret environment variables on Render:\n\n"
        + "\n".join(f"    {name}=..." for name in missing)
        + "\n\nOpenAI embeds the question. Groq writes the answer.\n"
        "Create the OpenAI key at https://platform.openai.com/api-keys\n"
        "Create the Groq key at https://console.groq.com"
    )


def page_content(chunk: dict) -> str:
    return f"{chunk['headline']}\n\n{chunk['summary']}\n\n{chunk['original_text']}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fuse(rankings: tuple[list[dict], ...], k: int) -> list[dict]:
    """Reciprocal rank fusion. Rank 1 scores 1/61, rank 2 scores 1/62, and so on."""
    scores: dict[str, float] = {}
    chosen: dict[str, dict] = {}
    for hits in rankings:
        for rank, chunk in enumerate(hits, start=1):
            key = page_content(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chosen.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chosen[key] for key in ordered[:k]]


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = litellm.embedding(model=EMBED_MODEL, input=texts)
    data = response.data if hasattr(response, "data") else response["data"]
    rows = []
    for item in data:
        if isinstance(item, dict):
            index, embedding = item["index"], item["embedding"]
        else:
            index, embedding = item.index, item.embedding
        rows.append((index, [round(float(value), 6) for value in embedding]))
    rows.sort()
    return [vector for _, vector in rows]


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), 16):
        vectors.extend(embed_batch(texts[start : start + 16]))
    return vectors


def build_vectors() -> None:
    require_key()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]
    vectors = embed_texts([page_content(chunk) for chunk in chunks])
    payload = {
        "model": EMBED_MODEL,
        "dims": len(vectors[0]),
        "count": len(vectors),
        "vectors": vectors,
    }
    VECTORS_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {len(vectors)} vectors to {VECTORS_PATH.name}", flush=True)


def message_text(response) -> str:
    message = response.choices[0].message
    content = message.content or getattr(message, "reasoning_content", None) or ""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


class Index:
    def __init__(self) -> None:
        if not VECTORS_PATH.is_file():
            raise SystemExit(
                "The OpenAI chunk vectors are missing. From the project folder run:\n\n"
                "    uv run python scripts/hosted.py --embed"
            )
        payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        saved = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
        self.chunks = payload["chunks"]
        self.vectors = saved["vectors"]
        if len(self.vectors) != len(self.chunks) or saved.get("model") != EMBED_MODEL:
            raise SystemExit("eval/openai_vectors.json does not match the saved chunks. Run --embed again.")
        self.bm25 = BM25Okapi([tokenize(page_content(chunk)) for chunk in self.chunks])

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        keyword_scores = self.bm25.get_scores(tokenize(question))
        keyword_order = sorted(range(len(self.chunks)), key=lambda i: float(keyword_scores[i]), reverse=True)
        keyword_hits = [self.chunks[i] for i in keyword_order[:FUSION_CANDIDATES]]
        query = embed_texts([question])[0]
        vector_order = sorted(range(len(self.chunks)), key=lambda i: cosine(query, self.vectors[i]), reverse=True)
        vector_hits = [self.chunks[i] for i in vector_order[:FUSION_CANDIDATES]]
        return fuse((keyword_hits, vector_hits), k)


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
        time.sleep(50)
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


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def is_refusal(reply: str) -> bool:
    text = normalize(reply).rstrip(".").lower()
    target = normalize(REFUSAL).rstrip(".").lower()
    return text == target


def cited_sentence(reply: str, chunk_texts: list[str]) -> str | None:
    reply_n = normalize(reply)
    for chunk in chunk_texts:
        for sentence in re.split(r"(?<=[.!?])\s+", normalize(chunk)):
            if len(sentence) >= 40 and sentence in reply_n:
                return sentence
    return None


def enforce(reply: str, chunk_texts: list[str]) -> tuple[str, str | None, str]:
    raw = reply.strip()
    if is_refusal(raw):
        return REFUSAL, None, "refused"
    sentence = cited_sentence(raw, chunk_texts)
    if sentence:
        return raw, sentence, "cited"
    return REFUSAL, None, "rejected"


@retry(wait=wait_exponential(multiplier=1, min=45, max=120), stop=stop_after_attempt(6), reraise=True)
def generate(question: str, records: list[dict], pause: bool = True) -> str:
    context = "\n\n".join(
        f"Chunk {record['rank']} from {record['source_file']}:\n{record['text']}"
        for record in records
    )
    response = litellm.completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about USDA food safety using only the retrieved chunks. "
                    "Copy one exact sentence from a chunk into your reply. "
                    f"If the chunks do not contain the answer, reply with exactly: {REFUSAL}"
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved chunks:\n{context}",
            },
        ],
        max_tokens=500,
        extra_body={"reasoning_effort": "low"},
    )
    if pause:
        time.sleep(50)
    return message_text(response)


def answer_question(index: Index, question: str) -> dict:
    records = ranked_records(index.search(question))
    raw = generate(question, records, pause=False)
    final, quote, _status = enforce(raw, [record["text"] for record in records])
    return {
        "answer": final,
        "quote": quote,
        "sources": [{"rank": record["rank"], "source": record["source_file"]} for record in records],
    }


if __name__ == "__main__":
    if "--embed" in sys.argv:
        build_vectors()
    else:
        raise SystemExit("Run: uv run python scripts/hosted.py --embed")
