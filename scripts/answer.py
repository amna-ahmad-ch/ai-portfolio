"""Answer with the winning retriever, then enforce a citation.

The winner is Groq rerank over the LLM chunks. The 16 in-scope questions
use the ordered chunks already saved in eval/llm_results.json. Refusal
questions are reranked once and that order is saved here.

The model must copy one exact sentence from a retrieved chunk. If that
sentence is not in the chunks, the reply is replaced with:

    I don't know based on these documents.

Every reply is written to eval/answer_results.json, so a later run does
not call Groq again.

    uv run python scripts/answer.py
"""

import json
import re
import sys
from pathlib import Path

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

import llm_chunks

ROOT = llm_chunks.ROOT
QUESTIONS = llm_chunks.QUESTIONS
RESULTS_PATH = llm_chunks.RESULTS_PATH
ANSWERS_PATH = ROOT / "eval" / "answer_results.json"
REFUSAL = "I don't know based on these documents."
MODEL = llm_chunks.MODEL


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def load_all_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    return questions


def saved_orders() -> dict[str, list[dict]]:
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        row["question"]: row.get("llm-rerank", [])
        for row in saved.get("orders", [])
        if row.get("llm-rerank")
    }


def load_answers() -> dict:
    if not ANSWERS_PATH.exists():
        return {"model": MODEL, "answers": []}
    return json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))


def save_answers(payload: dict) -> None:
    ANSWERS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
def generate(question: str, records: list[dict]) -> str:
    context = "\n\n".join(
        f"Chunk {record['rank']} from {record['source_file']}:\n{record['text']}"
        for record in records
    )
    messages = [
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
    ]
    response = litellm.completion(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        extra_body={"reasoning_effort": "low"},
    )
    llm_chunks.time.sleep(llm_chunks.PAUSE_SECONDS)
    return llm_chunks.message_text(response)


def retrieval_for(item: dict, orders: dict[str, list[dict]], store) -> list[dict]:
    saved = orders.get(item["question"])
    if saved:
        return saved
    chunks = llm_chunks.load_saved()["chunks"]
    found = llm_chunks.llm_rerank(
        item["question"],
        llm_chunks.vector_search(store, item["question"], llm_chunks.CANDIDATES),
    )
    return llm_chunks.ranked_records(found)


def summarize(answers: list[dict]) -> dict:
    summary = {
        "in_scope": {"kept": 0, "total": 0, "misses": []},
        "refusals": {"kept": 0, "total": 0, "misses": []},
    }
    for item in answers:
        bucket = summary["in_scope"] if item["kind"] == "in_scope" else summary["refusals"]
        bucket["total"] += 1
        passed = item["status"] == "cited" if item["kind"] == "in_scope" else item["final"] == REFUSAL
        if passed:
            bucket["kept"] += 1
        else:
            bucket["misses"].append(item["question"])
    return summary


def print_summary(summary: dict) -> None:
    for label in ("in_scope", "refusals"):
        row = summary[label]
        print(f"  {label}: {row['kept']}/{row['total']}")
        for question in row["misses"]:
            print(f"    miss: {question}")


def main() -> None:
    if "--show" in sys.argv and ANSWERS_PATH.exists():
        payload = load_answers()
        print_summary(payload["summary"])
        return

    llm_chunks.require_key()
    questions = load_all_questions()
    orders = saved_orders()
    payload = load_answers()
    done = {item["question"] for item in payload["answers"]}
    store = None
    for item in questions:
        if item["question"] in done:
            print(f"skip {item['question']}", flush=True)
            continue
        if item["question"] not in orders and store is None:
            store = llm_chunks.open_store()
        print(f"answer: {item['question']}", flush=True)
        records = retrieval_for(item, orders, store)
        raw = generate(item["question"], records)
        final, quote, status = enforce(raw, [record["text"] for record in records])
        payload["answers"].append(
            {
                "question": item["question"],
                "kind": "in_scope" if item["keywords"] else "refusal",
                "reference": item["answer"],
                "source": item["source"],
                "chunks": [{"rank": record["rank"], "source_file": record["source_file"]} for record in records],
                "raw": raw,
                "final": final,
                "quote": quote,
                "status": status,
            }
        )
        payload["summary"] = summarize(payload["answers"])
        save_answers(payload)
    payload["summary"] = summarize(payload["answers"])
    save_answers(payload)
    print_summary(payload["summary"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
