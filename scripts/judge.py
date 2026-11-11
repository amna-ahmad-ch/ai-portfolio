"""Score the shipped answers with Ed's LLM judge.

Each saved reply is compared with its reference answer on accuracy,
completeness, and relevance, from 1 to 5. The reply judged is the final
text after the citation check, which is the text a visitor would see.
A refusal's reference is the don't-know sentence.

Scores are written back into eval/answer_results.json. A later run skips
questions that already have a judge, so Groq is not called again.

    uv run python scripts/judge.py
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import answer
import llm_chunks

PAUSE_SECONDS = 20


class AnswerEval(BaseModel):
    feedback: str = Field(description="Short comparison of the reply with the reference answer")
    accuracy: float = Field(description="1 if the reply is wrong, otherwise up to 5 for a perfect match")
    completeness: float = Field(description="1 if key facts are missing, 5 only if every reference fact is included")
    relevance: float = Field(description="1 if off topic, 5 only if the reply answers the question and adds nothing else")


def reference_text(item: dict) -> str:
    if item["kind"] == "refusal" or item["reference"] == "refuse":
        return answer.REFUSAL
    return item["reference"]


def clamp(score: float) -> float:
    return max(1.0, min(5.0, float(score)))


@retry(wait=wait_exponential(multiplier=1, min=45, max=120), stop=stop_after_attempt(6), reraise=True)
def judge_one(item: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator assessing the quality of answers. "
                "Evaluate the generated answer by comparing it to the reference answer. "
                "Only give 5/5 scores for perfect answers."
            ),
        },
        {
            "role": "user",
            "content": f"""Question:
{item["question"]}

Generated Answer:
{item["final"]}

Reference Answer:
{reference_text(item)}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]
    response = llm_chunks.litellm.completion(
        model=answer.MODEL,
        messages=messages,
        response_format=AnswerEval,
        max_tokens=500,
        extra_body={"reasoning_effort": "low"},
    )
    llm_chunks.time.sleep(PAUSE_SECONDS)
    parsed = AnswerEval.model_validate_json(llm_chunks.message_text(response))
    return {
        "accuracy": clamp(parsed.accuracy),
        "completeness": clamp(parsed.completeness),
        "relevance": clamp(parsed.relevance),
        "feedback": parsed.feedback.strip(),
    }


def mean_scores(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    def avg(key: str) -> float:
        return round(sum(row["judge"][key] for row in rows) / len(rows), 2)

    return {
        "accuracy": avg("accuracy"),
        "completeness": avg("completeness"),
        "relevance": avg("relevance"),
        "count": len(rows),
    }


def summarize_judge(answers: list[dict]) -> dict:
    summary = {}
    for kind in ("in_scope", "refusal"):
        scores = mean_scores([item for item in answers if item["kind"] == kind and item.get("judge")])
        if scores:
            summary[kind] = scores
    return summary


def print_judge(summary: dict) -> None:
    for kind in ("in_scope", "refusal"):
        row = summary.get(kind)
        if not row:
            continue
        print(
            f"  {kind}: accuracy {row['accuracy']:.2f}  "
            f"completeness {row['completeness']:.2f}  "
            f"relevance {row['relevance']:.2f}  "
            f"({row['count']})"
        )


def main() -> None:
    payload = answer.load_answers()
    if "--show" in sys.argv:
        print_judge(payload.get("summary", {}).get("judge", {}))
        return

    llm_chunks.require_key()
    pending = [item for item in payload["answers"] if "judge" not in item]
    for item in pending:
        print(f"judge: {item['question']}", flush=True)
        item["judge"] = judge_one(item)
        scores = item["judge"]
        print(
            f"  accuracy {scores['accuracy']:.0f}  "
            f"completeness {scores['completeness']:.0f}  "
            f"relevance {scores['relevance']:.0f}",
            flush=True,
        )
        payload["summary"]["judge"] = summarize_judge(payload["answers"])
        answer.save_answers(payload)
    payload["summary"]["judge"] = summarize_judge(payload["answers"])
    answer.save_answers(payload)
    print_judge(payload["summary"]["judge"])


if __name__ == "__main__":
    main()
