"""Score every saved retrieval row and compare the winner with the baseline.

Local searches are scored again. The Groq chunk and answer rows are read
from the files already written, so this command does not call the API.
The baseline file is not rewritten. The command exits with an error when
the locked winner has fewer hits or a lower MRR than eval/baseline.json.

    uv run python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import keyword_search as search
import site_page

RESULTS_PATH = ROOT / "eval" / "results.json"
BASELINE_PATH = ROOT / "eval" / "baseline.json"
LLM_RESULTS_PATH = ROOT / "eval" / "llm_results.json"
ANSWER_RESULTS_PATH = ROOT / "eval" / "answer_results.json"
WINNER = ("llm-chunks", "llm-rerank")


def row(chunking: str, method: str, hits: int, total: int, mrr: float, misses: list[str]) -> dict:
    return {
        "chunking": chunking,
        "search": method,
        "hits": hits,
        "total": total,
        "mrr": round(mrr, 3),
        "misses": misses,
    }


def local_rows(questions: list[dict]) -> list[dict]:
    rows = []
    total = len(questions)
    for name in search.COLLECTIONS:
        store, chunks = search.load_chunks(name)
        checks = {
            "bm25": lambda question, chunks=chunks: search.bm25_search(chunks, question),
            "vector": lambda question, store=store: search.vector_search(store, question),
            "hybrid": lambda question, chunks=chunks, store=store: search.hybrid_search(chunks, store, question),
            "hybrid-rerank": lambda question, chunks=chunks, store=store: search.hybrid_rerank(chunks, store, question),
        }
        for method, search_one in checks.items():
            hits, misses, mrr = search.score(questions, search_one)
            rows.append(row(name, method, len(hits), total, mrr, misses))
            print(f"{name} / {method}: {len(hits)}/{total}  MRR {mrr:.3f}", flush=True)
    return rows


def saved_llm_rows() -> list[dict]:
    saved = json.loads(LLM_RESULTS_PATH.read_text(encoding="utf-8"))
    total = saved["questions"]
    rows = []
    for method in ("bm25", "vector", "hybrid", "llm-rerank"):
        item = saved[method]
        rows.append(row("llm-chunks", method, item["hits"], total, item["mrr"], item["misses"]))
    return rows


def answer_summary() -> dict:
    saved = json.loads(ANSWER_RESULTS_PATH.read_text(encoding="utf-8"))
    summary = saved["summary"]
    return {
        "in_scope_cited": summary["in_scope"]["kept"],
        "in_scope_total": summary["in_scope"]["total"],
        "in_scope_misses": summary["in_scope"]["misses"],
        "refusals_kept": summary["refusals"]["kept"],
        "refusals_total": summary["refusals"]["total"],
        "refusal_misses": summary["refusals"]["misses"],
    }


def winner_record(item: dict) -> dict:
    return {key: item[key] for key in ("chunking", "search", "hits", "total", "mrr")}


def check_baseline(winner: dict) -> None:
    """Fail when the locked winner is below the saved baseline."""
    if not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {BASELINE_PATH.name}")
        return
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    same_row = (winner["chunking"], winner["search"]) == (baseline["chunking"], baseline["search"])
    dropped = winner["hits"] < baseline["hits"] or round(winner["mrr"], 3) < round(baseline["mrr"], 3)
    if same_row and not dropped:
        print(
            f"Baseline held: {baseline['hits']}/{baseline['total']}  MRR {baseline['mrr']:.3f}"
        )
        return
    raise SystemExit(
        "Winner dropped below baseline: "
        f"{winner['chunking']} / {winner['search']} "
        f"{winner['hits']}/{winner['total']} MRR {winner['mrr']:.3f} "
        f"is below {baseline['chunking']} / {baseline['search']} "
        f"{baseline['hits']}/{baseline['total']} MRR {baseline['mrr']:.3f}"
    )


def main() -> None:
    questions = [item for item in search.load_questions() if item["keywords"]]
    rows = local_rows(questions) + saved_llm_rows()
    answers = answer_summary()
    winner = winner_record(next(item for item in rows if (item["chunking"], item["search"]) == WINNER))
    payload = {
        "questions": len(questions),
        "rows": rows,
        "answers": answers,
        "winner": winner,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWinner: {winner['chunking']} / {winner['search']}: {winner['hits']}/{winner['total']}  MRR {winner['mrr']:.3f}")
    print(f"Cited answers: {answers['in_scope_cited']}/{answers['in_scope_total']}")
    print(f"Refusals: {answers['refusals_kept']}/{answers['refusals_total']}")
    print(f"Wrote {RESULTS_PATH.name}")
    site_page.build()
    print("Wrote site/")
    check_baseline(winner)


if __name__ == "__main__":
    main()
