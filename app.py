"""Local portfolio server.

The home page lists projects. Ask My Docs is served with the ask box.
A question is answered by the locked winner only: Groq rerank over the
LLM chunks, then the citation check. The Groq key stays in .env.

    uv run python app.py
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import answer
import llm_chunks
import site_page

SITE = ROOT / "site"
QUESTION_LIMIT = 5
HOST = "127.0.0.1"
PORT = 8765
PROJECT_PATHS = {
    "/projects/ask-my-docs",
    "/projects/ask-my-docs/",
    "/projects/ask-my-docs/index.html",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}

store = None
counts: dict[str, int] = {}
ask_lock = threading.Lock()


def site_file(url_path: str) -> Path | None:
    rel = url_path.split("?", 1)[0]
    if rel == "/":
        rel = "/index.html"
    if rel.endswith("/"):
        rel += "index.html"
    candidate = (SITE / rel.lstrip("/")).resolve()
    if not candidate.is_relative_to(SITE.resolve()) or not candidate.is_file():
        return None
    return candidate


def answer_question(question: str) -> dict:
    pool = llm_chunks.vector_search(store, question, llm_chunks.CANDIDATES)
    ranked = llm_chunks.llm_rerank(question, pool, pause=False)
    records = llm_chunks.ranked_records(ranked)
    raw = answer.generate(question, records, pause=False)
    final, quote, _status = answer.enforce(raw, [record["text"] for record in records])
    return {
        "answer": final,
        "quote": quote,
        "sources": [{"rank": record["rank"], "source": record["source_file"]} for record in records],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)

    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict) -> None:
        self.send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in PROJECT_PATHS:
            page = site_page.render_project(site_page.LOCAL_ASK).encode("utf-8")
            self.send_bytes(200, "text/html; charset=utf-8", page)
            return
        file = site_file(path)
        if file is None:
            self.send_error(404)
            return
        content_type = CONTENT_TYPES.get(file.suffix, "application/octet-stream")
        self.send_bytes(200, content_type, file.read_bytes())

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Send the question as JSON."})
            return
        question = str(payload.get("question", "")).strip()
        if not question:
            self.send_json(400, {"error": "Type a question first."})
            return
        if len(question) > 400:
            self.send_json(400, {"error": "Keep the question under 400 characters."})
            return

        visitor = self.client_address[0]
        used = counts.get(visitor, 0)
        if used >= QUESTION_LIMIT:
            self.send_json(429, {"error": "Five questions are the limit from this browser.", "remaining": 0})
            return

        try:
            with ask_lock:
                result = answer_question(question)
        except Exception:
            self.send_json(502, {"error": "The model did not answer. Try once more in a minute.", "remaining": QUESTION_LIMIT - used})
            return

        counts[visitor] = used + 1
        result["remaining"] = QUESTION_LIMIT - counts[visitor]
        self.send_json(200, result)


def main() -> None:
    global store
    llm_chunks.require_key()
    store = llm_chunks.open_store()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Portfolio at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
