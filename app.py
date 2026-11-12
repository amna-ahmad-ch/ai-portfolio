"""Portfolio server for this computer and for Render.

Locally, the project page is the write-up and /try is the demo.
On Render, the service root is the demo and links back to the Cloudflare write-up.
A question uses hybrid search over the LLM chunks. Keyword rank and
OpenAI vectors are fused, Groq writes the reply, and the citation check
keeps or replaces it.

    uv run python app.py
"""

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import hosted
import site_page

SITE = ROOT / "site"
QUESTION_LIMIT = 5
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8765"))
PROJECT_PATHS = {
    "/projects/ask-my-docs",
    "/projects/ask-my-docs/",
    "/projects/ask-my-docs/index.html",
}
TRY_PATHS = {"/try", "/try/", "/try/index.html"}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
}

index = None
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
    return hosted.answer_question(index, question)


def visitor_address(handler: BaseHTTPRequestHandler) -> str:
    forwarded = handler.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return handler.client_address[0]


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

    def on_render(self) -> bool:
        return os.environ.get("RENDER") == "true"

    def send_demo(self) -> None:
        if self.on_render():
            home = os.environ.get("PORTFOLIO_URL", site_page.PORTFOLIO_URL)
            writeup = os.environ.get("WRITEUP_URL", site_page.WRITEUP_URL)
        else:
            home, writeup = "/", "/projects/ask-my-docs/"
        page = site_page.render_try(home, writeup).encode("utf-8")
        self.send_bytes(200, "text/html; charset=utf-8", page)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in TRY_PATHS or (path == "/" and self.on_render()):
            self.send_demo()
            return
        if path in {"/", "/index.html"}:
            page = site_page.render_home("/try").encode("utf-8")
            self.send_bytes(200, "text/html; charset=utf-8", page)
            return
        if path in PROJECT_PATHS:
            if self.on_render():
                self.send_response(302)
                self.send_header("Location", os.environ.get("WRITEUP_URL", site_page.WRITEUP_URL))
                self.end_headers()
                return
            page = site_page.render_project("/try").encode("utf-8")
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

        visitor = visitor_address(self)
        used = counts.get(visitor, 0)
        if used >= QUESTION_LIMIT:
            self.send_json(429, {"error": "Five questions are the limit from this browser.", "remaining": 0})
            return

        try:
            with ask_lock:
                result = answer_question(question)
        except Exception as exc:
            detail = re.sub(r"(gsk_|sk-)[A-Za-z0-9_\-]+", "[redacted]", str(exc))
            print(f"ask failed: {type(exc).__name__}: {detail[:300]}", flush=True)
            self.send_json(502, {"error": "The model did not answer. Try once more in a minute.", "remaining": QUESTION_LIMIT - used})
            return

        counts[visitor] = used + 1
        result["remaining"] = QUESTION_LIMIT - counts[visitor]
        self.send_json(200, result)


def main() -> None:
    global index
    hosted.require_key()
    index = hosted.Index()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Portfolio at http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
