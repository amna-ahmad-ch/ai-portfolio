"""Build the static portfolio from the saved evaluation.

The home page lists projects. Ask My Docs is the first project page.
The public copy shows the table only. The local server swaps in the ask box.

    uv run python scripts/site_page.py
"""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "eval" / "results.json"
HOME_TEMPLATE = ROOT / "web" / "home.html"
PROJECT_TEMPLATE = ROOT / "web" / "project.html"
SITE = ROOT / "site"
PROJECT_PAGE = SITE / "projects" / "ask-my-docs" / "index.html"

SEARCH_LABELS = {
    "bm25": "BM25",
    "vector": "Vector",
    "hybrid": "Hybrid",
    "hybrid-rerank": "Cross-encoder",
    "llm-rerank": "Groq rerank",
}

LOCAL_ASK = """
    <section class="panel">
      <h2>Ask the winner</h2>
      <p class="note">Five questions are allowed from this browser. If the sheets do not contain the answer, the reply stays “I don’t know based on these documents.”</p>
      <form id="ask">
        <label for="question">Question</label>
        <textarea id="question" name="question" maxlength="400" placeholder="How long can leftovers stay in the refrigerator?"></textarea>
        <div class="row">
          <button type="submit" id="send">Ask</button>
          <span class="quota" id="quota">5 questions left</span>
        </div>
      </form>
      <div id="result"></div>
    </section>
    <script>
      const form = document.querySelector("#ask");
      const question = document.querySelector("#question");
      const send = document.querySelector("#send");
      const result = document.querySelector("#result");
      const quota = document.querySelector("#quota");

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = question.value.trim();
        if (!text) {
          result.innerHTML = '<p class="error">Type a question first.</p>';
          return;
        }
        send.disabled = true;
        result.innerHTML = '<p class="note">Searching the sheets…</p>';
        try {
          const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text }),
          });
          const body = await response.json();
          if (typeof body.remaining === "number") {
            quota.textContent = body.remaining + (body.remaining === 1 ? " question left" : " questions left");
          }
          if (!response.ok) {
            result.innerHTML = '<p class="error"></p>';
            result.querySelector("p").textContent = body.error || "The request failed.";
            return;
          }
          const sources = (body.sources || []).map((item) => `<li>${item.rank}. ${item.source}</li>`).join("");
          result.innerHTML = '<p class="answer"></p><p class="quote"></p><ul class="sources"></ul>';
          result.querySelector(".answer").textContent = body.answer;
          const quote = result.querySelector(".quote");
          if (body.quote) quote.textContent = body.quote;
          else quote.remove();
          result.querySelector(".sources").innerHTML = sources;
        } catch (error) {
          result.innerHTML = '<p class="error">The page could not reach the server.</p>';
        } finally {
          send.disabled = false;
        }
      });
    </script>
"""

PUBLIC_ASK = """
    <section class="panel">
      <h2>Ask on this computer</h2>
      <p class="note">This page shows the saved evaluation. A new question loads a local embedding model and keeps the API key on the server, so the ask box is not published here.</p>
      <p><code>uv run python app.py</code></p>
      <p class="note">Then open http://127.0.0.1:8765/projects/ask-my-docs/</p>
    </section>
"""


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def table_rows(saved: dict) -> str:
    winner = saved["winner"]
    body = []
    for item in saved["rows"]:
        is_winner = (item["chunking"], item["search"]) == (winner["chunking"], winner["search"])
        label = SEARCH_LABELS.get(item["search"], item["search"])
        tag = '<span class="tag">Winner</span>' if is_winner else ""
        body.append(
            f'<tr class="{"winner" if is_winner else ""}">'
            f"<td>{escape(item['chunking'])}</td>"
            f"<td>{escape(label)}{tag}</td>"
            f"<td>{item['hits']}/{item['total']}</td>"
            f"<td>{item['mrr']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(body)


def stats_line(saved: dict) -> str:
    answers = saved["answers"]
    return (
        f"Cited answers kept {answers['in_scope_cited']} of {answers['in_scope_total']} "
        f"in-scope replies. Refusals held on {answers['refusals_kept']} of {answers['refusals_total']}."
    )


def render_home(saved: dict | None = None) -> str:
    saved = load_results() if saved is None else saved
    winner = saved["winner"]
    page = HOME_TEMPLATE.read_text(encoding="utf-8")
    return (
        page.replace("{{hits}}", str(winner["hits"]))
        .replace("{{total}}", str(winner["total"]))
        .replace("{{mrr}}", f"{winner['mrr']:.3f}")
    )


def render_project(ask_html: str, saved: dict | None = None) -> str:
    saved = load_results() if saved is None else saved
    page = PROJECT_TEMPLATE.read_text(encoding="utf-8")
    return (
        page.replace("{{rows}}", table_rows(saved))
        .replace("{{stats}}", escape(stats_line(saved)))
        .replace("{{ask}}", ask_html)
    )


def build() -> None:
    saved = load_results()
    home = SITE / "index.html"
    home.write_text("<!-- generated by scripts/site_page.py -->\n" + render_home(saved), encoding="utf-8")
    PROJECT_PAGE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_PAGE.write_text(
        "<!-- generated by scripts/site_page.py -->\n" + render_project(PUBLIC_ASK, saved),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
    print(f"Wrote {SITE.relative_to(ROOT)}")
