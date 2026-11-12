"""Build the static portfolio from the saved evaluation.

The home page lists projects. FoodSafety RAG is the first project page.
The public copy is the write-up and links to the Render demo.

    uv run python scripts/site_page.py
"""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "eval" / "results.json"
HOME_TEMPLATE = ROOT / "web" / "home.html"
PROJECT_TEMPLATE = ROOT / "web" / "project.html"
TRY_TEMPLATE = ROOT / "web" / "try.html"
SITE = ROOT / "site"
DEMO_URL = "https://ask-my-docs.onrender.com/"
PORTFOLIO_URL = "https://ai-portfolio-2ij.pages.dev/"
WRITEUP_URL = "https://ai-portfolio-2ij.pages.dev/projects/ask-my-docs/"
PROJECT_PAGE = SITE / "projects" / "ask-my-docs" / "index.html"

SEARCH_LABELS = {
    "bm25": "BM25",
    "vector": "Vector",
    "hybrid": "Hybrid",
    "hybrid-rerank": "Cross-encoder",
    "llm-rerank": "Groq rerank",
}

CHUNK_LABELS = {
    "character-500": "500 characters",
    "character-1000": "1,000 characters",
    "section": "Sections",
    "llm-chunks": "AI-written passages",
}

SEARCH_TIPS = {
    "vector": "sentence-transformers/all-MiniLM-L6-v2",
    "hybrid": "Using RRF",
    "hybrid-rerank": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm-rerank": "openai/gpt-oss-120b",
}

LOCAL_ASK = """
    <section class="panel">
      <h2>Ask the sheets</h2>
      <p class="note">This is hybrid search, not the winning reranker. Keyword rank and OpenAI vector rank are fused, then Groq writes from the top five chunks. Five questions are allowed from this browser. These are US rules.</p>
      <div class="chips">
        <button type="button" data-question="What is the Danger Zone for food?">Danger Zone</button>
        <button type="button" data-question="How long can leftovers stay in the refrigerator?">Leftovers</button>
        <button type="button" data-question="What is the capital of France?">A refusal</button>
      </div>
      <form id="ask">
        <label for="question">Question</label>
        <textarea id="question" name="question" maxlength="400" placeholder="How long can leftovers stay in the refrigerator?"></textarea>
        <div class="row">
          <button type="submit" id="send">Ask</button>
          <span class="quota" id="quota">5 questions left</span>
        </div>
      </form>
      <div id="result" aria-live="polite"></div>
    </section>
    <script>
      const form = document.querySelector("#ask");
      const question = document.querySelector("#question");
      const send = document.querySelector("#send");
      const result = document.querySelector("#result");
      const quota = document.querySelector("#quota");

      document.querySelectorAll("[data-question]").forEach((chip) => {
        chip.addEventListener("click", () => {
          question.value = chip.dataset.question;
          question.focus();
        });
      });

      question.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          form.requestSubmit();
        }
      });

      function line(className, text) {
        const node = document.createElement("p");
        node.className = className;
        node.textContent = text;
        return node;
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = question.value.trim();
        if (!text) {
          result.replaceChildren(line("error", "Type a question first."));
          return;
        }
        send.disabled = true;
        form.setAttribute("aria-busy", "true");
        result.replaceChildren(line("note", "Asking the sheets. This can take a little while."));
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
            result.replaceChildren(line("error", body.error || "The request failed."));
            return;
          }
          const reply = document.createElement("div");
          reply.className = "reply";
          reply.append(line("answer", body.answer));
          if (body.status === "rejected") {
            reply.append(line("note", "Relevant text was found, but the reply did not contain a verified sentence, so it was replaced."));
          }
          if (body.quote) reply.append(line("quote", body.quote));
          const passages = document.createElement("section");
          passages.className = "passages";
          const heading = document.createElement("h3");
          heading.textContent = "Retrieved chunks";
          const chunks = document.createElement("ol");
          chunks.className = "chunks";
          (body.chunks || []).forEach((item) => {
            const row = document.createElement("li");
            const name = document.createElement("p");
            name.className = "source";
            name.textContent = item.rank + ". " + item.source;
            const excerpt = document.createElement("p");
            excerpt.textContent = item.text || "";
            row.append(name, excerpt);
            chunks.append(row);
          });
          passages.append(heading, chunks);
          result.replaceChildren(reply, ...(body.chunks || []).length ? [passages] : [], line("rules", "These are US rules."));
        } catch (error) {
          result.replaceChildren(line("error", "The demo did not answer. The write-up link above still works."));
        } finally {
          send.disabled = false;
          form.removeAttribute("aria-busy");
        }
      });
    </script>
"""

def try_button(url: str) -> str:
    href = escape(url, quote=True)
    return f'<a class="go" href="{href}">Open the live demo</a>'


TRY_NOTE = """
    <section class="panel">
      <h2>Try it live</h2>
      <p class="note">This page is the write-up. The ask box is a separate live demo on Render’s free plan. The demo may take about a minute to wake up. If that page does not open, this write-up is complete without it.</p>
    </section>
"""


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def tip(text: str, title: str) -> str:
    return f'<span class="tip" title="{escape(title)}">{escape(text)}</span>'


def search_label(search: str) -> str:
    label = SEARCH_LABELS.get(search, search)
    title = SEARCH_TIPS.get(search)
    if title is None:
        return escape(label)
    return tip(label, title)


def row_mrr(saved: dict, chunking: str, search: str) -> str:
    for item in saved["rows"]:
        if item["chunking"] == chunking and item["search"] == search:
            return f"{item['mrr']:.3f}"
    return ""


def best_chart(saved: dict) -> str:
    best: dict[str, dict] = {}
    for item in saved["rows"]:
        current = best.get(item["chunking"])
        if current is None or item["mrr"] > current["mrr"]:
            best[item["chunking"]] = item
    items = []
    for name in ("character-500", "character-1000", "section", "llm-chunks"):
        item = best[name]
        width = max(0.0, min(100.0, item["mrr"] * 100))
        items.append(
            "<li>"
            f'<div class="name"><span>{escape(CHUNK_LABELS[name])}</span>'
            f'<span>{tip(f"{item["mrr"]:.3f}", "Ranking score (MRR)")} · {search_label(item["search"])}</span></div>'
            f'<div class="track"><span style="width:{width:.1f}%"></span></div>'
            "</li>"
        )
    scale = '<div class="scale" aria-hidden="true"><span>0</span><span>1</span></div>'
    return '<ul class="bars">\n' + "\n".join(items) + "\n</ul>\n" + scale


def table_rows(saved: dict) -> str:
    winner = saved["winner"]
    body = []
    for item in saved["rows"]:
        is_winner = (item["chunking"], item["search"]) == (winner["chunking"], winner["search"])
        tag = '<span class="tag">Best score</span>' if is_winner else ""
        body.append(
            f'<tr class="{"winner" if is_winner else ""}">'
            f"<td>{escape(CHUNK_LABELS.get(item['chunking'], item['chunking']))}</td>"
            f"<td>{search_label(item['search'])}{tag}</td>"
            f'<td class="num">{item["hits"]}/{item["total"]}</td>'
            f'<td class="num">{item["mrr"]:.3f}</td>'
            "</tr>"
        )
    return "\n".join(body)


def stats_line(saved: dict) -> str:
    answers = saved["answers"]
    line = (
        f"Cited answers kept {answers['in_scope_cited']} of {answers['in_scope_total']} "
        f"in-scope replies. Refusals held on {answers['refusals_kept']} of {answers['refusals_total']}."
    )
    judge = answers.get("judge", {}).get("in_scope")
    if judge:
        line += (
            f" A judge scored those shipped answers {judge['accuracy']:.2f} for accuracy, "
            f"{judge['completeness']:.2f} for completeness, and {judge['relevance']:.2f} for relevance, out of 5."
        )
    return line


def render_home(demo_url: str) -> str:
    page = HOME_TEMPLATE.read_text(encoding="utf-8")
    return page.replace("{{demo}}", escape(demo_url, quote=True))


def render_try(home_url: str, writeup_url: str) -> str:
    page = TRY_TEMPLATE.read_text(encoding="utf-8")
    return (
        page.replace("{{home}}", escape(home_url, quote=True))
        .replace("{{writeup}}", escape(writeup_url, quote=True))
        .replace("{{ask}}", LOCAL_ASK)
    )


def render_project(demo_url: str, saved: dict | None = None) -> str:
    saved = load_results() if saved is None else saved
    page = PROJECT_TEMPLATE.read_text(encoding="utf-8")
    winner = saved["winner"]
    answers = saved["answers"]
    judge = answers.get("judge", {}).get("in_scope", {})
    return (
        page.replace("{{rows}}", table_rows(saved))
        .replace("{{chart}}", best_chart(saved))
        .replace("{{try}}", try_button(demo_url))
        .replace("{{ask}}", TRY_NOTE)
        .replace("{{hits}}", str(winner["hits"]))
        .replace("{{total}}", str(winner["total"]))
        .replace("{{mrr}}", f"{winner['mrr']:.3f}")
        .replace("{{vector_mrr}}", row_mrr(saved, "llm-chunks", "vector"))
        .replace("{{hybrid_mrr}}", row_mrr(saved, "llm-chunks", "hybrid"))
        .replace("{{cited_total}}", str(answers["in_scope_total"]))
        .replace("{{cited}}", str(answers["in_scope_cited"]))
        .replace("{{refusals_total}}", str(answers["refusals_total"]))
        .replace("{{refusals}}", str(answers["refusals_kept"]))
        .replace("{{accuracy}}", f"{judge.get('accuracy', 0):.2f}")
        .replace("{{completeness}}", f"{judge.get('completeness', 0):.2f}")
        .replace("{{relevance}}", f"{judge.get('relevance', 0):.2f}")
    )


def build() -> None:
    saved = load_results()
    home = SITE / "index.html"
    home.write_text("<!-- generated by scripts/site_page.py -->\n" + render_home(DEMO_URL), encoding="utf-8")
    PROJECT_PAGE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_PAGE.write_text(
        "<!-- generated by scripts/site_page.py -->\n" + render_project(DEMO_URL, saved),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
    print(f"Wrote {SITE.relative_to(ROOT)}")
