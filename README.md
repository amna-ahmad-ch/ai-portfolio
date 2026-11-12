# Amna Ahmad

Portfolio site. FoodSafety RAG is the first project.

## Results at a glance

I compared four ways of cutting twelve USDA food-safety sheets and four ways of searching them. Retrieval was scored on 16 questions the sheets can answer.

How the documents were cut mattered more than how they were searched. With LLM-written chunks, vector search already found 16 of 16. A Groq reranker raised MRR from 0.886 to 0.964.

The sheets repeat 40°F, 2 hours, and 165°F, so near-duplicate chunks compete. That is why the corpus is food safety.

A reply is shown only when it copies a sentence from a retrieved chunk. That check kept 7 of 16 in-scope answers. The judge scored those shipped answers 2.62, 2.50, and 2.69 out of 5.

The table's winner is LLM chunks plus Groq rerank on a local MiniLM index. The ask box is a different configuration: the same 109 chunks, hybrid search, and OpenAI embeddings, with no rerank and no PyTorch.

## Run it

```
uv run python app.py
```

Open http://127.0.0.1:8765. The project page is http://127.0.0.1:8765/projects/ask-my-docs/.

`OPENAI_API_KEY` and `GROQ_API_KEY` stay in `.env`. Build the OpenAI chunk vectors once with `uv run python scripts/hosted.py --embed`.

The public files in `site/` are rebuilt with `uv run python scripts/site_page.py`. Cloudflare Pages serves that copy: the write-up, with a Try it live button. Render serves the ask box at the service root. Locally the same demo is at http://127.0.0.1:8765/try. The public demo is https://ask-my-docs.onrender.com/ once that service is deployed. Render’s free plan sleeps, so the first visit can take about a minute.

Notes for changing the evaluation are in `CONTRIBUTING.md`.
