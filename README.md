# Amna Ahmad

Portfolio site. Ask My Docs is the first project: retrieval over 12 USDA food-safety sheets, with the comparison saved in `eval/results.json`.

The public pages are the files in `site/`. Rebuild them after an eval run:

```
uv run python scripts/site_page.py
```

The ask box uses hybrid search over the 109 LLM chunks. Keyword rank and OpenAI `text-embedding-3-small` vector rank are fused, then Groq writes the reply from the top 5. Build the chunk vectors once with `uv run python scripts/hosted.py --embed`. `OPENAI_API_KEY` and `GROQ_API_KEY` stay in `.env` locally and in Render environment variables when deployed. The comparison table is still the saved MiniLM evaluation.

```
uv run python app.py
```

Then open http://127.0.0.1:8765

On Render, create a Blueprint from this repo, or a Python web service with build command `pip install -r requirements-render.txt` and start command `python app.py`. Set `OPENAI_API_KEY` and `GROQ_API_KEY` in the Render dashboard. The free service sleeps after 15 minutes with no visitors and wakes on the next request.
