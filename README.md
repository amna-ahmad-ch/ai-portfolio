# Amna Ahmad

Portfolio site. Ask My Docs is the first project: retrieval over 12 USDA food-safety sheets, with the comparison saved in `eval/results.json`.

The public pages are the files in `site/`. Rebuild them after an eval run:

```
uv run python scripts/site_page.py
```

The ask box stays on this computer, because search loads a local embedding model and the Groq key stays in `.env`.

```
uv run python app.py
```

Then open http://127.0.0.1:8765
