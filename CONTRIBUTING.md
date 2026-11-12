# Working on the evaluation

The comparison table is the saved MiniLM evaluation. Do not rescore it in order to change the published numbers, and do not overwrite `eval/baseline.json`.

`eval/run_eval.py` rebuilds the three local Chroma collections' scores and reads the Groq rows from JSON. It fails if the locked winner, LLM chunks with Groq rerank, gets worse. It does not call Groq. The GitHub Action does the same and has no API key.

The live server must not import `scripts/llm_chunks.py` or `scripts/answer.py`. Those imports load PyTorch. `scripts/hosted.py` is the ask path.

`vector_db/` is the local MiniLM index. It is gitignored. `eval/openai_vectors.json` is the live index and is committed.

Do not rerun `scripts/judge.py` unless the saved answers change. The judge scores the reply after the citation check.

Render, if used, must install `requirements-render.txt` and must not install `sentence-transformers` or PyTorch.
