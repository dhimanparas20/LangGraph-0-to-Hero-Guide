# LangGraph 0 to Hero — Part 3: Example Projects

Four self-contained projects that combine the lessons into working apps from
different domains. Each is a single file you can read top to bottom — open it,
run it, change it.

**How to run** (from the repo root, in WSL): `uv run examples/<file>.py`
**Required:** OpenAI key in `.env` (copy `.env.sample`).
**MongoDB / Redis:** all optional — every project detects them at startup and
degrades gracefully (with a one-line notice) when they are offline.

| # | Project | Domain | What it demonstrates |
|---|---------|--------|----------------------|
| 1 | `01_faq_support_bot.py` | Customer support | RAG over a product FAQ (Chroma), adaptive routing (chit-chat vs product questions), auto-opening support tickets in MongoDB when it can't answer. |
| 2 | `02_travel_planner.py` | Travel | Modern `create_agent` that picks its own tools — live weather, city distances, Wikipedia facts (from `modules/tools.py`) — with session memory and optional Redis caching of web results. |
| 3 | `03_document_summarizer.py` | Productivity | Map-reduce summarization of your PDFs (chunk → bullets → one paragraph), summaries embedded into Chroma for search, stored in MongoDB. |
| 4 | `04_expense_tracker.py` | Personal finance | Natural language → structured records via `with_structured_output` ({item, amount, category}), MongoDB as the system of record with an in-memory fallback, category totals. |

## Concepts map (lesson → project)

- Lessons 3 + 6 + 7 (routing, RAG, adaptive) → project 1
- Lessons 4 + 5 + pro 6 (agents, memory, modern API) → project 2
- Pro 3 pattern (map-reduce) + lesson 6 embeddings → project 3
- Lesson 7's structured output + Mongo utilities → project 4

## Notes

- Projects write to Mongo databases `examples.*` when available.
- Project 3 reads whatever PDFs are in `../pdfs/` (a `sample_knowledge.pdf`
  ships with the repo — replace it with your own).
- All four print `[mongo]`/`[redis]` status lines at startup so you always
  know which services are live.

---
*Part of the [LangGraph 0 to Hero Course](../README.md) • CC BY-NC 4.0 — free to learn, modify, and share; selling or paid teaching forbidden.*
