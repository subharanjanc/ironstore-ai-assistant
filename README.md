# IronStore Enterprise AI Assistant

**Live app: https://ironstore-ai-assistant-one.vercel.app/**
No sign-in needed.

| | |
|---|---|
| Live | https://ironstore-ai-assistant-one.vercel.app/ |
| API docs | https://ironstore-ai-assistant-one.vercel.app/docs |
| Health | https://ironstore-ai-assistant-one.vercel.app/health |

Mini-project, Ironhack AI Engineering FT Sept 26.
Brief: `ironhack-rmt-AI-student-materials/project-brief-enterprise-ai-assistant`

Submitted by Subharanjan Chakraborty.

Employees ask questions in plain language (any language) about IronStore's internal
policies and product catalogs. The assistant answers **only from the documents**, cites
every fact, links each source to the **right PDF page**, warns when information comes
from the outdated 2021 catalog, and says so when the answer isn't in the documents.

## How it works

```
question ──► follow-up? → rewrite as a standalone question (chat history)
         ──► router (policy / product / both, SKU, max price)  ∥  3 rewritten questions
         ──► SKU in the question? → exact metadata lookup
             otherwise hybrid search (dense + sparse) for all 4 versions, with metadata filters
         ──► dedup (by id and by text) ──► rerank (bge-reranker-v2-m3) ──► top 6
         ──► weak product match? → also search the 2021 catalog (flagged outdated)
         ──► answer (gpt-4.1-mini) with [S1] citations + PDF links
```

| Choice | Why |
|---|---|
| **Pinecone serverless**, one hybrid index (`dotproduct`) | dense = meaning, sparse = exact SKUs / IDs, in one query |
| `text-embedding-3-small` + `pinecone-sparse-english-v0` | cheap dense vectors; hosted keyword model, nothing to train |
| **One chunk per product** (3,291 products) + `sku`, `price`, `category` metadata | products are never cut in half; filters on SKU and price |
| Policies: 450-token chunks with a `[title · area · page · last reviewed]` header | the header carries context into the search |
| **Router** (gpt-4o-mini, structured output) → Pinecone filters | policy questions never see 3,000 product chunks; "under 300 €" becomes `price ≤ 300` |
| **Multi-query** (3 English rewrites) + **dedup** | different wording finds more; a German question still matches English documents |
| **Reranker** for ordering only | its scores are relative, so they are never used to refuse an answer |
| gpt-4o-mini for small jobs, gpt-4.1-mini for the answer | a small *model routing*: cheap where it's easy, careful where it matters |

The design was developed step by step in three notebooks (all included):

| Notebook | What it added | What it fixed |
|---|---|---|
| `ironstore_assistant_v1.ipynb` | semantic search, multi-query, dedup, sources | baseline |
| `ironstore_assistant_v2.ipynb` | hybrid search (dense + sparse), automatic α | SKU questions ("couldn't find" → found) |
| `ironstore_assistant_v3.ipynb` | product chunks, router + filters, reranker, old-catalog fallback | price filters, outdated data, German questions |

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | The chat page |
| `POST` | `/ask` | `{question, history}` → answer, sources with PDF links, route, timings |
| `GET` | `/files/{path}` | The source PDFs (opened at the cited page) |
| `GET` | `/health` | Liveness, models, whether the keys are set |
| `GET` | `/docs` | Swagger UI |

### `POST /ask`

```json
{ "question": "And is it in stock?",
  "history": [{ "question": "How much does the Nordview 55-inch TV cost?", "answer": "…€649.00 [S1]." }] }
```

```json
{ "standalone_question": "Is the Nordview NQ-55U7 55-inch 4K QLED Smart TV in stock?",
  "answer": "Yes … 18 units available … Rotterdam DC, Aisle E12, Bin 04 [S1].",
  "sources": [{ "n": 1, "title": "Product Catalog Electronics", "page": 1,
                "url": "/files/product_catalog/IronStore_Product_Catalog_Electronics.pdf#page=1", "outdated": false }],
  "route": { "target": "product" }, "path": "hybrid α=0.7",
  "stats": { "found": 32, "after_dedup": 14, "after_rerank": 6 }, "timings": { "retrieval_s": 2.4, "answer_s": 1.1 } }
```

## Data issues found and handled

- **Outdated:** `IronStore_Old_Product_Catalog_2021.pdf` is marked `outdated`, excluded from normal product search, used only as a fallback, and every answer based on it starts with a warning.
- **Inconsistent IDs:** two different policies share the ID `CS-POL-014` — answers cite file names, not IDs.
- **Mixed layouts:** the 63 catalogs use four different product layouts; the splitter anchors on each product's `Category:` line.

## Run locally

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # add OPENAI_API_KEY and PINECONE_API_KEY
python scripts/build_index.py # once: builds the Pinecone index (~1 min, ~2 cents)
python run_local.py           # http://localhost:8000
```

## Tests

```bash
python test_api.py          # 7 checks, no API calls, free
python test_api.py --live   # adds 5 real questions: SKU, policy, follow-up, outdated warning
```

## Deploy

Vercel, from this repo (`vercel.json`). Set `OPENAI_API_KEY` and `PINECONE_API_KEY` in the
project's environment variables **before** the first deploy — without them `/health` works and
`/ask` returns `503`. The Pinecone index must already exist (`scripts/build_index.py`).

## Brief checklist

| Requirement | Status |
|---|---|
| Answers from internal documents, conversational | done (chat with follow-ups) |
| Acknowledges when it cannot answer | done |
| Handles duplicates / outdated / inconsistent data | done (dedup, outdated flag + warning) |
| Bonus: sources with document, part and link | done (title, page, PDF link) |
| Bonus: API | done (FastAPI, tested) |
| Bonus: frontend | done |
| Bonus: deploy | done — https://ironstore-ai-assistant-one.vercel.app/ |
