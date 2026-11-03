"""IronStore assistant: the v3 notebook's retrieval pipeline as a module.

    follow-up → standalone question (only when there is chat history)
    router ∥ 3 rewrites → SKU fast path, or hybrid search with metadata filters
    → dedup → rerank → old-catalog fallback → answer with numbered sources

The Pinecone index `ironstore-v3` is built by `scripts/build_index.py` (same chunking as
the notebook); this module only reads from it.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field

INDEX_NAME = os.getenv("PINECONE_INDEX", "ironstore-v3")
EMBED_MODEL = "text-embedding-3-small"
SPARSE_MODEL = "pinecone-sparse-english-v0"
RERANK_MODEL = "bge-reranker-v2-m3"
SMALL_MODEL = os.getenv("SMALL_MODEL", "gpt-4o-mini")      # router, rewrites, follow-ups
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
TOP_K = 8                    # hits per question version
TOP_N = 6                    # chunks the answer model gets after reranking
FALLBACK_RERANK = 0.5        # product match weaker than this → also try the 2021 catalog
MAX_HISTORY_TURNS = 4
LIST_CANDIDATES = 500        # products scanned for a "list all …" question
MAX_LIST = 40                # products the answer model gets for a list

CODE_PATTERN = re.compile(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]+)+\b")

_openai = None
_pc = None
_index = None


def clients():
    """Created on first use, so importing the module (and /health) needs no keys."""
    global _openai, _pc, _index
    if _index is None:
        _openai = OpenAI()
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = _pc.Index(host=_pc.describe_index(INDEX_NAME).host)
    return _openai, _pc, _index


# ---------- models for structured output ----------

class Route(BaseModel):
    target: Literal["policy", "product", "both"] = Field(
        description="policy = internal rules/procedures (HR, IT, finance, legal, sales, support, deliveries); "
                    "product = items we sell (prices, specs, stock); both = needs rules and product info")
    sku: Optional[str] = Field(None, description="only an exact product code written in the question, letters/digits "
                                                 "joined by dashes like PRJ-PCB-PB1080; never a product name; else null")
    max_price: Optional[float] = Field(None, description="upper price limit in EUR if the question gives one, else null")
    wants_old_catalog: bool = Field(False, description="true only if the user asks about old, 2021 or past catalog data")
    list_all: bool = Field(False, description="true if the user wants a list of ALL products matching a condition, "
                                             "e.g. 'which drones do we sell under 300 euros?', 'list our projectors'")
    product_words: list[str] = Field(default_factory=list,
                                     description="for list questions: the product type in English, singular, lower case, "
                                                 "as it would appear in a product name or category, e.g. ['drone']")


class Rewrites(BaseModel):
    questions: list[str] = Field(description="3 alternative versions of the question")


class Standalone(BaseModel):
    question: str = Field(description="the follow-up rewritten as a complete question")


class Turn(BaseModel):
    question: str
    answer: str


def _parse(model, schema, prompt, temperature=0.0):
    client, _, _ = clients()
    result = client.chat.completions.parse(model=model, temperature=temperature, response_format=schema,
                                           messages=[{"role": "user", "content": prompt}])
    return result.choices[0].message.parsed


# ---------- steps ----------

def standalone_question(question, history):
    """'And is it in stock?' → 'Is the Nordview NQ-55U7 TV in stock?' (needs the earlier turns)."""
    if not history:
        return question
    turns = "\n".join(f"Q: {t.question}\nA: {t.answer[:600]}" for t in history[-MAX_HISTORY_TURNS:])
    return _parse(SMALL_MODEL, Standalone,
                  "Rewrite the last question so it can be understood without the conversation. "
                  "Keep product names, SKUs and numbers. If it is already complete, return it unchanged.\n\n"
                  f"Conversation:\n{turns}\n\nLast question: {question}").question


def route_question(question):
    route = _parse(SMALL_MODEL, Route, f"Route this IronStore employee question.\n\nQuestion: {question}")
    if route.sku and not CODE_PATTERN.fullmatch(route.sku.strip().upper()):   # names are not SKUs
        route.sku = None
    if route.sku and route.sku.strip().upper() not in question.upper():      # invented codes are not SKUs
        route.sku = None
    return route


def rewrite_question(question):
    rewrites = _parse(SMALL_MODEL, Rewrites,
                      "You help search IronStore's internal documents (HR, IT, finance, legal, sales, customer support, "
                      "deliveries, product catalogs). Write 3 different versions of the employee's question IN ENGLISH "
                      "(translate if needed; the documents are English). Use other words and synonyms an official "
                      "policy or catalog would use. Keep product names, numbers and SKUs unchanged.\n\n"
                      f"Question: {question}", temperature=0.3)
    return rewrites.questions[:3]


def make_filter(route, outdated=False):
    if route.target == "policy":
        return {"doc_type": {"$eq": "policy"}}
    if route.target == "product":
        flt = {"doc_type": {"$eq": "catalog"}, "outdated": {"$eq": outdated}}
        if route.max_price:
            flt["price"] = {"$lte": route.max_price}
        return flt
    return None if outdated else {"outdated": {"$eq": False}}


def choose_alpha(question):
    return 0.3 if CODE_PATTERN.search(question) else 0.7


def sparse_embed(texts):
    _, pc, _ = clients()
    result = pc.inference.embed(model=SPARSE_MODEL, inputs=texts,
                                parameters={"input_type": "query", "truncate": "END"})
    return [{"indices": e.sparse_indices, "values": e.sparse_values} for e in result.data]


def dense_embed(texts):
    client, _, _ = clients()
    return [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts).data]


def search(dense_vec, sparse_vec, alpha, flt, k=TOP_K):
    _, _, index = clients()
    alpha = min(max(alpha, 0.01), 1.0)
    result = index.query(
        vector=[v * alpha for v in dense_vec],
        sparse_vector={"indices": sparse_vec["indices"], "values": [v * (1 - alpha) for v in sparse_vec["values"]]},
        top_k=k, include_metadata=True, filter=flt)
    return [{"id": m.id, "score": round(m.score, 3), **m.metadata} for m in result.matches]


def dedup(hits):
    best = {}
    for hit in hits:
        best.setdefault(hit["id"], {**hit, "hits": 0})
        best[hit["id"]]["hits"] += 1
        best[hit["id"]]["score"] = max(best[hit["id"]]["score"], hit["score"])
    unique, seen = [], set()
    for hit in sorted(best.values(), key=lambda h: (-h["hits"], -h["score"])):
        body = " ".join(hit["text"].split("\n", 1)[-1].split()).lower()
        if body not in seen:
            seen.add(body)
            unique.append(hit)
    return unique


def rerank(question, candidates, top_n=TOP_N):
    if not candidates:
        return []
    _, pc, _ = clients()
    result = pc.inference.rerank(model=RERANK_MODEL, query=question, documents=[c["text"] for c in candidates],
                                 top_n=min(top_n, len(candidates)), return_documents=False)
    return [{**candidates[r.index], "rerank": round(r.score, 3)} for r in result.data]


def list_products(route, dense_vec):
    """'Which drones under 300 €?' → every matching product, not just the 6 best search hits.

    Pinecone filters on catalog / not outdated / price; the product type is then matched in each
    product's name and category metadata. One query, no reranking: a list must be complete, not ranked.
    """
    _, _, index = clients()
    words = [w.lower().strip() for w in route.product_words if w.strip()]
    if not words:
        return []
    flt = {"kind": {"$eq": "product"}, "outdated": {"$eq": route.wants_old_catalog}}
    if route.max_price:
        flt["price"] = {"$lte": route.max_price}
    matches = index.query(vector=dense_vec, top_k=LIST_CANDIDATES, include_metadata=True, filter=flt).matches
    found = {}
    for m in matches:                    # a long product can be split in two chunks: keep it once
        p = m.metadata
        if any(w in f"{p.get('name', '')} {p.get('category', '')}".lower() for w in words):
            found.setdefault(p.get("sku") or p.get("name"), p)
    found = sorted(found.values(), key=lambda p: p.get("price", 0))
    return [{**p, "id": p.get("sku") or p["name"], "score": 1.0, "rerank": 1.0,
             # one compact line per product keeps a 40-product list small for the answer model
             "text": f"[{p['title']} | page {int(p['page'])}] {p['name']} | SKU {p.get('sku', '?')} | "
                     f"€{p.get('price', 0):.2f} | {p.get('availability', '?')} | {p.get('category', '')}"}
            for p in found[:MAX_LIST]]


def retrieve(question):
    log = {}
    with ThreadPoolExecutor(max_workers=2) as pool:                      # router ∥ rewrites
        route_job = pool.submit(route_question, question)
        rewrite_job = pool.submit(rewrite_question, question)
        route = route_job.result()
        queries = [question] + rewrite_job.result()
    log["route"] = route.model_dump(exclude_defaults=True)
    dense_vecs = dense_embed(queries)
    _, _, index = clients()

    if route.sku:                                                          # exact SKU lookup
        hits = index.query(vector=dense_vecs[0], top_k=3, include_metadata=True,
                           filter={"sku": {"$eq": route.sku.strip().upper()}}).matches
        if hits:
            log["path"] = "SKU lookup"
            return [{"id": h.id, "score": 1.0, "rerank": 1.0, **h.metadata} for h in hits], log

    if route.target == "product" and route.list_all:                      # complete product list
        products = list_products(route, dense_vecs[0])
        if products:
            log["path"] = "product list"
            log.update(found=len(products), after_dedup=len(products), after_rerank=len(products))
            return products, log

    sparse_vecs = sparse_embed(queries)
    alpha = choose_alpha(question)

    def run(outdated):
        flt = make_filter(route, outdated)
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            results = pool.map(lambda pair: search(pair[0], pair[1], alpha, flt), zip(dense_vecs, sparse_vecs))
        return [hit for hits in results for hit in hits]

    raw = run(outdated=False)
    log["path"] = f"hybrid α={alpha}"
    kept = rerank(question, dedup(raw))
    best = kept[0]["rerank"] if kept else 0
    if route.target == "product" and (best < FALLBACK_RERANK or route.wants_old_catalog):
        raw += run(outdated=True)
        kept = rerank(question, dedup(raw))
        log["path"] += " + old 2021 catalog"
    log.update(found=len(raw), after_dedup=len(dedup(raw)), after_rerank=len(kept))
    return kept, log


SYSTEM_PROMPT = (
    "You are the IronStore internal assistant for employees.\n"
    "Answer ONLY with information from the sources below. Cite them inline like [S1] or [S2][S3].\n"
    "If the sources do not contain the answer, say: \"I couldn't find this in the internal documents\" "
    "and suggest which department could help. Never invent numbers, names or rules.\n"
    "If sources disagree, prefer the most recently reviewed one and mention the difference.\n"
    "Sources marked OUTDATED come from the 2021 catalog: say clearly that this information is outdated.\n"
    "If the sources are a product list, list every product that is the kind asked for, with price and availability; "
    "leave out accessories or spare parts unless they were asked for.\n"
    "Be concise; use bullet points for procedures and lists. Answer in the language of the question.\n\n"
    "Sources:\n{context}")

OUTDATED_WARNING = "⚠️ **Based on the outdated 2021 catalog — please check the current price and availability.**\n\n"


def ask(question, history=None):
    """Answer one question. `history` = earlier turns of this chat (the browser sends them back)."""
    timings, t0 = {}, time.time()
    standalone = standalone_question(question, history or [])
    kept, log = retrieve(standalone)
    timings["retrieval_s"] = round(time.time() - t0, 2)

    context = "\n\n".join(f"[S{i}] {c['text']}" for i, c in enumerate(kept, 1))
    client, _, _ = clients()
    t1 = time.time()
    answer = client.chat.completions.create(
        model=ANSWER_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                  {"role": "user", "content": standalone}],
    ).choices[0].message.content
    timings["answer_s"] = round(time.time() - t1, 2)

    cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", answer) if 0 < int(n) <= len(kept)})
    if any(kept[n - 1].get("outdated") for n in cited):          # warning added by code, not left to the LLM
        answer = OUTDATED_WARNING + answer
    sources = [{
        "n": n,
        "title": kept[n - 1]["title"],
        "page": int(kept[n - 1]["page"]),
        "area": kept[n - 1]["area"],
        "file": kept[n - 1]["source"],
        "outdated": bool(kept[n - 1].get("outdated")),
        "rerank": kept[n - 1].get("rerank"),
    } for n in cited]
    return {"question": question, "standalone_question": standalone, "answer": answer,
            "sources": sources, "route": log.get("route"), "path": log.get("path"),
            "stats": {k: log[k] for k in ("found", "after_dedup", "after_rerank") if k in log}, "timings": timings}
