"""FastAPI backend for the IronStore Enterprise AI Assistant.

    GET  /                    the chat page
    POST /ask                 a question (+ earlier turns) -> answer with numbered sources
    GET  /files/{path}        the source PDFs, so every answer links to the right page
    GET  /health              liveness, models, whether the keys are set

Stateless: the browser sends the earlier turns back on every call.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

try:
    from . import rag
except ImportError:                                  # pragma: no cover
    import rag

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ROOT / "documents"
PAGE = Path(__file__).resolve().parent / "static" / "index.html"
MAX_QUESTION_CHARS = 1000

app = FastAPI(title="IronStore Enterprise AI Assistant", version="3.0.0",
              description="Answers employee questions from IronStore's internal documents, with sources. "
                          "Hybrid search (Pinecone) + router + reranker. Mini-project, Ironhack AI FT Sept 26.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=MAX_QUESTION_CHARS,
                          description="The employee's question, any language")
    history: list[rag.Turn] = Field(default_factory=list,
                                    description="Earlier question/answer pairs of this chat, oldest first")


class Source(BaseModel):
    n: int
    title: str
    page: int
    area: str
    file: str
    outdated: bool
    rerank: float | None = None
    url: str


class AskResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[Source]
    route: dict | None = None
    path: str | None = None
    stats: dict = {}
    timings: dict = {}


def document_url(area, file, page):
    folder = "product_catalog" if area == "product_catalog" else f"internal_docs_by_area/{area}"
    return f"/files/{folder}/{file}#page={page}"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page():
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "index": rag.INDEX_NAME, "answer_model": rag.ANSWER_MODEL,
            "small_model": rag.SMALL_MODEL, "openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "pinecone_key": bool(os.getenv("PINECONE_API_KEY"))}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY")):
        raise HTTPException(503, "The server has no OPENAI_API_KEY / PINECONE_API_KEY configured.")
    try:
        result = rag.ask(request.question.strip(), request.history)
    except Exception as error:                       # upstream API trouble -> a clear message, not a 500 trace
        raise HTTPException(502, f"Search or answer service failed: {type(error).__name__}") from error
    for source in result["sources"]:
        source["url"] = document_url(source["area"], source["file"], source["page"])
    return result


@app.get("/files/{path:path}", include_in_schema=False)
def files(path: str):
    target = (DOCUMENTS / path).resolve()
    if DOCUMENTS.resolve() not in target.parents or target.suffix.lower() != ".pdf" or not target.is_file():
        raise HTTPException(404, "Document not found")
    return FileResponse(target, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{target.name}"'})
