"""Build the Pinecone index `ironstore-v3` from the PDFs in ../documents (same steps as notebook v3).

    python scripts/build_index.py            # creates / fills the index if needed
    python scripts/build_index.py --rebuild  # wipes and re-uploads everything

Needs OPENAI_API_KEY and PINECONE_API_KEY. Takes about a minute and costs about 2 cents.
"""

import os
import re
import sys
import time
from bisect import bisect_right
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"
INDEX_NAME = os.getenv("PINECONE_INDEX", "ironstore-v3")
EMBED_MODEL, EMBED_DIM = "text-embedding-3-small", 1536
SPARSE_MODEL = "pinecone-sparse-english-v0"

BULLET = r"(?:•\s*)?"
CATEGORY_LINE = re.compile(rf"^{BULLET}Category:")
FIELD_LINE = re.compile(rf"^{BULLET}(?:SKU|Product title|Product code|Item code|Brand)\b")
PAGE_LINE = re.compile(r"^Page \d+")
CODE = r"[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+"

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=450, chunk_overlap=60)


def read_pdf(path):
    pages = [page.get_text() for page in pymupdf.open(path)]
    text = "".join(pages)
    head = text[:600]
    match = re.search(r"Last reviewed(?: in|:)?\s*([A-Z][a-z]+ \d{4})", head)
    last_reviewed = match.group(1) if match else "unknown"
    year = int(last_reviewed[-4:]) if match else 0
    if path.parent.name == "product_catalog":
        title = path.stem.replace("IronStore_", "").replace("_", " ")
    else:
        title = head.strip().splitlines()[0]
    page_starts, pos = [], 0
    for page in pages:
        page_starts.append(pos)
        pos += len(page)
    return {"path": path, "file": path.name, "area": path.parent.name,
            "doc_type": "catalog" if path.parent.name == "product_catalog" else "policy",
            "title": title, "last_reviewed": last_reviewed, "outdated": 0 < year < 2026,
            "text": text, "page_starts": page_starts}


def product_starts(text):
    """Every product has a 'Category:' line; walk up over SKU/title lines and the name to find its start."""
    lines, offsets, pos = text.split("\n"), [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    starts = []
    for i, line in enumerate(lines):
        if not CATEGORY_LINE.match(line):
            continue
        j = i - 1
        while j >= 0 and FIELD_LINE.match(lines[j]):
            j -= 1
        if j >= 0 and not PAGE_LINE.match(lines[j]):
            j -= 1
        if j >= 0 and PAGE_LINE.match(lines[j]):
            j -= 1
        starts.append(offsets[j + 1])
    return starts


def product_metadata(block):
    def field(pattern):
        match = re.search(pattern, block, re.M)
        return match.group(1).strip() if match else ""
    name = next(l for l in block.split("\n") if l.strip() and not PAGE_LINE.match(l))
    name = re.sub(rf"^(?:\d+\.\s+|{CODE}\s+[—–-]\s+)", "", name.strip())
    sku = field(rf"SKU:\s*({CODE})") or field(rf"^(?:Page[^\n]*\n)?({CODE})\s+[—–-]")
    price = field(r"Price:\s*€\s*([\d.,]+)").replace(",", "")
    meta = {"name": name, "category": field(rf"^{BULLET}Category:\s*(.+)$"),
            "availability": field(rf"^{BULLET}Availability:\s*(.+)$")}
    if sku:
        meta["sku"] = sku
    if price:
        meta["price"] = float(price)
    return meta


def make_chunk(doc, n, start, body, extra=None):
    page = bisect_right(doc["page_starts"], start)
    header = (f"[{doc['title']} | {doc['area']} | page {page} | last reviewed {doc['last_reviewed']}"
              + (" | OUTDATED" if doc["outdated"] else "") + "]")
    return {"id": f"{doc['path'].stem}-{n:04d}", "text": header + "\n" + body.strip(),
            "source": doc["file"], "title": doc["title"], "area": doc["area"], "doc_type": doc["doc_type"],
            "page": page, "last_reviewed": doc["last_reviewed"], "outdated": doc["outdated"], **(extra or {})}


def split_text(doc, text, offset, first_n, extra=None):
    out, start = [], 0
    for piece in splitter.split_text(text):
        start = max(text.find(piece, start), start)
        out.append(make_chunk(doc, first_n + len(out), offset + start, piece, extra))
    return out


def build_chunks():
    chunks = []
    for doc in (read_pdf(p) for p in sorted(DOCS_DIR.rglob("*.pdf"))):
        text = doc["text"]
        if doc["doc_type"] == "policy":
            chunks += split_text(doc, text, 0, 0, {"kind": "policy"})
            continue
        starts = product_starts(text)
        doc_chunks = split_text(doc, text[:starts[0]], 0, 0, {"kind": "catalog intro"})
        for a, b in zip(starts, starts[1:] + [len(text)]):
            block, meta = text[a:b], {"kind": "product", **product_metadata(text[a:b])}
            if len(block) <= 2500:
                doc_chunks.append(make_chunk(doc, len(doc_chunks), a, block, meta))
            else:
                for part in split_text(doc, block, a, len(doc_chunks), meta):
                    part["text"] = part["text"].replace("\n", f"\n{meta['name']}\n", 1)
                    doc_chunks.append(part)
        chunks += doc_chunks
    return chunks


def sparse_embed(pc, texts):
    out = []
    for i in range(0, len(texts), 96):
        for attempt in range(5):
            try:
                result = pc.inference.embed(model=SPARSE_MODEL, inputs=texts[i:i + 96],
                                            parameters={"input_type": "passage", "truncate": "END"})
                break
            except Exception as error:
                if "429" not in str(error) or attempt == 4:
                    raise
                time.sleep(20)
        out += [{"indices": e.sparse_indices, "values": e.sparse_values} for e in result.data]
    return out


def main():
    rebuild = "--rebuild" in sys.argv
    chunks = build_chunks()
    print(len(chunks), "chunks")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="dotproduct",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index(host=pc.describe_index(INDEX_NAME).host)
    already = index.describe_index_stats().total_vector_count
    if not rebuild and already == len(chunks):
        print("index already complete:", already, "vectors")
        return
    if already:
        index.delete(delete_all=True)
    texts = [c["text"] for c in chunks]
    client = OpenAI()
    dense = []
    for i in range(0, len(texts), 500):
        dense += [d.embedding for d in client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 500]).data]
    sparse = sparse_embed(pc, texts)
    index.upsert(vectors=[{"id": c["id"], "values": d, "sparse_values": s, "metadata": c}
                          for c, d, s in zip(chunks, dense, sparse)], batch_size=100, show_progress=False)
    print("uploaded", len(chunks), "vectors to", INDEX_NAME)


if __name__ == "__main__":
    main()
