"""Checks for the IronStore API.

    python test_api.py          # offline checks, no API calls, free
    python test_api.py --live   # adds real questions (needs both keys, costs < 1 cent)
"""

import os
import socket
import sys

# this PC tries IPv6 first and waits ~10 s per call; prefer IPv4 (no effect elsewhere)
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: sorted(_getaddrinfo(*a, **k), key=lambda x: x[0] != socket.AF_INET)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
from fastapi.testclient import TestClient  # noqa: E402

import index  # noqa: E402

client = TestClient(index.app)
passed = 0


def check(name, condition):
    global passed
    assert condition, name
    passed += 1
    print("ok  ", name)


r = client.get("/health")
check("health is ok", r.status_code == 200 and r.json()["status"] == "ok")
r = client.get("/")
check("chat page is served", r.status_code == 200 and "IronStore Assistant" in r.text)
r = client.get("/files/product_catalog/IronStore_Product_Catalog_Electronics.pdf")
check("source PDF is served", r.status_code == 200 and r.headers["content-type"] == "application/pdf")
r = client.get("/files/../api/rag.py")
check("no files outside documents/", r.status_code == 404)
r = client.post("/ask", json={"question": "x"})
check("too-short question is rejected", r.status_code == 422)
check("document link points to the right folder",
      index.document_url("HR_policies", "IronStore_Remote_Work_Policy.pdf", 2)
      == "/files/internal_docs_by_area/HR_policies/IronStore_Remote_Work_Policy.pdf#page=2")

saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "PINECONE_API_KEY")}
r = client.post("/ask", json={"question": "How do I request time off?"})
check("missing keys -> 503, not a crash", r.status_code == 503)
os.environ.update({k: v for k, v in saved.items() if v})

if "--live" in sys.argv:
    r = client.post("/ask", json={"question": "What is SKU PRJ-PCB-PB1080 and how much does it cost?"})
    data = r.json()
    check("SKU question -> exact lookup, €399", r.status_code == 200 and data["path"] == "SKU lookup"
          and "399" in data["answer"])
    check("source has a working PDF link", client.get(data["sources"][0]["url"].split("#")[0]).status_code == 200)

    r = client.post("/ask", json={"question": "How many days per week do hybrid employees go to the office?"})
    data = r.json()
    check("policy question -> policy route, two days",
          data["route"]["target"] == "policy" and ("two" in data["answer"] or "2" in data["answer"]))

    follow = client.post("/ask", json={"question": "And how much notice must managers give for extra office days?",
                                       "history": [{"question": data["question"], "answer": data["answer"]}]}).json()
    check("follow-up uses the chat history", "office" in follow["standalone_question"].lower()
          and ("five" in follow["answer"].lower() or "5" in follow["answer"]))

    r = client.post("/ask", json={"question": "What is the price of the IronStore 24 inch Full HD office monitor?"})
    check("2021-only product gets the outdated warning", r.json()["answer"].startswith("⚠️"))

    r = client.post("/ask", json={"question": "Which drones do we sell under 300 euros?"})
    data = r.json()
    check("list question -> complete product list under the price limit",
          data["path"] == "product list" and "79.90" in data["answer"] and "189" in data["answer"]
          and "329" not in data["answer"])

print(f"\n{passed} checks passed")
