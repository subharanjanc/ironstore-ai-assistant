"""Run the app on this PC: python run_local.py  ->  http://localhost:8000"""

import os
import socket
import sys

import uvicorn

# this PC tries IPv6 first and waits ~10 s per call; prefer IPv4
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: sorted(_getaddrinfo(*a, **k), key=lambda x: x[0] != socket.AF_INET)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import index  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(index.app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
