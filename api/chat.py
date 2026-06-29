"""
Vercel serverless function: POST /api/chat

Receives a JSON body { "message": "...", "k": 5, "force_live": false } and
returns the grounded RAG answer produced by rag.answer(). Implemented with the
stdlib BaseHTTPRequestHandler (no Flask) to keep the serverless bundle small.
"""

from http.server import BaseHTTPRequestHandler
import json

import rag


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            message = (data.get("message") or "").strip()
            if not message:
                return self._send(400, {"error": "empty message"})
            k = int(data.get("k", 5))
            force_live = bool(data.get("force_live", False))
            result = rag.answer(message, k=k, force_live=force_live)
            self._send(200, result)
        except Exception as e:  # never leak a stack trace to the client
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
