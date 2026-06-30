"""
Vercel serverless function: /api/chat

POST { "message": "...", "k": 5, "force_live": false } -> grounded RAG answer.
GET  -> lightweight diagnostic (env var presence + import status), so cold-start
        failures surface as readable JSON instead of an empty 500.

Heavy imports are done lazily inside the handlers and wrapped in try/except, so a
configuration problem returns a clear JSON error rather than crashing the whole
function at module load.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import traceback

# On Vercel the function runs from /var/task, so the api/ directory is not on
# sys.path and `import rag` (a sibling module) fails. Add this file's directory
# explicitly so rag/search/web_search resolve both locally and on Vercel.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REQUIRED_ENV = [
    "GEMINI_API_KEY", "HF_TOKEN", "CHROMA_API_KEY",
    "CHROMA_TENANT", "CHROMA_DATABASE", "TAVILY_API_KEY",
]


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Diagnostic: report config + whether the RAG module imports."""
        env_present = {k: bool(os.environ.get(k)) for k in REQUIRED_ENV}
        import_ok, import_err = True, None
        try:
            import rag  # noqa: F401
        except Exception as e:
            import_ok = False
            import_err = f"{type(e).__name__}: {e}"
        self._send(200, {
            "status": "diagnostic",
            "env_present": env_present,
            "missing_env": [k for k, v in env_present.items() if not v],
            "rag_import_ok": import_ok,
            "rag_import_error": import_err,
        })

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            message = (data.get("message") or "").strip()
            if not message:
                return self._send(400, {"error": "empty message"})

            missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
            if missing:
                return self._send(500, {
                    "error": "Missing environment variables: " + ", ".join(missing)
                })

            import rag  # lazy import so import errors are catchable
            k = int(data.get("k", 5))
            force_live = bool(data.get("force_live", False))
            result = rag.answer(message, k=k, force_live=force_live)
            self._send(200, result)
        except Exception as e:
            self._send(500, {
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:],
            })
