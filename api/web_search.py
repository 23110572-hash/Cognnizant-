"""
Cognizant RAG — live web search (Tavily).

Used for real-time / volatile facts that should NOT be answered from the static
Chroma knowledge base, e.g. the current CTSH stock price, today's news, or any
question the user marks as "latest / current / now".

Tavily returns LLM-ready, ranked, pre-extracted content, so results can be fed
straight into a prompt as grounding context.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Lightweight router: trigger live web search when a query is time-sensitive.
LIVE_TRIGGERS = (
    "current", "today", "now", "latest", "live", "right now", "this week",
    "stock price", "share price", "trading at", "market cap", "as of today",
    "recent", "breaking", "this morning", "yesterday",
)


def needs_live_web(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in LIVE_TRIGGERS)


def search_web(query: str, max_results: int = 5, scope_cognizant: bool = True) -> dict:
    """Run a Tavily search and return a compact, citation-ready result dict."""
    q = query
    if scope_cognizant and "cognizant" not in q.lower() and "ctsh" not in q.lower():
        q = f"Cognizant (CTSH) {query}"
    resp = _client.search(
        query=q,
        max_results=max_results,
        search_depth="basic",
        include_answer=True,
    )
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score"),
        }
        for r in resp.get("results", [])
    ]
    return {
        "query": q,
        "answer": resp.get("answer", ""),
        "results": results,
    }


def format_context(web: dict) -> str:
    """Render web results as grounding context with inline source citations."""
    lines = []
    if web.get("answer"):
        lines.append(f"Summary: {web['answer']}")
    for i, r in enumerate(web["results"], 1):
        lines.append(f"[{i}] {r['title']} ({r['url']})\n{r['content']}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "current CTSH stock price"
    print(f"needs_live_web({query!r}) -> {needs_live_web(query)}")
    print("=" * 70)
    web = search_web(query, max_results=4)
    print(format_context(web))
