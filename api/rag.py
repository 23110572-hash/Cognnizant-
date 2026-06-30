from __future__ import annotations

import os
import re
import sys
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import search
import web_search

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

TOP_K = 5

SYSTEM_INSTRUCTION = """You are the Cognizant Assistant. You answer questions \
about Cognizant Technology Solutions ( CTSH).

How to answer:
- Be SMART and TO THE POINT. Lead with the direct answer in the first sentence. \
Keep it short — usually 1-4 sentences, or 3-5 short bullets for lists. No filler.
- Use the provided CONTEXT as the source . Combine information from multiple chunks when needed.
- If the answer isn't in the context, politely say you don't have that detail and \
invite another Cognizant question. Never guess.
- For real-time facts (current stock price, today's news), use the live results \
and give the figure, noting prices change continuously.
- NEVER mention documents, sources, knowledge bases, files, pages, or that you \
are using any context. NEVER output bracketed tags like [KB1] or [WEB2]. Just \
answer naturally, as if you simply know it.
"""

# LangChain LCEL chain: prompt -> Groq chat model -> plain string.
# Built once at import time and reused for every question.
_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.2,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "QUESTION: {question}\n\n"
     "Answer using the context above, with inline citations."),
])

_chain = _prompt | _llm | StrOutputParser()

# --- Query expansion (for retrieval recall on broad questions) ---------------
# Broad/overview questions ("what are the main services?") embed poorly against a
# KB of narrow, single-topic pages, so we expand into a few diverse search
# queries. Retrieval for all variants is then done in ONE parallel/batched pass
# (see search.search_many) to keep this fast.
_expand_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite a user's question into diverse search queries that maximize "
     "retrieval recall over a Cognizant knowledge base (services, industries, "
     "financials, leadership, strategy). Output 3 short, varied search queries, "
     "one per line, no numbering, no extra text. Cover different facets of the "
     "question (e.g. for 'main services' include cloud, data and AI, consulting, "
     "cybersecurity, application services, business process services)."),
    ("human", "{question}"),
])
_expand_chain = _expand_prompt | _llm | StrOutputParser()


# --- Reference-tag cleanup ---------------------------------------------------
_TAG_RE = re.compile(r"\s*\[(?:KB|WEB)\s*\d+[^\]]*\]")


def _clean_answer(text: str) -> str:
    """Safety net: strip any leaked [KB#]/[WEB#] reference tags from the answer."""
    text = _TAG_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _expand_queries(question: str) -> list[str]:
    """Return the original question plus a few LLM-generated variations (best-effort)."""
    queries = [question]
    try:
        raw = _expand_chain.invoke({"question": question})
        for line in raw.splitlines():
            q = line.strip().lstrip("-•*0123456789. ").strip()
            if q and q.lower() != question.lower():
                queries.append(q)
    except Exception:
        pass  # expansion is best-effort; fall back to the raw question
    return queries[:4]


def _retrieve_kb(question: str, k: int) -> list[dict]:
    """Multi-query retrieval done in a single fast pass.

    Expand the question, then embed all variants in parallel and run ONE batched
    Chroma query (search.search_many). Fuse by best distance and apply a
    per-document diversity pass so breadth questions surface several pages.
    """
    queries = _expand_queries(question)
    hits = search.search_many(queries, k=k)

    best: dict[tuple, dict] = {}
    for h in hits:
        m = h["meta"]
        key = (m.get("doc_id"), m.get("chunk_index"))
        if key not in best or h["distance"] < best[key]["distance"]:
            best[key] = h
    ranked = sorted(best.values(), key=lambda h: h["distance"])

    # Diversity: take the best chunk from each distinct doc first, then backfill.
    seen_docs: set = set()
    primary: list[dict] = []
    extra: list[dict] = []
    for h in ranked:
        doc_id = h["meta"].get("doc_id")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            primary.append(h)
        else:
            extra.append(h)
    return (primary + extra)[:k]


def build_context(question: str, k: int = TOP_K, force_live: bool = False):
    """Retrieve from Chroma (+ Tavily if live) and return (context, sources, used_live)."""
    sources: list[dict] = []
    blocks: list[str] = []

    # 1) Knowledge base (fast batched multi-query retrieval + diversity pass)
    kb_hits = _retrieve_kb(question, k=k)
    if kb_hits:
        kb_lines = ["### KNOWLEDGE BASE EXCERPTS"]
        for i, h in enumerate(kb_hits, 1):
            m = h["meta"]
            label = f"{m['doc_title']} - {m.get('section_path', '')}".strip(" -")
            body = h["text"].split("\n", 1)[-1].strip()
            kb_lines.append(f"[KB{i}] ({label})\n{body}")
            sources.append({
                "ref": f"KB{i}", "type": "knowledge_base",
                "label": label, "source_path": m.get("source_path"),
                "distance": round(h["distance"], 3),
            })
        blocks.append("\n\n".join(kb_lines))

    # 2) Live web (when time-sensitive or explicitly forced)
    used_live = False
    if force_live or web_search.needs_live_web(question):
        used_live = True
        web = web_search.search_web(question, max_results=5)
        web_lines = ["### LIVE WEB RESULTS (real-time)"]
        if web.get("answer"):
            web_lines.append(f"Summary: {web['answer']}")
        for i, r in enumerate(web["results"], 1):
            web_lines.append(f"[WEB{i}] {r['title']} ({r['url']})\n{r['content'][:800]}")
            sources.append({
                "ref": f"WEB{i}", "type": "live_web",
                "label": r["title"], "url": r["url"],
            })
        blocks.append("\n\n".join(web_lines))

    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return context, sources, used_live


def answer(question: str, k: int = TOP_K, force_live: bool = False) -> dict:
    context, sources, used_live = build_context(question, k=k, force_live=force_live)
    raw = _chain.invoke({"context": context, "question": question})
    return {
        "question": question,
        "answer": _clean_answer(raw or ""),
        "used_live_web": used_live,
        "sources": sources,
    }


def _print(result: dict) -> None:
    print("\n" + "-" * 72)
    print(result["answer"].strip())
    print("-" * 72)
    tag = "knowledge base + live web" if result["used_live_web"] else "knowledge base"
    print(f"(grounded on: {tag})")
    print("Sources:")
    for s in result["sources"]:
        if s["type"] == "knowledge_base":
            print(f"  [{s['ref']}] {s['label']}  ({s['source_path']}, dist={s['distance']})")
        else:
            print(f"  [{s['ref']}] {s['label']}  {s['url']}")


def main() -> None:
    if len(sys.argv) > 1:
        _print(answer(" ".join(sys.argv[1:])))
        return
    print("Cognizant Knowledge Assistant. Ask a question (Ctrl+C or 'quit' to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q or q.lower() in {"quit", "exit"}:
            print("Bye.")
            break
        try:
            _print(answer(q))
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
