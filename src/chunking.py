from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# --- Configuration -----------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "data" / "web"
OUT_DIR = ROOT / "chunks"

MAX_TOKENS = 512      # target ceiling per chunk
OVERLAP_TOKENS = 64   # overlap between sub-chunks of a long section

ENC = tiktoken.get_encoding("cl100k_base")

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Stage 1: split markdown by headers, keeping header metadata. Headers are kept
# out of the body (we re-attach a clean prefix ourselves below).
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=True,
)

# Stage 2: token-aware recursive splitter for oversized sections. Separators are
# ordered so it prefers paragraph -> line -> sentence -> word boundaries.
_token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=MAX_TOKENS,
    chunk_overlap=OVERLAP_TOKENS,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
)


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# --- Data model --------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    source_type: str          # always "web" now (markdown only)
    category: str
    doc_title: str
    section_path: str         # e.g. "About Cognizant > What we do"
    chunk_index: int
    token_count: int
    char_count: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


# --- Helpers -----------------------------------------------------------------

def categorize(doc_id: str) -> str:
    name = doc_id.lower()
    if name.startswith("industry-"):
        return "industry"
    if name.startswith("services-"):
        return "services"
    if name.startswith("cognizant-neuro") or name.startswith("cognizant-agentic"):
        return "ai-platforms"
    if name in ("board-of-directors", "leadership-team", "executive-bios-and-leadership-thinking"):
        return "leadership"
    if name in ("press-q4-fy2025-results", "recent-news-2026"):
        return "news"
    if name in ("stock-ownership-and-share-info", "strategy-and-future-goals"):
        return "financials-strategy"
    if name in ("partnerships",):
        return "partnerships"
    if name in ("talent-hr-and-careers",):
        return "talent-hr"
    if name in ("locations-and-operations", "global-presence-by-region"):
        return "locations"
    if name in ("top-clients-and-case-studies",):
        return "clients"
    if name in ("responsible-ai",):
        return "responsible-ai"
    if name in ("culture-and-values", "about-cognizant", "company-history-and-overview",
                "cognizant-company-profile-and-recognition"):
        return "company-overview"
    if name == "thought-leadership-ai-future-of-work":
        return "thought-leadership"
    return "other"


def make_chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{doc_id}:{idx}:{text[:64]}".encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}__{idx:03d}__{h}"


def _section_path(meta: dict) -> tuple[str, str]:
    """Return (doc_title, section_path) from header metadata."""
    h1 = meta.get("h1", "").strip()
    parts = [meta.get(k, "").strip() for k in ("h1", "h2", "h3")]
    parts = [p for p in parts if p]
    doc_title = h1
    section_path = " > ".join(parts) if parts else "(root)"
    return doc_title, section_path


# --- Markdown chunking -------------------------------------------------------

def chunk_markdown_file(path: Path) -> list[Chunk]:
    doc_id = path.stem
    category = categorize(doc_id)
    raw = path.read_text(encoding="utf-8")

    # Stage 1: header-aware split.
    sections = _md_splitter.split_text(raw)

    # Fallback doc title from filename if the file has no H1.
    fallback_title = doc_id.replace("-", " ").title()

    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        body = (sec.page_content or "").strip()
        if not body:
            continue
        doc_title, section_path = _section_path(sec.metadata)
        if not doc_title:
            doc_title = fallback_title

        header_prefix = (
            f"[{doc_title} | {section_path}]\n"
            if section_path != "(root)" else f"[{doc_title}]\n"
        )

        # Stage 2: only split further if the section exceeds the token budget.
        budget = MAX_TOKENS - n_tokens(header_prefix)
        if n_tokens(body) <= budget:
            pieces = [body]
        else:
            pieces = _token_splitter.split_text(body)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            text = header_prefix + piece
            chunks.append(Chunk(
                chunk_id=make_chunk_id(doc_id, idx, text),
                doc_id=doc_id,
                source_path=str(path.relative_to(ROOT)).replace("\\", "/"),
                source_type="web",
                category=category,
                doc_title=doc_title,
                section_path=section_path,
                chunk_index=idx,
                token_count=n_tokens(text),
                char_count=len(text),
                text=text,
            ))
            idx += 1
    return chunks


# --- Driver ------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    all_chunks: list[Chunk] = []

    web_files = sorted(WEB_DIR.glob("*.md"))
    print(f"Found {len(web_files)} markdown files.")

    for f in web_files:
        cs = chunk_markdown_file(f)
        all_chunks.extend(cs)
        print(f"  [web] {f.name}: {len(cs)} chunks")

    # Write chunks.jsonl
    out_path = OUT_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # Write manifest
    manifest: dict = {}
    for c in all_chunks:
        m = manifest.setdefault(c.doc_id, {
            "source_type": c.source_type,
            "category": c.category,
            "doc_title": c.doc_title,
            "chunk_count": 0,
            "total_tokens": 0,
        })
        m["chunk_count"] += 1
        m["total_tokens"] += c.token_count
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote {len(all_chunks)} chunks -> {out_path}")
    print(f"Wrote manifest -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
