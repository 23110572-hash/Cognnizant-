# Cognizant RAG — Pipeline (Stages 2 & 3)

## Stage 2: Chunking & QA
```bash
pip install -r requirements.txt
python src/chunking.py        # -> chunks/chunks.jsonl + chunks/manifest.json
python src/verify_chunks.py   # -> chunks/qa_report.json (+ console report)
```

### Inputs
- `data/web/*.md`   — cleaned website content (header-aware chunking)
- `data/pdfs/*.pdf` — investor filings (sentence-aware, page-tracked chunking)
- `data/pdfs_excluded/` — two oversized/unparseable sustainability PDFs (see its README)

### Chunk schema (`chunks/chunks.jsonl`, one JSON object per line)
`chunk_id, doc_id, source_path, source_type, category, doc_title, section_path,
chunk_index, token_count, char_count, page_start, page_end, text`.
Each `text` is prefixed with `[doc_title | section]` so chunks are self-describing.

### Strategy
- Markdown: split on the doc's own H1/H2/H3; one chunk per section unless it
  exceeds the budget, then sentence-split with overlap.
- PDF: per-page extraction, accumulate to budget, sentence-aware split with
  overlap, page ranges recorded for citation.
- Target <=512 tokens, 64-token overlap.

### QA checks
Structural (hard-fail gates), size distribution, exact-duplicate detection,
source coverage, and embedding-free coherence (mid-clause cut rate, intra-doc
similarity, orphan detection).

### Result
1,791 chunks (30 web docs + 7 PDFs). QA PASSED: 0 structural failures, 0 dup ids,
0 empty; mean max-sibling similarity ~0.33; 3 benign lexical orphans;
~9.6% mid-clause cuts (concentrated in dense financial tables).

---

## Stage 3: Embed & store in Chroma Cloud
```bash
python src/test_connectivity.py          # verify HF + Chroma creds
python src/embed_and_store.py            # embed all chunks + upsert to Chroma
python src/embed_and_store.py --recreate # drop & rebuild the collection
python src/search.py "your question"     # retrieval sanity check
```

### Secrets
Credentials live in `.env` (gitignored), read via python-dotenv:
`HF_TOKEN, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE, EMBED_MODEL,
CHROMA_COLLECTION`. NOTE: keys shared in plaintext should be rotated.

### Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2` via HF Inference API (384-dim).
- Cached to `chunks/embeddings_cache.jsonl` (incremental, resumable).
- 8-worker concurrency with retry/backoff for transient 429/503.

### Storage
- Chroma Cloud collection `cognizant_kb`, cosine space.
- Each record: id, 384-d embedding, document text, sanitized metadata.
  Upserted in batches of 200.

### Result
1,791 vectors stored. Retrieval validated on sample queries — relevant,
correctly-sourced chunks with cosine distances ~0.25–0.33.

---

## Next stage (not yet built)
Retrieval-augmented generation: wrap `search.py` retrieval with an LLM to
generate grounded, cited answers; add the live web/Tavily path for real-time
facts (e.g., current stock price).

---

# Stage 4: Live web search (Tavily)
`src/web_search.py`
- `needs_live_web(q)` — routes time-sensitive queries (current/latest/stock price/...)
- `search_web(q)` — Tavily advanced search, auto-scoped to Cognizant/CTSH
- `format_context(web)` — citation-ready rendering with source URLs
```bash
python src/web_search.py "current CTSH stock price"
```

# Stage 5: RAG chatbot (Gemini)
`src/rag.py` — ties everything together.
- Always retrieves top-k from Chroma (static facts).
- Adds Tavily live results when the query is time-sensitive.
- Generates a grounded, cited answer with Gemini (`gemini-2.5-flash`),
  using a system prompt that forbids ungrounded claims and requires citations
  ([doc - section] for KB, URL for web).
```bash
python src/check_gemini.py                 # list models available to the key
python src/rag.py "Who is Cognizant's CEO?"   # one-shot
python src/rag.py                              # interactive chat
```

Model note: `gemini-2.5-flash` works on the provided key; `gemini-2.0-flash`
returned 429 (separate quota bucket); `gemini-1.5-flash` is 404 (unavailable).

## Full pipeline summary
1. data collection -> data/web + data/pdfs
2. chunking + QA   -> chunks/ (1,791 chunks)
3. embed + store   -> Chroma Cloud `cognizant_kb` (384-d, 1,791 vectors)
4. live web        -> Tavily
5. RAG answer      -> Gemini, grounded + cited, KB for static / web for real-time
