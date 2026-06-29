from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import chromadb

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "chunks" / "chunks.jsonl"
CACHE_PATH = ROOT / "chunks" / "embeddings_cache.jsonl"

HF_TOKEN = os.environ["HF_TOKEN"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "cognizant_kb")

MAX_WORKERS = 8
ADD_BATCH = 200
MAX_RETRIES = 4

_hf = InferenceClient(provider="auto", api_key=HF_TOKEN)


# --- Embedding ---------------------------------------------------------------

def embed_one(text: str) -> list[float]:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            out = _hf.feature_extraction(text, model=EMBED_MODEL)
            arr = np.asarray(out, dtype=float)
            if arr.ndim == 2:          # token-level -> mean pool
                arr = arr.mean(axis=0)
            return arr.astype(float).tolist()
        except Exception as e:         # 429/503/model-loading/transient
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"embedding failed after {MAX_RETRIES} retries: {last_err}")


# --- IO helpers --------------------------------------------------------------

def load_chunks(limit: int | None) -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks[:limit] if limit else chunks


def load_cache() -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["chunk_id"]] = rec["embedding"]
    return cache


def append_cache(records: list[dict]) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def sanitize_meta(c: dict) -> dict:
    meta = {
        "doc_id": c["doc_id"],
        "source_path": c["source_path"],
        "source_type": c["source_type"],
        "category": c["category"],
        "doc_title": c["doc_title"],
        "section_path": c["section_path"],
        "chunk_index": int(c["chunk_index"]),
        "token_count": int(c["token_count"]),
    }
    if c.get("page_start") is not None:
        meta["page_start"] = int(c["page_start"])
    if c.get("page_end") is not None:
        meta["page_end"] = int(c["page_end"])
    return meta


# --- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process first N chunks")
    ap.add_argument("--recreate", action="store_true", help="drop & recreate the collection")
    args = ap.parse_args()

    chunks = load_chunks(args.limit)
    print(f"Loaded {len(chunks)} chunks. Embedding with {EMBED_MODEL}.")

    cache = load_cache()
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    print(f"  cached: {len(chunks) - len(todo)} | to embed: {len(todo)}")

    # Embed missing ones concurrently, persisting as we go.
    done = 0
    pending: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(embed_one, c["text"]): c for c in todo}
        for fut in as_completed(futures):
            c = futures[fut]
            emb = fut.result()
            cache[c["chunk_id"]] = emb
            pending.append({"chunk_id": c["chunk_id"], "embedding": emb})
            done += 1
            if len(pending) >= 50:
                append_cache(pending)
                pending = []
            if done % 100 == 0 or done == len(todo):
                print(f"  embedded {done}/{len(todo)}")
    if pending:
        append_cache(pending)

    # Connect to Chroma Cloud
    cc = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    if args.recreate:
        try:
            cc.delete_collection(COLLECTION)
            print(f"Dropped existing collection '{COLLECTION}'.")
        except Exception:
            pass
    coll = cc.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL},
    )

    # Upsert in batches
    print(f"Upserting {len(chunks)} chunks into Chroma collection '{COLLECTION}'...")
    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i:i + ADD_BATCH]
        coll.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[cache[c["chunk_id"]] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[sanitize_meta(c) for c in batch],
        )
        print(f"  upserted {min(i + ADD_BATCH, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection now holds {coll.count()} vectors.")


if __name__ == "__main__":
    main()
