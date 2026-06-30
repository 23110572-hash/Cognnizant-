
from __future__ import annotations

import os
import sys
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import chromadb

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = os.environ.get("CHROMA_COLLECTION", "cognizant_kb")

_hf = InferenceClient(provider="auto", api_key=HF_TOKEN)


def embed(text: str) -> list[float]:
    out = _hf.feature_extraction(text, model=EMBED_MODEL)
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.astype(float).tolist()


def get_collection():
    cc = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    return cc.get_collection(COLLECTION)


def search(query: str, k: int = 4):
    coll = get_collection()
    res = coll.query(query_embeddings=[embed(query)], n_results=k,
                     include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"distance": dist, "meta": meta, "text": doc})
    return hits


def main():
    queries = sys.argv[1:] or [
        "Who is the CEO of Cognizant and what is his background?",
        "What was Cognizant's full-year 2025 revenue?",
        "What is Cognizant's AI builder strategy and the Synapse skilling goal?",
        "Which companies does Cognizant partner with for AI?",
        "Where are Cognizant's largest delivery centers located?",
    ]
    for q in queries:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        for i, h in enumerate(search(q, k=3), 1):
            m = h["meta"]
            src = m.get("section_path", "")
            print(f"[{i}] dist={h['distance']:.3f} | {m['doc_id']} | {src}")
            snippet = h["text"].split("\n", 1)[-1].strip()[:240].replace("\n", " ")
            print(f"     {snippet}")


if __name__ == "__main__":
    main()
