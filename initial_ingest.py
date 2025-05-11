#!/usr/bin/env python3
# initial_ingest.py

from rag_utils import load_cached_docs, load_faiss_index, ingest_new_pdfs

def main():
    cached = load_cached_docs()
    idx    = load_faiss_index()
    idx, cached, processed = ingest_new_pdfs(idx, cached)
    if processed:
        print(f"✅ Embedded: {', '.join(processed)}")
    else:
        print("⚠️ No new PDFs found—your originals are already ingested.")

if __name__=="__main__":
    main()
