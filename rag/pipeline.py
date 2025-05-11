from pathlib import Path
from .cache import load_cached_docs, load_faiss_index
from .ingestion import ingest_new_pdfs
from .query import query_all_docs

class RAGPipeline:
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.source_dir.mkdir(exist_ok=True)
        self.cached_docs = load_cached_docs()
        self.index       = load_faiss_index()

    def uploaded_pdf(self, pdf_path: Path) -> str:
        if not pdf_path.exists():
            return f"❗ Not found: {pdf_path.name}"
        dest = self.source_dir / pdf_path.name
        if not dest.exists():
            dest.write_bytes(pdf_path.read_bytes())
        self.index, self.cached_docs, new = ingest_new_pdfs(self.index, self.cached_docs)
        return f"✅ Embedded: {', '.join(new)}" if new else f"ℹ️ Already: {pdf_path.name}"

    def query(self, query: str) -> str:
        if not query.strip():
            return "❗ Please enter a question."
        ans, refs = query_all_docs(query)
        return ans + "\n\nReferences:\n" + "\n".join(refs)
