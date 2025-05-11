#!/usr/bin/env python3
# rag_utils.py

import os
import io
import pickle
import base64
from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoProcessor
from ibm_granite_community.notebook_utils import get_env_var
from langchain_community.llms import Replicate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from langchain_core.documents import Document

load_dotenv()  # load REPLICATE_API_TOKEN from .env

# ─── Paths & caches ─────────────────────────────────
SOURCE_DIR  = Path("./source_documents")
CACHE_DIR   = Path("./.cache")
DOC_CACHE   = CACHE_DIR / "docling_docs.pkl"
INDEX_CACHE = CACHE_DIR / "faiss_index"

SOURCE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ─── Model identifiers ───────────────────────────────
EMBED_MODEL   = "ibm-granite/granite-embedding-30m-english"
LLM_MODEL     = "ibm-granite/granite-3.2-8b-instruct"
VISION_MODEL  = "ibm-granite/granite-vision-3.2-2b"

# ─── Docling converter setup ─────────────────────────
pdf_opts   = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
fmt_opts   = { InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts) }
converter  = DocumentConverter(format_options=fmt_opts)

# ─── Embeddings & LLM setup ─────────────────────────
tok       = AutoTokenizer.from_pretrained(EMBED_MODEL)
embedder  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

rep_token = get_env_var("REPLICATE_API_TOKEN")
llm       = Replicate(
    model=LLM_MODEL,
    replicate_api_token=rep_token,
    model_kwargs={"max_tokens":1000, "min_tokens":100},
)

# ─── Cache helpers ───────────────────────────────────
def load_cached_docs():
    return pickle.loads(DOC_CACHE.read_bytes()) if DOC_CACHE.exists() else {}

def save_cached_docs(d):
    DOC_CACHE.write_bytes(pickle.dumps(d))

def load_faiss_index():
    if not INDEX_CACHE.exists():
        return None
    return FAISS.load_local(
        str(INDEX_CACHE),
        embedder,
        allow_dangerous_deserialization=True
    )

def save_faiss_index(idx):
    idx.save_local(str(INDEX_CACHE))

# ─── Convert a single PDF into Docling→LangChain Documents ─────────
def convert_pdf_to_docling(path: Path):
    doc    = converter.convert(source=str(path)).document
    chunks = []

    # 1) Text chunks
    for ch in HybridChunker(tokenizer=tok).chunk(doc):
        items = ch.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue
        page_no = items[0].prov[0].page_no

        chunks.append(Document(
            page_content=ch.text,
            metadata={
                "filename": path.name,
                "page":     page_no,
                "type":     "text",
            }
        ))

    # 2) Tables (enumerated per-PDF)
    for table_idx, tbl in enumerate(doc.tables, start=1):
        page_no = tbl.prov[0].page_no
        md      = tbl.export_to_markdown(doc=doc)
        chunks.append(Document(
            page_content=md,
            metadata={
                "filename":  path.name,
                "page":      page_no,
                "type":      "table",
                "table_num": table_idx,
            }
        ))

    # 3) Images (enumerated per-PDF), include captions
    processor = AutoProcessor.from_pretrained(VISION_MODEL)
    vision_llm = Replicate(
        model=VISION_MODEL,
        replicate_api_token=rep_token,
        model_kwargs={"max_tokens":200}
    )

    def encode_image(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    for image_idx, pic in enumerate(doc.pictures, start=1):
        page_no = pic.prov[0].page_no
        img     = pic.get_image(doc)
        caption = pic.caption_text(doc)

        conversation = [{
            "role": "user",
            "content": [
                {"type":"image"},
                {"type":"text", "text": f"Caption: {caption}"},
                {"type":"text", "text": "Please analyze this image and summarize in 3 sentences."},
            ],
        }]
        vision_prompt = processor.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
        )
        summary = vision_llm.invoke(
            vision_prompt,
            image=encode_image(img),
        )

        chunks.append(Document(
            page_content=summary,
            metadata={
                "filename":   path.name,
                "page":       page_no,
                "type":       "image",
                "image_num":  image_idx,
            }
        ))

    return chunks

# ─── Ingest only new PDFs into FAISS & cache ────────────────────────────
def ingest_new_pdfs(faiss_index, cached_docs):
    all_pdfs = sorted(SOURCE_DIR.glob("*.pdf"))
    to_proc  = [p for p in all_pdfs if p.name not in cached_docs]
    if not to_proc:
        return faiss_index, cached_docs, []

    new_chunks = []
    for p in to_proc:
        docs = convert_pdf_to_docling(p)
        cached_docs[p.name] = docs
        new_chunks.extend(docs)

    texts = [d.page_content for d in new_chunks]
    metas = [d.metadata     for d in new_chunks]

    if faiss_index is None:
        faiss_index = FAISS.from_texts(texts, embedder, metadatas=metas)
    else:
        faiss_index.add_texts(texts, metadatas=metas)

    save_cached_docs(cached_docs)
    save_faiss_index(faiss_index)

    return faiss_index, cached_docs, [p.name for p in to_proc]

# ─── Human-readable Reference formatter ────────────────────────────────
def format_references(metas):
    seen, out = set(), []
    for m in metas:
        parts = [m["filename"], f"p.{m['page']}"]
        if m.get("type") == "table" and m.get("table_num") is not None:
            parts.append(f"Table {m['table_num']}")
        if m.get("type") == "image" and m.get("image_num") is not None:
            parts.append(f"Image {m['image_num']}")
        ref = "(" + ", ".join(parts) + ")"
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out

# ─── Query helper ─────────────────────────────────────────────────────
def query_all_docs(query: str):
    faiss_index = load_faiss_index()
    if faiss_index is None:
        raise RuntimeError("Index not found. Run your ingestion first.")

    retriever = faiss_index.as_retriever(search_kwargs={"k": 4})

    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Keeping the CONTEXT given below in mind, generate a concise summary to answer the QUESTION. Keep the answer brief and to the point.
If you cannot find an answer in the CONTEXT, say "Unknown".
Do not hallucinate or diverge away from the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=False
    )

    answer = qa_chain.run(query).strip()
    docs   = retriever.get_relevant_documents(query)
    refs   = format_references([d.metadata for d in docs])

    return answer, refs
