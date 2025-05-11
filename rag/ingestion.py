# rag/ingestion.py
from pathlib import Path
import io
import base64

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem

from .models import tokenizer, vision_processor, vision_llm, embedder
from .cache  import SOURCE_DIR, save_cached_docs, save_faiss_index

# ─── Converter setup ───────────────────────────────────────────────
pdf_opts  = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
fmt_opts  = { InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts) }
converter = DocumentConverter(format_options=fmt_opts)

def convert_pdf_to_docling(path: Path) -> list[Document]:
    """
    Turn one PDF into a list of LangChain Documents:
      - text chunks
      - table chunks (as Markdown)
      - image summaries (via vision_llm)
    """
    doc    = converter.convert(source=str(path)).document
    chunks = []

    # 1) Text chunks
    for ch in HybridChunker(tokenizer=tokenizer).chunk(doc):
        items = ch.meta.doc_items
        # skip pure-table chunks
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

    # 2) Table chunks
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

    # 3) Image chunks (with caption + vision summary)
    def encode_image(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    for image_idx, pic in enumerate(doc.pictures, start=1):
        page_no = pic.prov[0].page_no
        img     = pic.get_image(doc)
        caption = pic.caption_text(doc)

        # build a vision-prompt
        conv = [{
            "role": "user",
            "content": [
                {"type":"image"},
                {"type":"text", "text": f"Caption: {caption}"},
                {"type":"text", "text": "Please analyze this image and summarize in 3 sentences."},
            ],
        }]
        vision_prompt = vision_processor.apply_chat_template(
            conversation=conv,
            add_generation_prompt=True,
        )
        summary = vision_llm.invoke(vision_prompt, image=encode_image(img))

        chunks.append(Document(
            page_content=summary,
            metadata={
                "filename":  path.name,
                "page":      page_no,
                "type":      "image",
                "image_num": image_idx,
            }
        ))

    return chunks


def ingest_new_pdfs(faiss_index, cached_docs):
    """
    Scan SOURCE_DIR for *.pdf, ingest only the new ones into FAISS.
    Returns (updated_index, updated_cache, [new_filenames]).
    """
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
