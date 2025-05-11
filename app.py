# # 

# #!/usr/bin/env python3
# # app.py

# import uuid
# from pathlib import Path

# import gradio as gr

# from rag_utils import (
#     SOURCE_DIR,
#     load_cached_docs,
#     load_faiss_index,
#     ingest_new_pdfs,
#     query_all_docs,
# )

# # ─── INITIAL LOAD ───────────────────────────────────────────────────────────────
# cached_docs = load_cached_docs()
# faiss_index = load_faiss_index()
# SOURCE_DIR.mkdir(exist_ok=True)

# def gradio_rag(uploaded, query: str) -> str:
#     # 1) Persist any uploaded PDFs (each uf is a str path)
#     for uf in uploaded or []:
#         temp_path = Path(uf)
#         filename  = temp_path.name
#         data      = temp_path.read_bytes()
#         dest = SOURCE_DIR / filename
#         if not dest.exists():
#             dest.write_bytes(data)

#     # 2) Ingest only the new PDFs
#     global faiss_index, cached_docs
#     faiss_index, cached_docs, new_files = ingest_new_pdfs(faiss_index, cached_docs)

#     # 3) Run the RAG query over all docs
#     answer, refs = query_all_docs(query)

#     # 4) Assemble the output
#     msg = ""
#     if new_files:
#         msg += f"✅ Embedded: {', '.join(new_files)}\n\n"
#     else:
#         msg += f"This PDF is already chunked"
#     msg += answer + "\n\nReferences:\n" + "\n".join(refs)
#     return msg

# if __name__ == "__main__":
#     with gr.Blocks() as demo:
#         gr.Markdown("## 📚 Multimodal Multi-Document RAG")
#         uploader = gr.File(
#             label="Upload new PDFs",
#             file_count="multiple",      # allow any number of uploads
#             file_types=[".pdf"],        # restrict to PDFs
#         )
#         query_in = gr.Textbox(
#             label="Your question",
#             lines=2,
#             placeholder="Ask anything…"
#         )
#         output = gr.Textbox(
#             label="Answer & References",
#             lines=15
#         )
#         run_btn = gr.Button("Run")
#         run_btn.click(
#             fn=gradio_rag,
#             inputs=[uploader, query_in],
#             outputs=output
#         )
#     demo.launch()

#!/usr/bin/env python3
# app.py

from pathlib import Path
import gradio as gr

from rag_utils import (
    SOURCE_DIR,
    load_cached_docs,
    load_faiss_index,
    ingest_new_pdfs,
    query_all_docs,
)

# ─── INITIAL SETUP ───────────────────────────────────────────────────────────────
# Load on-disk cache/index into memory
cached_docs = load_cached_docs()
faiss_index = load_faiss_index()
# Ensure our PDF folder exists
SOURCE_DIR.mkdir(exist_ok=True)

# ─── UPLOAD HANDLER ──────────────────────────────────────────────────────────────
def upload_pdf(pdf_path: str) -> str:
    """
    1) Copies the uploaded PDF (from its temp-path string) into SOURCE_DIR
    2) Calls ingest_new_pdfs, which only processes truly new files
    3) Returns either Embedded or Already embedded
    """
    if not pdf_path:
        return "❗ No PDF provided."
    src = Path(pdf_path)
    filename = src.name
    dest = SOURCE_DIR / filename

    # Persist file if it wasn't there already
    if not dest.exists():
        dest.write_bytes(src.read_bytes())

    # Ingest only new docs
    global faiss_index, cached_docs
    faiss_index, cached_docs, new_files = ingest_new_pdfs(faiss_index, cached_docs) # using global in the above line ensures that when ingest_new_pdfs returns these variables. they are copied into the same ones in the line above

    if new_files:
        return f"✅ Embedded: {', '.join(new_files)}"
    else:
        return f"ℹ️ Already embedded: {filename}"

# ─── QUERY HANDLER ────────────────────────────────────────────────────────────────
def get_answers(query: str) -> str:
    """
    Runs RetrievalQA over all ingested PDFs and returns:
      <answer>

      References:
      (File1.pdf, p.3)
      (File2.pdf, p.7, Figure 1)
    """
    if not query or not query.strip():
        return "❗ Please enter a question."
    answer, refs = query_all_docs(query)
    return answer + "\n\nReferences:\n" + "\n".join(refs)

# ─── GRADIO UI LAYOUT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with gr.Blocks() as demo:
        # Section 1: PDF upload
        gr.Markdown("### 📥 Upload New PDF")
        with gr.Row():
            pdf_uploader = gr.File(
                label="Select a PDF to ingest",
                file_count="single",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Upload New PDF")
        pdf_status = gr.Textbox(label="Upload Status", lines=1)
        upload_btn.click(fn=upload_pdf, inputs=pdf_uploader, outputs=pdf_status)

        gr.Markdown("---")

        # Section 2: Ask questions
        gr.Markdown("### ❓ Ask a Question")
        query_in  = gr.Textbox(label="Your question", lines=2, placeholder="Type here…")
        ask_btn   = gr.Button("Get Answers")
        answer_out = gr.Textbox(label="Answer & References", lines=10)
        ask_btn.click(fn=get_answers, inputs=query_in, outputs=answer_out)

    demo.launch()
