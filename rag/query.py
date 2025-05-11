from .cache import load_faiss_index
from .models import text_llm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Replicate


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

def query_all_docs(query: str):
    idx = load_faiss_index()
    if idx is None:
        raise RuntimeError("Index missing")
    retriever = idx.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate(
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

    qa = RetrievalQA.from_chain_type(llm=text_llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt":prompt})
    answer = qa.run(query).strip()
    docs   = retriever.get_relevant_documents(query)
    return answer, format_references([d.metadata for d in docs])
