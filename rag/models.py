# rag/models.py
from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoProcessor
from ibm_granite_community.notebook_utils import get_env_var
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Replicate

# ─── Load credentials ───────────────────────────────────────
load_dotenv()
REPLICATE_TOKEN = get_env_var("REPLICATE_API_TOKEN")

# ─── Model identifiers ─────────────────────────────────────
EMBED_MODEL  = "ibm-granite/granite-embedding-30m-english"
LLM_MODEL    = "ibm-granite/granite-3.2-8b-instruct"
VISION_MODEL = "ibm-granite/granite-vision-3.2-2b"

# ─── Embeddings & Tokenizer ─────────────────────────────────
embedder  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

# ─── LLM for text QA ────────────────────────────────────────
text_llm = Replicate(
    model=LLM_MODEL,
    replicate_api_token=REPLICATE_TOKEN,
    model_kwargs={"max_tokens": 1000, "min_tokens": 100},
)

# ─── Processor & LLM for images ────────────────────────────
vision_processor = AutoProcessor.from_pretrained(VISION_MODEL)
vision_llm = Replicate(
    model=VISION_MODEL,
    replicate_api_token=REPLICATE_TOKEN,
    model_kwargs={"max_tokens": 200},
)
