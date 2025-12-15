# --- Model Runtime Overrides ---
import os

# --- Hugging Face Hub / Mirror Configuration ---
# If you are behind a firewall, set HF_ENDPOINT to a mirror base URL, e.g.:
#   export HF_ENDPOINT="https://hf-mirror.com"
HF_ENDPOINT = (os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_ENDPOINT") or "").strip() or None
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Directory Configuration ---
MARKDOWN_DIR = "markdown_docs"
PARENT_STORE_PATH = "parent_store"
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip() or None
LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest").strip()
LLM_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
