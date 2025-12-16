# --- Model Runtime Overrides ---
import os
from pathlib import Path

# --- Hugging Face Hub / Mirror Configuration ---
# If you are behind a firewall, set HF_ENDPOINT to a mirror base URL, e.g.:
#   export HF_ENDPOINT="https://hf-mirror.com"
HF_ENDPOINT = (os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_ENDPOINT") or "").strip() or None
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Directory Configuration ---
_REPO_ROOT = Path(__file__).resolve().parent.parent

def _resolve_path(env_var: str, default_rel: str) -> str:
    raw = (os.getenv(env_var) or "").strip()
    value = raw or default_rel
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path.resolve())

MARKDOWN_DIR = _resolve_path("MARKDOWN_DIR", "markdown_docs")
PARENT_STORE_PATH = _resolve_path("PARENT_STORE_PATH", "parent_store")
QDRANT_DB_PATH = _resolve_path("QDRANT_DB_PATH", "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"

# --- DeepSeek API Configuration ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

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
