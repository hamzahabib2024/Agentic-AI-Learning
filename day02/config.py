
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

# ChromaDB collection
COLLECTION_NAME = "pdf_documents"

# Ollama models
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "qwen3:1.7b"

# Document chunking
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30

# Retrieval
TOP_K = 4

# Batch processing
BATCH_SIZE = 32
