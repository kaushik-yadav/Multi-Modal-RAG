# constants.py
import os

from dotenv import load_dotenv

load_dotenv()


# OpenRouter config
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_MODEL = "qwen/qwen-2.5-vl-7b-instruct"
OPENROUTER_MAX_TOKENS = 500
OPENROUTER_TEMPERATURE = 0.1
ASEEMBLY_API_KEY=os.getenv('ASSEMBLYAI_API_KEY')
# FAISS Configuration
# Default paths (fallback if no session paths are set)
DEFAULT_FAISS_INDEX_PATH = "data/faiss_index.index"
DEFAULT_FAISS_META_PATH = "data/faiss_metadata.json"
DEFAULT_ID_MAP_PATH = "data/id_map.json"

# Get paths from environment variables (set by session)
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH)
FAISS_META_PATH = os.environ.get("FAISS_META_PATH", DEFAULT_FAISS_META_PATH)
ID_MAP_PATH = os.environ.get("ID_MAP_PATH", DEFAULT_ID_MAP_PATH)

# Embedding Model
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# File Processing
SUPPORTED_EXTENSIONS = {
    'document': ['.pdf', '.doc', '.docx', '.txt'],
    'image': ['.png', '.jpg', '.jpeg', '.webp', '.bmp'],
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
}

# Retrieval Parameters
DEFAULT_FETCH_K = 75
DEFAULT_K = 20
MMR_LAMBDA = 0.5

UPLOAD_DIR = "uploads"
THUMBNAIL_DIR = "thumbnails"
DATA_DIR = "data"