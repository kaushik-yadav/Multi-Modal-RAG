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
FAISS_INDEX_PATH = "data/faiss_index.index"
FAISS_META_PATH = "data/faiss_metadata.json"
ID_MAP_PATH = "data/id_map.json"

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
DEFAULT_K = 12
MMR_LAMBDA = 0.5

UPLOAD_DIR = "uploads"
THUMBNAIL_DIR = "thumbnails"
DATA_DIR = "data"