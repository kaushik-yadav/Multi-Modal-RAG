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

# Embedding Model
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval Parameters
DEFAULT_FETCH_K = 75
DEFAULT_K = 20
MMR_LAMBDA = 0.5