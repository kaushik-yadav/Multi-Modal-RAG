from utils.constants import GROQ_API_KEY
from processors.image_processor import get_image_caption, init_groq

# Initialize Groq processor
init_groq(GROQ_API_KEY)