from constants import GROQ_API_KEY
from rag_system.groq_processor import get_image_caption, init_groq

# Initialize Groq processor
init_groq(GROQ_API_KEY)

# The get_image_caption function is now imported from groq_processor
# This file maintains compatibility with existing code