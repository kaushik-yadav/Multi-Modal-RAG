import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model

def embed_texts(texts):
    """
    Embed list of texts with proper error handling and normalization
    """
    if not texts:
        return np.array([]).astype('float32')
    
    # Filter out empty texts
    valid_texts = [text for text in texts if text and str(text).strip()]
    if not valid_texts:
        return np.array([]).astype('float32')
    
    try:
        model = get_model()
        embeddings = model.encode(valid_texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        return normalized_embeddings.astype('float32')
    
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Return zero embeddings for error recovery
        return np.zeros((len(texts), 384)).astype('float32')