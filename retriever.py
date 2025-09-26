import logging
from typing import Any, Dict, List

import numpy as np

from constants import DEFAULT_FETCH_K, DEFAULT_K, MMR_LAMBDA
from embedding import embed_texts
from indexer import load_index_and_meta

logger = logging.getLogger(__name__)

def mmr_retrieve(query: str, k: int = DEFAULT_K, fetch_k: int = DEFAULT_FETCH_K) -> List[Dict[str, Any]]:
    """
    Improved MMR retrieval with proper error handling and vector operations
    """
    try:
        index, items = load_index_and_meta()
        if index is None or not items:
            logger.warning("No index or items available")
            return []
        
        # Validate index state
        if index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Embed query
        query_embedding = embed_texts([query])
        if len(query_embedding) == 0:
            logger.error("Query embedding failed")
            return []
        
        query_embedding = query_embedding[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Adjust fetch_k if necessary
        fetch_k = min(fetch_k, index.ntotal)
        
        # Initial search
        scores, indices = index.search(query_embedding, fetch_k)
        scores = scores[0]  # Shape (fetch_k,)
        indices = indices[0]  # Shape (fetch_k,)
        
        # Get candidate items and their embeddings
        candidate_items = []
        valid_indices = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(items):
                candidate_items.append(items[idx])
                valid_indices.append(idx)
            else:
                logger.warning(f"Invalid index {idx} found in search results")
        
        if not candidate_items:
            return []
        
        # Re-embed candidate items for accurate similarity calculation
        candidate_texts = [item.get("content", "") for item in candidate_items]
        candidate_embeddings = embed_texts(candidate_texts).astype('float32')
        
        # MMR selection
        selected_indices = []
        selected_embeddings = []
        
        # Start with the most relevant document
        first_idx = np.argmax(scores[:len(valid_indices)])
        selected_indices.append(first_idx)
        selected_embeddings.append(candidate_embeddings[first_idx])
        
        # Remove selected from candidates
        remaining_indices = [i for i in range(len(candidate_items)) if i != first_idx]
        remaining_embeddings = np.delete(candidate_embeddings, first_idx, axis=0)
        remaining_scores = np.delete(scores[:len(valid_indices)], first_idx)
        
        # Select remaining items using MMR
        while len(selected_indices) < min(k, len(candidate_items)) and remaining_indices:
            # Calculate similarity to query
            sim_to_query = remaining_scores
            
            # Calculate max similarity to selected documents
            if selected_embeddings:
                # Normalize embeddings for cosine similarity
                norms_remaining = np.linalg.norm(remaining_embeddings, axis=1, keepdims=True)
                norms_selected = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
                
                # Compute cosine similarities
                sim_to_selected = []
                for i in range(len(remaining_embeddings)):
                    max_sim = 0
                    for sel_emb in selected_embeddings:
                        # Ensure shapes are compatible
                        emb1 = remaining_embeddings[i].reshape(1, -1)
                        emb2 = sel_emb.reshape(1, -1)
                        sim = np.dot(emb1, emb2.T)[0][0]
                        max_sim = max(max_sim, sim)
                    sim_to_selected.append(max_sim)
                
                sim_to_selected = np.array(sim_to_selected)
            else:
                sim_to_selected = np.zeros(len(remaining_indices))
            
            # MMR scoring
            mmr_scores = MMR_LAMBDA * sim_to_query - (1 - MMR_LAMBDA) * sim_to_selected
            
            # Select document with highest MMR score
            next_idx = np.argmax(mmr_scores)
            selected_idx = remaining_indices[next_idx]
            
            selected_indices.append(selected_idx)
            selected_embeddings.append(remaining_embeddings[next_idx])
            
            # Remove selected from remaining
            remaining_indices.pop(next_idx)
            remaining_embeddings = np.delete(remaining_embeddings, next_idx, axis=0)
            remaining_scores = np.delete(remaining_scores, next_idx)
        
        # Return selected items
        results = []
        for idx in selected_indices:
            if idx < len(candidate_items):
                item = candidate_items[idx]
                # Add score information
                item['retrieval_score'] = float(scores[idx] if idx < len(scores) else 0.0)
                results.append(item)
        
        logger.info(f"MMR retrieval returned {len(results)} items")
        return results
        
    except Exception as e:
        logger.error(f"MMR retrieval failed: {e}")
        # Fallback to simple retrieval
        return simple_retrieve(query, k)

def simple_retrieve(query: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """Simple retrieval as fallback"""
    try:
        index, items = load_index_and_meta()
        if index is None or not items:
            return []
        
        query_embedding = embed_texts([query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        k = 14
        scores, indices = index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(items):
                item = items[idx].copy()
                item['retrieval_score'] = float(score)
                results.append(item)
        
        return results
        
    except Exception as e:
        logger.error(f"Simple retrieval also failed: {e}")
        return []