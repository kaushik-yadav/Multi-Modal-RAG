import json
import logging
from typing import Any, Dict, List

import requests

from constants import (
    OPENROUTER_API_KEY,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_URL,
)
from retriever import mmr_retrieve

logger = logging.getLogger(__name__)

def format_cross_modal_context(retrieved_items: List[Dict[str, Any]]) -> str:
    """Format context with cross-modal references"""
    context_blocks = []
    
    for i, item in enumerate(retrieved_items, 1):
        item_type = item.get('type', 'unknown')
        source = item.get('source', 'Unknown')
        content = item.get('content', '')
        
        # Create cross-modal references
        if item_type == 'image':
            prefix = f"[IMAGE {i}: {source}]"
            if item.get('page'):
                prefix += f" (Page {item['page']})"
            if item.get('caption'):
                content = f"Image Description: {item['caption']}"
            else:
                content = f"Image Content: {content}"
                
        elif item_type == 'audio':
            prefix = f"[AUDIO {i}: {source}]"
            if item.get('start_sec'):
                prefix += f" (Timestamp: {item['start_sec']}s)"
            content = f"Transcript: {content}"
            
        else:  # document
            prefix = f"[DOCUMENT {i}: {source}]"
            if item.get('page'):
                prefix += f" (Page {item['page']})"
            content = f"Content: {content}"
        
        context_blocks.append(f"{prefix}\n{content}\n")
    
    return "\n".join(context_blocks)

def generate_cross_modal_answer(query: str, context: str, retrieved_items: List[Dict[str, Any]]) -> str:
    """Generate answer with cross-modal awareness"""
    
    # Count modalities in retrieved items
    modalities = {}
    for item in retrieved_items:
        modality = item.get('type', 'unknown')
        modalities[modality] = modalities.get(modality, 0) + 1
    
    modality_info = ", ".join([f"{count} {modality}(s)" for modality, count in modalities.items()])
    
    prompt = f"""You are a multimodal assistant that answers questions by synthesizing information from various sources including documents, images, and audio recordings.

CONTEXT FROM MULTIPLE SOURCES ({modality_info}):
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Analyze the question and identify which modalities are most relevant
2. Synthesize information across different sources and modalities
3. Explicitly mention when information comes from different types of sources
4. If images are referenced, describe what they show based on their captions
5. If audio is referenced, summarize the relevant spoken content
6. Provide specific references to source numbers [1], [2], etc.
7. Highlight connections between different modalities when applicable

ANSWER:"""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant that integrates information from documents, images, and audio recordings."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "temperature": OPENROUTER_TEMPERATURE,
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "MultiModal RAG"
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return f"I encountered an error while generating an answer. Status: {response.status_code}"
        
        result = response.json()
        answer_text = result["choices"][0]["message"]["content"].strip()
        return answer_text
        
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "I encountered an error while generating the answer. Please try again."

def answer(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Enhanced QA function with cross-modal support
    """
    if not query or not query.strip():
        return {
            "answer_text": "Please provide a valid question.",
            "citations": [],
            "retrieved": [],
            "error": "Empty query"
        }
    
    try:
        # Retrieve relevant items across all modalities
        retrieved_items = mmr_retrieve(query, k=k)
        
        if not retrieved_items:
            return {
                "answer_text": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "citations": [],
                "retrieved": [],
                "error": "No results found"
            }
        
        # Format context and generate answer
        context = format_cross_modal_context(retrieved_items)
        answer_text = generate_cross_modal_answer(query, context, retrieved_items)
        
        # Prepare detailed citations with proof
        citations = []
        for i, item in enumerate(retrieved_items, 1):
            citation = {
                "number": i,
                "type": item.get('type', 'unknown'),
                "source": item.get('source', 'Unknown'),
                "content": item.get('content', ''),
                "metadata": {
                    k: v for k, v in item.items() 
                    if k not in ['content', 'uuid', 'embedding']
                }
            }
            
            # Add modality-specific proof
            if item.get('type') == 'image':
                citation['proof_type'] = 'visual'
                citation['image_path'] = item.get('orig_path') or item.get('image_path')
                citation['thumbnail'] = item.get('thumbnail')
            elif item.get('type') == 'audio':
                citation['proof_type'] = 'audio'
                citation['audio_path'] = item.get('orig_path')
                citation['timestamp'] = item.get('start_sec')
            else:
                citation['proof_type'] = 'text'
                if item.get('page'):
                    citation['page_reference'] = item.get('page')
            
            citations.append(citation)
        
        return {
            "answer_text": answer_text,
            "citations": citations,
            "retrieved": retrieved_items,
            "retrieval_count": len(retrieved_items),
            "modalities_used": list(set(item.get('type') for item in retrieved_items))
        }
        
    except Exception as e:
        logger.error(f"QA process failed: {e}")
        return {
            "answer_text": "I encountered an error while processing your question. Please try again.",
            "citations": [],
            "retrieved": [],
            "error": str(e)
        }