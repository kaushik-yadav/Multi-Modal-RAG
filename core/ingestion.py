import io
import logging
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image
from pydub import AudioSegment

from core.vector_indexer import add_items
from processors.audio_processor import create_audio_chunks
from processors.image_initializer import get_image_caption
from processors.text_processor import extract_text_chunks

logger = logging.getLogger(__name__)

def get_session_thumbnail_dir():
    """Get session-specific thumbnail directory"""
    session_id = os.environ.get("CURRENT_SESSION_ID", "default")
    thumbnail_dir = Path("user_data") / session_id / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    return thumbnail_dir

def get_session_figures_dir():
    """Get session-specific figures directory"""
    session_id = os.environ.get("CURRENT_SESSION_ID", "default")
    figures_dir = Path("user_data") / session_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir

# Replace the global directory declarations with:
THUMBNAIL_DIR = get_session_thumbnail_dir()
FIGURES_DIR = get_session_figures_dir()

def _create_thumbnail(image_path: str, max_size: tuple = (300, 300)) -> str:
    """Create thumbnail for image"""
    try:
        thumb_path = THUMBNAIL_DIR / f"thumb_{Path(image_path).name}"
        if not thumb_path.exists():
            with Image.open(image_path) as img:
                img.thumbnail(max_size)
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(thumb_path)
        return str(thumb_path)
    except Exception as e:
        logger.error(f"Thumbnail creation failed for {image_path}: {e}")
        return None

def _create_doc_items(path: str):
    """Extract content from documents using the updated text_processor"""
    try:
        chunks = extract_text_chunks(path)
        items = []
        
        for chunk in chunks:
            # Handle regular text chunks
            if chunk.get("type") == "document" and not chunk.get("error"):
                items.append({
                    "uuid": str(uuid4()),
                    "type": "document",
                    "source": os.path.basename(path),
                    "page": chunk.get("page"),
                    "content": chunk.get("content", ""),
                    "orig_path": path
                })
            # Handle images extracted from PDFs
            elif chunk.get("type") == "image" and chunk.get("image_path"):
                # Process the extracted image with captioning
                image_path = chunk.get("image_path")
                if os.path.exists(image_path):
                    try:
                        caption = get_image_caption(image_path) or f"Image from page {chunk.get('page')}"
                        thumbnail_path = _create_thumbnail(image_path)
                        
                        items.append({
                            "uuid": str(uuid4()),
                            "type": "image",
                            "source": os.path.basename(path),  # Original PDF name
                            "page": chunk.get("page"),
                            "caption": caption,
                            "content": caption,
                            "orig_path": image_path,  # Path to extracted image
                            "thumbnail": thumbnail_path,
                            "pdf_source": os.path.basename(path)  # Reference to original PDF
                        })
                    except Exception as e:
                        logger.error(f"Failed to process extracted image {image_path}: {e}")
                        # Fallback: create item without caption
                        items.append({
                            "uuid": str(uuid4()),
                            "type": "image", 
                            "source": os.path.basename(path),
                            "page": chunk.get("page"),
                            "caption": f"Image from page {chunk.get('page')}",
                            "content": f"Image from page {chunk.get('page')}",
                            "orig_path": image_path,
                            "pdf_source": os.path.basename(path)
                        })
        
        return items
    except Exception as e:
        logger.error(f"Document processing failed for {path}: {e}")
        return []

def _create_image_item(path: str):
    """Process standalone image file"""
    try:
        caption = get_image_caption(path) or "Image content"
        thumbnail_path = _create_thumbnail(path)
        
        return {
            "uuid": str(uuid4()),
            "type": "image",
            "source": os.path.basename(path),
            "caption": caption,
            "content": caption,  # Use caption for embedding
            "orig_path": path,
            "thumbnail": thumbnail_path
        }
    except Exception as e:
        logger.error(f"Image processing failed for {path}: {e}")
        return None

def _create_audio_items(path: str):
    """Process audio file into optimal chunks based on duration"""
    try:
        abs_path = str(Path(path).resolve())
        
        chunks = create_audio_chunks(abs_path)
        
        items = []
        for chunk in chunks:
            items.append({
                "uuid": str(uuid4()),
                "type": "audio",
                "source": os.path.basename(path),
                "start_sec": chunk['start_sec'],
                "end_sec": chunk['end_sec'],
                "duration": chunk['duration'],
                "content": chunk['content'],
                "orig_path": abs_path,
                "chunk_index": chunk['chunk_index'],
                "metadata": {
                    "chunk_duration": chunk['duration'],
                    "total_chunks": chunk['total_chunks']
                }
            })
        
        logger.info(f"Created {len(items)} audio chunks for {path}")
        return items
        
    except Exception as e:
        logger.error(f"Audio processing failed for {path}: {e}")
        # Fallback: create a single chunk
        abs_path = str(Path(path).resolve())
        return [{
            "uuid": str(uuid4()),
            "type": "audio",
            "source": os.path.basename(path),
            "start_sec": 0,
            "end_sec": 30,  # Default 30-second chunk
            "content": f"Audio file: {os.path.basename(path)}",
            "orig_path": abs_path
        }]

def ingest_paths(paths: list) -> list:
    """Main ingestion function"""
    created_items = []
    
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            continue
            
        path_lower = path.lower()
        
        try:
            if path_lower.endswith(('.pdf', '.docx', '.doc', '.txt')):
                items = _create_doc_items(path)
                created_items.extend(items)
                logger.info(f"Processed document: {path} -> {len(items)} items")
                
            elif path_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                item = _create_image_item(path)
                if item:
                    created_items.append(item)
                    logger.info(f"Processed image: {path}")
                    
            elif path_lower.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
                items = _create_audio_items(path)
                created_items.extend(items)
                logger.info(f"Processed audio: {path} -> {len(items)} items")
                
            else:
                logger.warning(f"Unsupported file type: {path}")
                    
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue
    
    # Add to index
    if created_items:
        result = add_items(created_items)
        if result[0] is not None:
            logger.info(f"Successfully indexed {len(created_items)} new items")
        else:
            logger.error("Failed to add items to index")
    
    return created_items

@st.cache_data(ttl=3600)
def extract_audio_segment(audio_path, start_sec, end_sec, max_duration=30):
    """
    Extract specific audio segment with proper caching
    """
    try:
        if not os.path.exists(audio_path):
            return None
            
        audio = AudioSegment.from_file(audio_path)
        start_ms = int(start_sec * 1000)
        # Ensure we don't exceed 30 seconds
        actual_end_sec = min(end_sec, start_sec + max_duration)
        end_ms = int(actual_end_sec * 1000)
        
        segment = audio[start_ms:end_ms]
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None