import io
import logging
import os
from pathlib import Path
from uuid import uuid4

from PIL import Image
from pydub import AudioSegment

from indexer import build_index_from_items  # Changed from add_items
from rag_system.audio_processor import transcribe_audio
from rag_system.image_processor import get_image_caption
from rag_system.text_processor import extract_text_chunks

logger = logging.getLogger(__name__)

THUMBNAIL_DIR = Path("thumbnails")
THUMBNAIL_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

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
    """Process audio file"""
    try:
        transcription = transcribe_audio(path)
        if not transcription:
            logger.warning(f"No transcription obtained for {path}")
            return []
        
        # Handle different transcription formats
        if isinstance(transcription, dict) and 'segments' in transcription:
            # Structured transcription with timestamps
            items = []
            for segment in transcription['segments']:
                items.append({
                    "uuid": str(uuid4()),
                    "type": "audio",
                    "source": os.path.basename(path),
                    "start_sec": segment.get('start', 0),
                    "end_sec": segment.get('end', 0),
                    "content": segment.get('text', ''),
                    "orig_path": path
                })
            return items
        else:
            # Plain text transcription - split into chunks
            text = str(transcription)
            words = text.split()
            chunks = []
            chunk_size = 100  # words per chunk
            
            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i + chunk_size])
                chunks.append({
                    "uuid": str(uuid4()),
                    "type": "audio",
                    "source": os.path.basename(path),
                    "start_sec": i * 5,  # Estimate timing
                    "end_sec": (i + chunk_size) * 5,
                    "content": chunk_text,
                    "orig_path": path
                })
            return chunks
            
    except Exception as e:
        logger.error(f"Audio processing failed for {path}: {e}")
        return []

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
    
    # Replace the existing index with the new items
    if created_items:
        result = build_index_from_items(created_items)  # This replaces the index
        if result[0] is not None:
            logger.info(f"Successfully indexed {len(created_items)} new items")
        else:
            logger.error("Failed to add items to index")
    
    return created_items

def render_citations(citations: list):
    """Render citations in Streamlit"""
    import streamlit as st
    
    if not citations:
        st.info("No citations available")
        return
    
    for i, citation in enumerate(citations, 1):
        if isinstance(citation, str):
            # Legacy string format
            st.markdown(f"**[{i}]** {citation}")
            continue
            
        # Structured citation format
        citation_type = citation.get('type', 'unknown')
        source = citation.get('source', 'Unknown')
        
        with st.expander(f"[{i}] {citation_type.title()}: {source}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Type:**", citation_type)
                st.write("**Source:**", source)
                
                if citation.get('page'):
                    st.write("**Page:**", citation['page'])
                if citation.get('start_sec'):
                    st.write("**Timestamp:**", f"{citation['start_sec']}s")
                if citation.get('caption'):
                    st.write("**Caption:**", citation['caption'])
                if citation.get('pdf_source'):
                    st.write("**PDF Source:**", citation['pdf_source'])
            
            with col2:
                content_preview = citation.get('content_preview', citation.get('content', ''))
                st.write("**Content:**", content_preview)
                
                # Show image thumbnail
                if citation_type == 'image' and citation.get('orig_path'):
                    image_path = citation['orig_path']
                    if os.path.exists(image_path):
                        st.image(image_path, width=200)
                    elif citation.get('thumbnail') and os.path.exists(citation['thumbnail']):
                        st.image(citation['thumbnail'], width=200)
                
                # Show audio player
                elif citation_type == 'audio' and citation.get('orig_path'):
                    audio_path = citation['orig_path']
                    if os.path.exists(audio_path):
                        try:
                            audio = AudioSegment.from_file(audio_path)
                            start_ms = int(citation.get('start_sec', 0) * 1000)
                            end_ms = int(citation.get('end_sec', 0) * 1000) if citation.get('end_sec') else None
                            
                            # Limit snippet to 30 seconds max
                            if end_ms and (end_ms - start_ms) > 30000:
                                end_ms = start_ms + 30000
                            
                            snippet = audio[start_ms:end_ms] if end_ms else audio[start_ms:start_ms + 30000]
                            buf = io.BytesIO()
                            snippet.export(buf, format="mp3")
                            buf.seek(0)
                            
                            st.audio(buf.read(), format="audio/mp3")
                        except Exception as e:
                            st.error(f"Could not play audio: {e}") 