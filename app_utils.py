import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from time import time

import streamlit as st
from PIL import Image
from pydub import AudioSegment
from pydub.utils import which

from indexer import get_index_stats
from ingest_pipeline import ingest_paths
from qa_openrouter import answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session-aware wrapper functions
def set_session_paths(user_paths):
    """Set environment variables for session-specific paths"""
    os.environ["FAISS_INDEX_PATH"] = user_paths["faiss_index"]
    os.environ["FAISS_META_PATH"] = user_paths["faiss_metadata"]
    os.environ["ID_MAP_PATH"] = user_paths["id_map"]
    os.environ["CURRENT_SESSION_ID"] = st.session_state.user_session_id

def restore_original_paths(original_paths):
    """Restore original environment variables"""
    for key, value in original_paths.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    os.environ.pop("CURRENT_SESSION_ID", None)

def get_user_session_id():
    """Generate a unique session ID for each user"""
    try:
        # Create a simple session ID based on time and random factor
        import random
        import time
        session_id = f"user_{int(time.time())}_{random.randint(1000, 9999)}"
        return session_id
    except:
        # Fallback: use a random session ID
        import random
        return f"user_{random.randint(1000, 9999)}"

def cleanup_old_sessions():
    """Remove old session directories to free up space"""
    try:
        user_data_dir = Path("user_data")
        if not user_data_dir.exists():
            return
        
        current_time = time.time()
        sessions_to_remove = []
        
        # Get all session directories with their creation times
        session_dirs = []
        for session_dir in user_data_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("user_"):
                try:
                    # Get directory creation time
                    dir_mtime = session_dir.stat().st_mtime
                    session_dirs.append((session_dir, dir_mtime))
                except:
                    continue
        
        # Sort by modification time (oldest first)
        session_dirs.sort(key=lambda x: x[1])
        
        # Remove sessions older than timeout
        SESSION_TIMEOUT = 120000
        for session_dir, mtime in session_dirs:
            age = current_time - mtime
            if age > SESSION_TIMEOUT:
                sessions_to_remove.append(session_dir)
        MAX_SESSIONS = 2
        # If we still have too many sessions, remove oldest ones
        if len(session_dirs) - len(sessions_to_remove) > MAX_SESSIONS:
            remaining_sessions = [s for s in session_dirs if s[0] not in sessions_to_remove]
            excess_count = len(remaining_sessions) - MAX_SESSIONS
            for i in range(excess_count):
                sessions_to_remove.append(remaining_sessions[i][0])
        
        # Actually remove the directories
        removed_count = 0
        for session_dir in sessions_to_remove:
            try:
                # Skip current session
                if hasattr(st.session_state, 'user_session_id') and session_dir.name.endswith(st.session_state.user_session_id.split('_')[-1]):
                    continue
                    
                import shutil
                shutil.rmtree(session_dir)
                removed_count += 1
                logger.info(f"Removed old session directory: {session_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove session directory {session_dir}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old session directories")
            
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

def get_session_info():
    """Get information about current session usage"""
    try:
        user_data_dir = Path("user_data")
        if not user_data_dir.exists():
            return {"total_sessions": 0, "current_session_size": 0, "current_session_size_mb": 0}
        
        total_sessions = len([d for d in user_data_dir.iterdir() if d.is_dir()])
        
        # Calculate current session size
        current_session_size = 0
        if hasattr(st.session_state, 'user_paths'):
            session_dir = st.session_state.user_paths["user_data_dir"]
            if session_dir.exists():
                for file in session_dir.rglob("*"):
                    if file.is_file():
                        try:
                            current_session_size += file.stat().st_size
                        except:
                            continue
        
        return {
            "total_sessions": total_sessions,
            "current_session_size": current_session_size,
            "current_session_size_mb": current_session_size / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return {"total_sessions": 0, "current_session_size": 0, "current_session_size_mb": 0}

def get_user_specific_paths(session_id):
    """Get user-specific file paths"""
    user_data_dir = Path("user_data") / session_id
    user_data_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "faiss_index": str(user_data_dir / "faiss_index.index"),
        "faiss_metadata": str(user_data_dir / "faiss_metadata.json"),
        "id_map": str(user_data_dir / "id_map.json"),
        "uploads": user_data_dir / "uploads",
        "user_data_dir": user_data_dir
    }

def process_uploaded_files(uploaded_files, user_paths):
    """Process uploaded files with user isolation - FIXED VERSION"""
    if not uploaded_files:
        return {"processed": 0, "errors": []}
    
    # Create user-specific uploads directory
    user_paths["uploads"].mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        try:
            file_path = user_paths["uploads"] / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(str(file_path))
            logger.info(f"Saved file: {file_path} (exists: {file_path.exists()}, size: {file_path.stat().st_size if file_path.exists() else 'N/A'})")
        except Exception as e:
            logger.error(f"Failed to save {uploaded_file.name}: {e}")
    
    # Ingest files with proper environment variable handling
    if saved_paths:
        with st.spinner(f"Processing {len(saved_paths)} files..."):
            try:
                # Store original environment variables
                original_paths = {
                    "FAISS_INDEX_PATH": os.environ.get("FAISS_INDEX_PATH"),
                    "FAISS_META_PATH": os.environ.get("FAISS_META_PATH"),
                    "ID_MAP_PATH": os.environ.get("ID_MAP_PATH")
                }
                
                # Set user-specific paths
                set_session_paths(user_paths)
                
                # Call ingest_paths
                results = ingest_paths(saved_paths)
                
                # Restore original environment variables
                restore_original_paths(original_paths)
                
                return {
                    "processed": len(results) if results else 0,
                    "total_files": len(saved_paths)
                }
                
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                # Restore original environment variables on error
                restore_original_paths(original_paths)
                return {"processed": 0, "errors": [f"Indexing failed: {str(e)}"]}
    
    return {"processed": 0, "errors": ["Failed to save files"]}

def remove_uploaded_file(filename, user_paths):
    """Remove a file from the knowledge base"""
    try:
        # Remove file from uploads directory
        file_path = user_paths["uploads"] / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Removed file: {file_path}")
        
        # Rebuild index without the removed file
        if user_paths["uploads"].exists():
            remaining_files = []
            for remaining_file in user_paths["uploads"].iterdir():
                if remaining_file.is_file() and remaining_file.name != filename:
                    remaining_files.append(str(remaining_file))
            
            if remaining_files:
                # Reindex remaining files
                original_paths = {
                    "FAISS_INDEX_PATH": os.environ.get("FAISS_INDEX_PATH"),
                    "FAISS_META_PATH": os.environ.get("FAISS_META_PATH"),
                    "ID_MAP_PATH": os.environ.get("ID_MAP_PATH")
                }
                
                set_session_paths(user_paths)
                
                # Clear existing index files
                index_path = Path(user_paths["faiss_index"])
                meta_path = Path(user_paths["faiss_metadata"])
                id_map_path = Path(user_paths["id_map"])
                
                if index_path.exists():
                    index_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                if id_map_path.exists():
                    id_map_path.unlink()
                
                # Reindex remaining files
                results = ingest_paths(remaining_files)
                restore_original_paths(original_paths)
                
                return len(results) if results else 0
            else:
                # No files left, clear index
                index_path = Path(user_paths["faiss_index"])
                meta_path = Path(user_paths["faiss_metadata"])
                id_map_path = Path(user_paths["id_map"])
                
                if index_path.exists():
                    index_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                if id_map_path.exists():
                    id_map_path.unlink()
                
                return 0
        return 0
    except Exception as e:
        logger.error(f"Failed to remove file {filename}: {e}")
        return -1

def extract_audio_segment(audio_path, start_sec, end_sec, max_duration=30):
    """Extract specific audio segment with enhanced error handling for Streamlit Cloud"""
    try:
        # Try multiple path resolution strategies with absolute paths
        possible_paths = []
        
        # Convert to Path object for consistent handling
        audio_path_obj = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
        
        # Strategy 1: Original path (absolute)
        if audio_path_obj.is_absolute() and audio_path_obj.exists():
            possible_paths.append(audio_path_obj)
        
        # Strategy 2: Relative to current working directory
        cwd_path = Path.cwd() / audio_path_obj.name
        if cwd_path.exists():
            possible_paths.append(cwd_path)
        
        # Strategy 3: User session upload directory
        if hasattr(st.session_state, 'user_paths') and st.session_state.user_paths:
            user_upload_path = st.session_state.user_paths["uploads"] / audio_path_obj.name
            if user_upload_path.exists():
                possible_paths.append(user_upload_path)
        
        # Strategy 4: Direct filename search in user data
        if hasattr(st.session_state, 'user_session_id'):
            user_data_dir = Path("user_data") / st.session_state.user_session_id / "uploads"
            direct_path = user_data_dir / audio_path_obj.name
            if direct_path.exists():
                possible_paths.append(direct_path)
        
        # Strategy 5: Search all supported audio extensions
        base_name = audio_path_obj.stem
        for ext in ['.mp3', '.wav', '.m4a', '.ogg']:
            if hasattr(st.session_state, 'user_paths') and st.session_state.user_paths:
                ext_path = st.session_state.user_paths["uploads"] / f"{base_name}{ext}"
                if ext_path.exists():
                    possible_paths.append(ext_path)
        
        # Find the first existing path
        actual_path = None
        for path in possible_paths:
            if path.exists():
                actual_path = path
                logger.info(f"Found audio file at: {actual_path}")
                break
        
        if not actual_path:
            logger.error(f"Audio file not found. Searched paths: {[str(p) for p in possible_paths]}")
            return None
            
        logger.info(f"Extracting audio from {actual_path}: {start_sec}s to {end_sec}s")
        
        # Load audio with error handling for different formats
        try:
            audio = AudioSegment.from_file(str(actual_path))
        except Exception as load_error:
            logger.error(f"Failed to load audio file {actual_path}: {load_error}")
            # Try different audio libraries/methods
            try:
                audio = AudioSegment.from_mp3(str(actual_path))
            except:
                try:
                    audio = AudioSegment.from_wav(str(actual_path))
                except:
                    logger.error(f"Could not load audio file with any method")
                    return None
        
        # Convert seconds to milliseconds
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # Ensure we don't exceed audio length
        if end_ms > len(audio):
            end_ms = len(audio)
            logger.warning(f"Adjusted end time to audio length: {end_ms/1000}s")
        
        if start_ms >= len(audio):
            logger.warning(f"Start time {start_ms/1000}s exceeds audio length {len(audio)/1000}s")
            return None
        
        # Extract the specific segment
        segment = audio[start_ms:end_ms]
        
        if len(segment) == 0:
            logger.warning(f"Empty audio segment extracted")
            return None
        
        # Export to bytes
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        audio_bytes = buf.read()
        
        logger.info(f"Successfully extracted {len(audio_bytes)} bytes of audio")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def find_audio_file(audio_path, filename):
    """Enhanced audio file finder with comprehensive search strategies"""
    logger.info(f"Searching for audio file: {filename} (original path: {audio_path})")
    
    # Convert to Path objects
    if isinstance(audio_path, str):
        audio_path = Path(audio_path)
    filename_path = Path(filename)
    
    search_paths = []
    
    # Strategy 1: Try the original path
    if audio_path and audio_path.exists():
        logger.info(f"Found at original path: {audio_path}")
        return str(audio_path)
    
    # Strategy 2: Current working directory
    cwd_path = Path.cwd() / filename
    search_paths.append(cwd_path)
    
    # Strategy 3: User session upload directory (primary)
    if hasattr(st.session_state, 'user_paths') and st.session_state.user_paths:
        user_upload_path = st.session_state.user_paths["uploads"] / filename
        search_paths.append(user_upload_path)
    
    # Strategy 4: User data directory variations
    if hasattr(st.session_state, 'user_session_id'):
        user_data_dir = Path("user_data") / st.session_state.user_session_id
        search_paths.extend([
            user_data_dir / "uploads" / filename,
            user_data_dir / filename,
        ])
    
    # Strategy 5: Different extensions
    base_name = filename_path.stem
    for ext in ['.mp3', '.wav', '.m4a', '.ogg']:
        if hasattr(st.session_state, 'user_paths') and st.session_state.user_paths:
            ext_path = st.session_state.user_paths["uploads"] / f"{base_name}{ext}"
            search_paths.append(ext_path)
    
    # Strategy 6: Recursive search in user directory
    if hasattr(st.session_state, 'user_session_id'):
        user_data_dir = Path("user_data") / st.session_state.user_session_id
        if user_data_dir.exists():
            for audio_file in user_data_dir.rglob("*"):
                if audio_file.is_file() and audio_file.name == filename:
                    search_paths.append(audio_file)
    
    # Check all search paths
    for path in search_paths:
        if path and path.exists():
            logger.info(f"Found audio file at: {path}")
            return str(path)
    
    # Log all attempted paths for debugging
    logger.error(f"Audio file '{filename}' not found. Searched paths:")
    for i, path in enumerate(search_paths, 1):
        logger.error(f"  {i}. {path} (exists: {path.exists() if path else False})")
    
    return None

def render_citations(citations: list):
    """Render citations with enhanced audio handling and debugging"""
    if not citations:
        st.info("No citations available")
        return
    
    for i, citation in enumerate(citations, 1):
        if isinstance(citation, str):
            st.markdown(f"**[{i}]** {citation}")
            continue
            
        citation_type = citation.get('type', 'unknown').capitalize()
        source = citation.get('source', 'Unknown')
        
        # Create expandable citation card with modern styling
        with st.expander(f"🔗 Source [{i}] • {citation_type}: {source}", expanded=False):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("##### 📋 Metadata")
                st.write("**Type:**", citation_type)
                st.write("**Source:**", source)
                
                if citation.get('page'):
                    st.write("**Page:**", citation['page'])
                if citation.get('start_sec') is not None:
                    st.write("**Start Time:**", f"{citation['start_sec']:.1f}s")
                if citation.get('end_sec') is not None:
                    st.write("**End Time:**", f"{citation['end_sec']:.1f}s")
                if citation.get('caption'):
                    caption = citation.get('caption', '')
                    display_caption = caption[:100] + "..." if len(caption) > 100 else caption
                    st.write("**Caption:**", display_caption)
                if citation.get('chunk_index') is not None:
                    st.write("**Chunk:**", f"{citation['chunk_index'] + 1}")
            
            with col2:
                st.markdown("##### 📝 Content Preview")
                content = citation.get('content', '')
                display_content = content[:300] + "..." if len(content) > 300 else content
                st.markdown(f'<div class="content-preview">{display_content}</div>', unsafe_allow_html=True)
            
            # Proof display section
            st.markdown("---")
            st.markdown("#### 🔍 Evidence")
            
            # Image proof
            if citation_type.lower() == 'image':
                image_path = citation.get('orig_path') or citation.get('image_path')
                if image_path and os.path.exists(image_path):
                    try:
                        st.image(image_path, caption=f"Original image from {source}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not display image: {e}")
                else:
                    st.warning("Image file not available for display")
            
            # Audio proof - ENHANCED VERSION
            elif citation_type.lower() == 'audio':
                
                # Extract audio information with multiple fallback strategies
                audio_path = None
                filename = source or "Unknown"
                start_sec = 0
                end_sec = 30
                chunk_index = 0
                
                # Strategy 1: Top-level fields
                if citation.get("audio_path"):
                    audio_path = citation["audio_path"]
                if citation.get("source"):
                    filename = citation["source"]
                
                # Strategy 2: Metadata fields
                meta = citation.get("metadata", {})
                if meta:
                    audio_path = audio_path or meta.get("orig_path") or meta.get("audio_path")
                    start_sec = meta.get("start_sec", start_sec)
                    end_sec = meta.get("end_sec", end_sec) 
                    chunk_index = meta.get("chunk_index", chunk_index)
                
                # Strategy 3: Direct fields
                start_sec = citation.get("start_sec", start_sec)
                end_sec = citation.get("end_sec", end_sec)
                chunk_index = citation.get("chunk_index", chunk_index)
                
                # Ensure end_sec is reasonable
                if end_sec <= start_sec:
                    end_sec = start_sec + 30
                
                st.markdown(f"""
                <div class="audio-info">
                    🎵 <strong>{filename}</strong><br/>
                    Segment {chunk_index + 1}: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec - start_sec:.1f}s duration)
                </div>
                """, unsafe_allow_html=True)
                
                # Try to find the audio file
                found_audio_path = find_audio_file(audio_path, filename)
                
                if found_audio_path and os.path.exists(found_audio_path):
                    try:
                        # Show file info
                        file_size = os.path.getsize(found_audio_path)
                        
                        # Extract and play the correct segment
                        with st.spinner("Extracting audio segment..."):
                            audio_bytes = extract_audio_segment(found_audio_path, start_sec, end_sec)
                        
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                        else:
                            st.error("Could not extract audio segment")
                            
                            # Fallback: Try to play the whole file
                            try:
                                with open(found_audio_path, "rb") as f:
                                    full_audio = f.read()
                                st.warning("Playing full audio file instead:")
                                st.audio(full_audio, format="audio/mp3")
                            except Exception as fallback_error:
                                st.error(f"Could not play full audio: {fallback_error}")
                                
                    except Exception as e:
                        st.error(f"Could not process audio: {e}")
                        logger.error(f"Audio processing error: {e}")
                        
                        # Final fallback: download option
                        try:
                            with open(found_audio_path, "rb") as f:
                                audio_data = f.read()
                            st.download_button(
                                label="📥 Download Audio File",
                                data=audio_data,
                                file_name=filename,
                                mime="audio/mpeg",
                                key=f"download_audio_{i}_{chunk_index}_{hash(filename)}"
                            )
                        except Exception as download_error:
                            st.error(f"Download also failed: {download_error}")
                else:
                    st.error(f"❌ Audio file not found: {filename}")
            
            # Document proof
            elif citation_type.lower() == 'document':
                content = citation.get('content', '')
                            
                if citation.get('page'):
                    st.info(f"📄 Referenced from page {citation['page']}")
                if content:
                    st.text_area(
                        "Document Excerpt:",
                        value=content,
                        height=350,
                        key=f"doc_excerpt_{i}_{hash(str(citation))}",
                        disabled=True
                    )

