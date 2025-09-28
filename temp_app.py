import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from pydub import AudioSegment

from indexer import get_index_stats
from ingest_pipeline import ingest_paths
from qa_openrouter import answer

# Create directories
directories = ["uploads", "thumbnails", "figures", "data", "data/pdf_images", "user_data"]
for directory in directories:
    Path(directory).mkdir(exist_ok=True)

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
        except Exception as e:
            logger.error(f"Failed to save {uploaded_file.name}: {e}")
    
    # Ingest files with proper environment variable handling
    if saved_paths:
        with st.spinner(f"Processing {len(saved_paths)} files..."):
            try:
                # Set environment variables for user-specific indexing
                original_index_path = os.environ.get("FAISS_INDEX_PATH")
                original_meta_path = os.environ.get("FAISS_META_PATH")
                
                os.environ["FAISS_INDEX_PATH"] = user_paths["faiss_index"]
                os.environ["FAISS_META_PATH"] = user_paths["faiss_metadata"]
                
                # Call ingest_paths
                results = ingest_paths(saved_paths)
                
                # Restore original environment variables
                if original_index_path:
                    os.environ["FAISS_INDEX_PATH"] = original_index_path
                else:
                    os.environ.pop("FAISS_INDEX_PATH", None)
                    
                if original_meta_path:
                    os.environ["FAISS_META_PATH"] = original_meta_path
                else:
                    os.environ.pop("FAISS_META_PATH", None)
                
                return {
                    "processed": len(results) if results else 0,
                    "total_files": len(saved_paths)
                }
                
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                # Restore original environment variables on error
                if original_index_path:
                    os.environ["FAISS_INDEX_PATH"] = original_index_path
                if original_meta_path:
                    os.environ["FAISS_META_PATH"] = original_meta_path
                return {"processed": 0, "errors": [f"Indexing failed: {str(e)}"]}
    
    return {"processed": 0, "errors": ["Failed to save files"]}

@st.cache_data(ttl=3600)
def extract_audio_segment(audio_path, start_sec, end_sec, max_duration=30):
    """Extract specific audio segment with proper timing"""
    try:
        # Try multiple path resolution strategies
        possible_paths = [
            audio_path,  # Original path
            Path(audio_path),  # As Path object
            Path("user_data") / st.session_state.user_session_id / "uploads" / Path(audio_path).name,
            st.session_state.user_paths["uploads"] / Path(audio_path).name,
        ]
        
        actual_path = None
        for path in possible_paths:
            if isinstance(path, Path) and path.exists():
                actual_path = path
                break
            elif isinstance(path, str) and os.path.exists(path):
                actual_path = Path(path)
                break
        
        if not actual_path or not actual_path.exists():
            logger.error(f"Audio file not found at any path: {audio_path}")
            return None
            
        logger.info(f"Extracting audio from {actual_path}: {start_sec}s to {end_sec}s")
        audio = AudioSegment.from_file(actual_path)
        
        # Convert seconds to milliseconds
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # Ensure we don't exceed audio length
        if end_ms > len(audio):
            end_ms = len(audio)
        
        # Extract the specific segment
        segment = audio[start_ms:end_ms]
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None

def find_audio_file(audio_path, filename):
    """Find audio file using multiple search strategies"""
    # Strategy 1: Try the original path
    if audio_path and os.path.exists(audio_path):
        return audio_path
    
    # Strategy 2: Try with user session path
    user_audio_path = st.session_state.user_paths["uploads"] / filename
    if user_audio_path.exists():
        return str(user_audio_path)
    
    # Strategy 3: Search in user data directory
    user_data_dir = st.session_state.user_paths["user_data_dir"]
    for ext in ['.mp3', '.wav', '.m4a', '.ogg']:
        possible_path = user_data_dir / "uploads" / f"{Path(filename).stem}{ext}"
        if possible_path.exists():
            return str(possible_path)
    
    # Strategy 4: Search recursively in user data directory
    for audio_file in user_data_dir.rglob("*" + Path(filename).suffix):
        if audio_file.name == filename:
            return str(audio_file)
    
    return None

def render_citations(citations: list):
    """Render citations with fixed audio timing and unique keys"""
    print(citations)
    if not citations:
        st.info("No citations available")
        return
    
    for i, citation in enumerate(citations, 1):
        if isinstance(citation, str):
            st.markdown(f"**[{i}]** {citation}")
            continue
            
        citation_type = citation.get('type', 'unknown').capitalize()
        source = citation.get('source', 'Unknown')
        
        # Create expandable citation card
        with st.expander(f"[{i}] {citation_type}: {source}", expanded=False):
            col1, col2 = st.columns([1, 3])
            
            with col1:
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
                content = citation.get('content', '')
                display_content = content[:300] + "..." if len(content) > 300 else content
                st.write("**Content:**", display_content)
            
            # Proof display section
            st.markdown("---")
            st.subheader("🔍 Proof")
            
            # Image proof
            if citation_type.lower() == 'image':
                image_path = citation.get('orig_path') or citation.get('image_path')
                if image_path and os.path.exists(image_path):
                    try:
                        # Add custom CSS
                        st.markdown("""
                        <style>
                        .image-container img {
                            width: auto !important;
                            height: 100% !important;
                            max-width: 100% !important;
                            object-fit: contain !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Display image with custom container
                        st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                        st.image(image_path, caption=f"Original image from {source}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Could not display image: {e}")
                else:
                    st.warning("Image file not available for display")
            
            # Audio proof - FIXED TIMING AND UNIQUE KEYS
            elif citation_type.lower() == 'audio':
                # top-level fields
                audio_path = citation.get("audio_path") or citation.get("metadata", {}).get("orig_path")
                filename = citation.get("source", "")

                # check if timing info is inside metadata
                meta = citation.get("metadata", {})

                start_sec = meta.get("start_sec", 0)
                end_sec = meta.get("end_sec", start_sec + 30)
                chunk_index = meta.get("chunk_index", 0)
                st.write(f"**Audio File:** {filename}")
                st.write(f"**Chunk {chunk_index + 1}:** {start_sec:.1f}s - {end_sec:.1f}s")
                st.write(f"**Duration:** {end_sec - start_sec:.1f} seconds")
                
                # Try to find the audio file
                found_audio_path = find_audio_file(audio_path, filename)
                
                if found_audio_path and os.path.exists(found_audio_path):
                    try:
                        # Extract and play the correct segment
                        audio_bytes = extract_audio_segment(found_audio_path, start_sec, end_sec)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            st.success(f"✅ Audio segment {chunk_index + 1} loaded successfully")
                        else:
                            st.error("Could not extract audio segment")
                            st.info("This might be due to file format issues or the segment being too short.")
                            
                    except Exception as e:
                        st.error(f"Could not play audio: {e}")
                        
                        # Fallback download with UNIQUE KEY
                        try:
                            with open(found_audio_path, "rb") as f:
                                audio_data = f.read()
                            st.download_button(
                                label="📥 Download Audio File (Fallback)",
                                data=audio_data,
                                file_name=filename,
                                mime="audio/mpeg",
                                key=f"download_fallback_{i}_{chunk_index}_{hash(filename)}"  # Unique key
                            )
                        except Exception as download_error:
                            st.error(f"Download also failed: {download_error}")
                else:
                    st.warning(f"Audio file not found: {filename}")
                    
                    # Show available audio files for debugging
                    uploads_dir = st.session_state.user_paths["uploads"]
                    if uploads_dir.exists():
                        audio_files = list(uploads_dir.glob("*.mp3")) + list(uploads_dir.glob("*.wav")) + list(uploads_dir.glob("*.m4a"))
                        if audio_files:
                            st.write("**Available audio files in your session:**")
                            for af in audio_files:
                                st.write(f"• {af.name}")
            
            # Document proof
            elif citation_type.lower() == 'document':
                content = citation.get('content', '')
                if content:
                    st.text_area(
                        "Document Excerpt:",
                        value=content,
                        height=150,
                        key=f"doc_excerpt_{i}_{hash(str(citation))}",  # Unique key
                        disabled=True
                    )
                
                if citation.get('page'):
                    st.info(f"Referenced from page {citation['page']}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SmartSearch AI - Multimodal Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# Initialize user session
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = get_user_session_id()
    st.session_state.user_paths = get_user_specific_paths(st.session_state.user_session_id)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'index_stats' not in st.session_state:
    st.session_state.index_stats = {"total_items": 0, "index_exists": False}

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-header {
        font-size: 1.6rem;
        color: #1e40af;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
    }
    .answer-box {
        background-color: #f0f9ff;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1e293b;
    }
    .file-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1e293b;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .user-session-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid #8b5cf6;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: white;
    }
    .warning-message {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
    .audio-debug {
        background-color: #f0f9ff;
        border: 1px solid #7dd3fc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🔍 SmartSearch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal Document Intelligence • Search Across Documents, Images & Audio</div>', unsafe_allow_html=True)

# User session info
st.sidebar.markdown(f"""
<div class="user-session-info">
    <strong>👤 Your Private Session</strong><br>
    <small>Session ID: {st.session_state.user_session_id}</small>
</div>
""", unsafe_allow_html=True)

# Sidebar - File Management
st.sidebar.header("📁 Document Management")

# File upload section
st.sidebar.markdown("### Add Documents to Your Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop files here",
    type=['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'mp3', 'wav', 'm4a'],
    accept_multiple_files=True,
    help="Supported formats: PDF, Word, Text, Images, Audio files",
    key="file_uploader"
)

if uploaded_files:
    st.sidebar.markdown(f"**Selected files:** {len(uploaded_files)}")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")

# Process files button
if st.sidebar.button("🚀 Process & Index Files", type="primary", use_container_width=True):
    if not uploaded_files:
        st.sidebar.error("❌ Please select files first")
    else:
        result = process_uploaded_files(uploaded_files, st.session_state.user_paths)
        
        if result["processed"] > 0:
            st.sidebar.markdown(f"""
            <div class="success-message">
                ✅ Successfully processed {result['processed']} items from {len(uploaded_files)} files
            </div>
            """, unsafe_allow_html=True)
            
            # Update processed files list
            new_files = [f.name for f in uploaded_files]
            st.session_state.processed_files.extend(new_files)
            
            # Update index stats
            try:
                # Set environment variables for stats
                original_index_path = os.environ.get("FAISS_INDEX_PATH")
                original_meta_path = os.environ.get("FAISS_META_PATH")
                
                os.environ["FAISS_INDEX_PATH"] = st.session_state.user_paths["faiss_index"]
                os.environ["FAISS_META_PATH"] = st.session_state.user_paths["faiss_metadata"]
                
                st.session_state.index_stats = get_index_stats()
                
                # Restore environment variables
                if original_index_path:
                    os.environ["FAISS_INDEX_PATH"] = original_index_path
                if original_meta_path:
                    os.environ["FAISS_META_PATH"] = original_meta_path
                    
            except Exception as e:
                logger.error(f"Failed to get index stats: {e}")
                st.session_state.index_stats = {"total_items": result["processed"], "index_exists": True}
            
            st.rerun()
        else:
            error_msg = result.get("errors", ["Unknown error"])
            st.sidebar.error(f"❌ Failed to process files: {error_msg}")

# Index statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Knowledge Base Status")

stats = st.session_state.index_stats
if stats.get("index_exists") and stats.get("total_items", 0) > 0:
    st.sidebar.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 2rem; font-weight: bold;">{stats['total_items']}</div>
        <div>Total Documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Type distribution
    if stats.get('type_distribution'):
        for file_type, count in stats.get('type_distribution', {}).items():
            icon = "📄" if file_type == "document" else "🖼️" if file_type == "image" else "🎵"
            st.sidebar.write(f"{icon} {file_type.title()}: **{count}**")
else:
    st.sidebar.info("💡 Add documents to build your knowledge base")

# Display current files
if st.session_state.processed_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Your Files")
    unique_files = list(set(st.session_state.processed_files))
    for file in sorted(unique_files)[:5]:
        st.sidebar.write(f"• {file}")
    if len(unique_files) > 5:
        st.sidebar.write(f"... and {len(unique_files) - 5} more")

# Audio debug information
if st.sidebar.checkbox("🔧 Show Audio Debug Info"):
    st.sidebar.markdown("### Audio Debug Information")
    uploads_dir = st.session_state.user_paths["uploads"]
    if uploads_dir.exists():
        audio_files = list(uploads_dir.glob("*.mp3")) + list(uploads_dir.glob("*.wav")) + list(uploads_dir.glob("*.m4a"))
        st.sidebar.write(f"**Audio files found:** {len(audio_files)}")
        for af in audio_files:
            st.sidebar.write(f"• {af.name}")
    else:
        st.sidebar.write("Uploads directory not found")

# Main content area
tab1, tab2 = st.tabs(["🔍 Smart Search", "📚 Your Documents"])

with tab1:
    st.markdown("""
    <div class="section-header">Ask Anything About Your Documents</div>
    <p style="color: #64748b; font-size: 1.1rem;">
    Ask questions in plain English. Your searches will only use files you've uploaded in this session.
    </p>
    """, unsafe_allow_html=True)
    
    # Check if user has files
    has_files = len(st.session_state.processed_files) > 0
    
    if not has_files:
        st.markdown("""
        <div class="warning-message">
            ⚠️ <strong>No files uploaded yet.</strong> Please upload and process files first to enable search.
        </div>
        """, unsafe_allow_html=True)
    
    # Simple query interface
    query_text = st.text_area(
        "Your question:",
        placeholder="e.g., 'What are the main points discussed in the meeting about the project timeline?' or 'Show me diagrams related to the system architecture'",
        height=100,
        key="main_query"
    )
    
    # Search button
    search_button = st.button("🔍 Search Knowledge Base", 
                              type="primary", 
                              use_container_width=True,
                              key="search_button")
    
    if search_button:
        if not query_text:
            st.error("❌ Please enter a question first.")
        elif not has_files:
            st.error("❌ Please upload and process files first.")
        else:
            with st.spinner("Searching across your documents..."):
                try:
                    # Set user-specific paths for search
                    original_index_path = os.environ.get("FAISS_INDEX_PATH")
                    original_meta_path = os.environ.get("FAISS_META_PATH")
                    
                    os.environ["FAISS_INDEX_PATH"] = st.session_state.user_paths["faiss_index"]
                    os.environ["FAISS_META_PATH"] = st.session_state.user_paths["faiss_metadata"]
                    
                    # Perform search
                    response = answer(query_text, k=6)
                    
                    # Restore original paths
                    if original_index_path:
                        os.environ["FAISS_INDEX_PATH"] = original_index_path
                    if original_meta_path:
                        os.environ["FAISS_META_PATH"] = original_meta_path
                    
                    # Display results
                    st.markdown("### 💡 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                        {response["answer_text"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources with enhanced proof display
                    if response.get("citations"):
                        st.markdown("### 📋 Sources & Evidence")
                        st.info("💡 **Click on each citation to view the original proof (images, audio segments, document excerpts)**")
                        
                        # Audio timing debug info
                        audio_citations = [c for c in response.get("citations", []) if isinstance(c, dict) and c.get('type') == 'audio']
                        if audio_citations:
                            st.markdown("""
                            <div class="audio-debug">
                                <strong>Audio Segments Information:</strong><br>
                                Each audio citation should play the specific segment mentioned in the transcription.
                                If you hear the wrong segment, please check the start/end times shown above.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        render_citations(response["citations"])
                    else:
                        st.info("No specific sources were referenced in this answer.")
                    
                except Exception as e:
                    st.error(f"Sorry, we encountered an error while processing your request. Please try again.")
                    logger.error(f"Search error: {e}")

with tab2:
    st.markdown("""
    <div class="section-header">Your Document Library</div>
    <p style="color: #64748b;">
    All documents currently in your private session
    </p>
    """, unsafe_allow_html=True)
    
    if st.session_state.processed_files:
        unique_files = list(set(st.session_state.processed_files))
        
        # Summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            doc_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.pdf', '.doc', '.docx', '.txt']])
            st.metric("Documents", doc_count)
        
        with col2:
            img_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            st.metric("Images", img_count)
            
        with col3:
            audio_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.mp3', '.wav', '.m4a']])
            st.metric("Audio Files", audio_count)
        
        # File browser
        st.markdown("### 📂 Browse Files")
        
        # Filter by type
        file_type_filter = st.selectbox(
            "Filter by type:",
            ["All files", "Documents", "Images", "Audio"],
            key="file_filter"
        )
        
        filtered_files = unique_files
        if file_type_filter == "Documents":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.pdf', '.doc', '.docx', '.txt']]
        elif file_type_filter == "Images":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        elif file_type_filter == "Audio":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.mp3', '.wav', '.m4a']]
        
        # Display files
        if filtered_files:
            for file in sorted(filtered_files):
                file_ext = Path(file).suffix.lower()
                icon = "📄" if file_ext in ['.pdf', '.doc', '.docx', '.txt'] else "🖼️" if file_ext in ['.jpg', '.jpeg', '.png', '.bmp'] else "🎵"
                
                st.markdown(f"""
                <div class="file-card">
                    <div style="display: flex; align-items: center;">
                        <div style="font-size: 1.5rem; margin-right: 1rem;">{icon}</div>
                        <div style="flex: 1;">
                            <strong>{file}</strong><br>
                            <small style="color: #64748b;">{file_ext.upper().replace('.', '')} file</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No files match the current filter.")
        
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <h3>📚 Your library is empty</h3>
            <p>Add documents to your knowledge base using the sidebar uploader.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "SmartSearch AI • Private Multimodal Search • "
    "Each session is completely isolated and private"
    "</div>",
    unsafe_allow_html=True
)