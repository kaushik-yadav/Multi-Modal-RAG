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
                        st.image(image_path, caption=f"Original image from {source}", use_column_width=True)
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
                        # Extract and play the correct segment
                        audio_bytes = extract_audio_segment(found_audio_path, start_sec, end_sec)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            st.success(f"✅ Audio segment {chunk_index + 1} loaded successfully")
                        else:
                            st.error("Could not extract audio segment")
                            
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
                    st.info(f"📄 Referenced from page {citation['page']}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SmartSearch AI - Multimodal Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔮"
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

# Modern Glassmorphic CSS with elegant gradients
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container glassmorphism */
    .main {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styles */
    .main-header {
    font-size: 4rem;
    font-weight: 800;
    color: #fff; /* visible text */
    text-shadow: 0 0 5px #667eea, 0 0 10px #764ba2, 0 0 20px #764ba2;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -2px;
    animation: glow 2s ease-in-out infinite alternate;
    }

    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        to { text-shadow: 0 0 30px rgba(118, 75, 162, 0.5); }
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Card styles with glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Answer box with gradient border */
    .answer-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        z-index: -1;
        margin: -2px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c);
        animation: gradient-border 3s ease infinite;
    }
    
    @keyframes gradient-border {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-size: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* File card styling */
    .file-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        transition: all 0.3s ease;
    }
    
    .file-card:hover {
        transform: translateX(5px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Session info card */
    .session-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    /* Content preview */
    .content-preview {
        background: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Audio info box */
    .audio-info {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
    }
    
    /* Stats display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Spinner custom */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Animated gradient background for special elements */
    .gradient-bg {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Main header with animated gradient
st.markdown('<div class="main-header">🔮 SmartSearch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal Document Intelligence • AI-Powered Knowledge Discovery</div>', unsafe_allow_html=True)

# Sidebar with modern styling
with st.sidebar:
    # Session info with glassmorphic card
    st.markdown(f"""
    <div class="session-card">
        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
            👤 Private Session
        </div>
        <div style="font-size: 0.85rem; opacity: 0.8;">
            ID: {st.session_state.user_session_id}
        </div>
        <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem;">
            🔒 Your data is completely isolated
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Management Section
    st.markdown("### 📁 Knowledge Base Manager")
    
    # Modern file uploader
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'mp3', 'wav', 'm4a'],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Text, Images, Audio",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div class="glass-card">
            <div style="color: white; font-weight: 600;">📎 {len(uploaded_files)} files selected</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View selected files", expanded=False):
            for file in uploaded_files:
                st.write(f"• {file.name}")
    
    # Process button with modern styling
    if st.button("⚡ Index Files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please select files first")
        else:
            result = process_uploaded_files(uploaded_files, st.session_state.user_paths)
            
            if result["processed"] > 0:
                st.success(f"✨ Successfully indexed {result['processed']} items!")
                
                # Update processed files list
                new_files = [f.name for f in uploaded_files]
                st.session_state.processed_files.extend(new_files)
                
                # Update index stats
                try:
                    original_index_path = os.environ.get("FAISS_INDEX_PATH")
                    original_meta_path = os.environ.get("FAISS_META_PATH")
                    
                    os.environ["FAISS_INDEX_PATH"] = st.session_state.user_paths["faiss_index"]
                    os.environ["FAISS_META_PATH"] = st.session_state.user_paths["faiss_metadata"]
                    
                    st.session_state.index_stats = get_index_stats()
                    
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
                st.error(f"Failed to process files: {error_msg}")
    
    # Knowledge Base Statistics
    st.markdown("---")
    st.markdown("### 📊 Knowledge Base Stats")
    
    stats = st.session_state.index_stats
    if stats.get("index_exists") and stats.get("total_items", 0) > 0:
        # Display total items with modern card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_items']}</div>
            <div class="metric-label">Total Documents</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Type distribution with icons
        if stats.get('type_distribution'):
            st.markdown("#### 📈 Distribution")
            for file_type, count in stats.get('type_distribution', {}).items():
                icon = "📄" if file_type == "document" else "🖼️" if file_type == "image" else "🎵"
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 0.5rem; margin: 0.3rem 0;">
                    <span style="color: white;">{icon} {file_type.title()}: <strong>{count}</strong></span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Upload files to build your knowledge base")
    
    # Current files display
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### 📚 Your Library")
        unique_files = list(set(st.session_state.processed_files))
        
        with st.expander(f"View {len(unique_files)} files", expanded=False):
            for file in sorted(unique_files):
                file_ext = Path(file).suffix.lower()
                icon = "📄" if file_ext in ['.pdf', '.doc', '.docx', '.txt'] else "🖼️" if file_ext in ['.jpg', '.jpeg', '.png'] else "🎵"
                st.write(f"{icon} {file}")
    
    # Audio debug (hidden by default)
    if st.checkbox("🔧 Debug Mode", key="debug_mode"):
        uploads_dir = st.session_state.user_paths["uploads"]
        if uploads_dir.exists():
            audio_files = list(uploads_dir.glob("*.mp3")) + list(uploads_dir.glob("*.wav")) + list(uploads_dir.glob("*.m4a"))
            st.write(f"Audio files: {len(audio_files)}")
            for af in audio_files[:3]:
                st.write(f"• {af.name}")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Smart Search", "📚 Document Library", "📈 Analytics"])

with tab1:
    # Search interface with modern design
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: white; font-weight: 600;">Ask Anything About Your Documents</h2>
            <p style="color: rgba(255,255,255,0.7); font-size: 1rem;">
                Powered by advanced AI • Natural language understanding • Multi-modal search
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if user has files
        has_files = len(st.session_state.processed_files) > 0
        
        if not has_files:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <h3 style="color: white;">🚀 Get Started</h3>
                <p style="color: rgba(255,255,255,0.7);">Upload and index your documents to enable AI-powered search</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Query input with modern styling
        query_text = st.text_area(
            "",
            placeholder="Try: 'What are the key findings?' or 'Show me all diagrams related to...'",
            height=100,
            key="main_query",
            label_visibility="collapsed"
        )
        
        # Search button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
        with col_btn2:
            search_button = st.button("🔮 Search with AI", 
                                      type="primary", 
                                      use_container_width=True,
                                      key="search_button",
                                      disabled=not has_files)
        
        if search_button:
            if not query_text:
                st.error("Please enter a question")
            else:
                with st.spinner("🔍 Searching across your knowledge base..."):
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
                        
                        # Display results with modern styling
                        st.markdown("### ✨ AI Response")
                        st.markdown(f"""
                        <div class="answer-box">
                            <div style="color: white; font-size: 1.1rem; line-height: 1.8;">
                                {response["answer_text"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show citations with enhanced display
                        if response.get("citations"):
                            st.markdown("### 📌 Sources & Evidence")
                            st.markdown("""
                            <div class="glass-card">
                                <p style="color: rgba(255,255,255,0.8);">
                                    💡 Click on citations below to view original sources
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            render_citations(response["citations"])
                        else:
                            st.info("No specific sources were referenced")
                        
                    except Exception as e:
                        st.error(f"Search failed. Please try again.")
                        logger.error(f"Search error: {e}")

with tab2:
    # Document Library with modern grid layout
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: white; font-weight: 600;">Your Document Library</h2>
        <p style="color: rgba(255,255,255,0.7);">
            Browse and manage your uploaded documents
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.processed_files:
        unique_files = list(set(st.session_state.processed_files))
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        doc_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.pdf', '.doc', '.docx', '.txt']])
        img_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        audio_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.mp3', '.wav', '.m4a']])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(unique_files)}</div>
                <div class="metric-label">Total Files</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{doc_count}</div>
                <div class="metric-label">📄 Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{img_count}</div>
                <div class="metric-label">🖼️ Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{audio_count}</div>
                <div class="metric-label">🎵 Audio</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Filter controls
        st.markdown("---")
        col1, col2 = st.columns([2, 4])
        
        with col1:
            file_type_filter = st.selectbox(
                "Filter by type",
                ["All files", "Documents", "Images", "Audio"],
                key="file_filter"
            )
        
        # Apply filter
        filtered_files = unique_files
        if file_type_filter == "Documents":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.pdf', '.doc', '.docx', '.txt']]
        elif file_type_filter == "Images":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        elif file_type_filter == "Audio":
            filtered_files = [f for f in unique_files if Path(f).suffix.lower() in ['.mp3', '.wav', '.m4a']]
        
        # Display filtered files
        st.markdown(f"### 📁 Files ({len(filtered_files)})")
        
        if filtered_files:
            # Create a grid layout for files
            cols = st.columns(2)
            for idx, file in enumerate(sorted(filtered_files)):
                file_ext = Path(file).suffix.lower()
                icon = "📄" if file_ext in ['.pdf', '.doc', '.docx', '.txt'] else "🖼️" if file_ext in ['.jpg', '.jpeg', '.png', '.bmp'] else "🎵"
                
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div class="file-card">
                        <div style="display: flex; align-items: center;">
                            <div style="font-size: 2rem; margin-right: 1rem;">{icon}</div>
                            <div style="flex: 1;">
                                <div style="font-weight: 600; font-size: 1rem;">{Path(file).stem}</div>
                                <div style="opacity: 0.7; font-size: 0.85rem;">{file_ext.upper()[1:]} • {Path(file).name}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No files match the selected filter")
    else:
        # Empty state with call to action
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
            <h3 style="color: white;">Your library is empty</h3>
            <p style="color: rgba(255,255,255,0.7);">
                Upload documents using the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Analytics Dashboard
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: white; font-weight: 600;">Analytics Dashboard</h2>
        <p style="color: rgba(255,255,255,0.7);">
            Insights into your knowledge base
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.processed_files:
        unique_files = list(set(st.session_state.processed_files))
        
        # File type distribution
        doc_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.pdf', '.doc', '.docx', '.txt']])
        img_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        audio_count = len([f for f in unique_files if Path(f).suffix.lower() in ['.mp3', '.wav', '.m4a']])
        
        # Create pie chart data
        import json
        chart_data = {
            "Documents": doc_count,
            "Images": img_count,
            "Audio": audio_count
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">📊 File Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for file_type, count in chart_data.items():
                if count > 0:
                    percentage = (count / len(unique_files)) * 100
                    st.markdown(f"""
                    <div style="margin: 1rem 0;">
                        <div style="color: white; margin-bottom: 0.5rem;">{file_type}: {count} ({percentage:.1f}%)</div>
                        <div style="background: rgba(255,255,255,0.1); border-radius: 20px; height: 30px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {percentage}%; border-radius: 20px; transition: width 0.5s ease;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: white;">📈 Quick Stats</h4>
                <div style="margin-top: 1rem;">
                    <p style="color: rgba(255,255,255,0.8);">
                        • Total indexed items: <strong>{}</strong><br/>
                        • Active session: <strong>{}</strong><br/>
                        • Last updated: <strong>Just now</strong>
                    </p>
                </div>
            </div>
            """.format(
                st.session_state.index_stats.get("total_items", 0),
                st.session_state.user_session_id[:10] + "..."
            ), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h3 style="color: white;">No data yet</h3>
            <p style="color: rgba(255,255,255,0.7);">
                Upload and index documents to see analytics
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 2rem;'>
        <p>🔮 SmartSearch AI • Powered by Advanced AI • © 2024</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">Your session is private and secure • All data is isolated</p>
    </div>
    """,
    unsafe_allow_html=True
)