import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from indexer import get_index_stats
from ingest_pipeline import ingest_paths, render_citations
from qa_openrouter import answer
from rag_system.audio_processor import transcribe_audio
from rag_system.groq_processor import get_image_caption

directories = ["uploads", "thumbnails", "figures", "data", "data/pdf_images"]

for directory in directories:
    Path(directory).mkdir(exist_ok=True)
    #print(f"Created directory: {directory}")

#print("All directories created successfully!")

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not uploaded_files:
        return {"processed": 0, "errors": []}
    
    # Create uploads directory
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        try:
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(str(file_path))
        except Exception as e:
            logger.error(f"Failed to save {uploaded_file.name}: {e}")
    
    # Ingest files
    if saved_paths:
        with st.spinner(f"Processing {len(saved_paths)} files..."):
            results = ingest_paths(saved_paths)
        
        return {
            "processed": len(results) if results else 0,
            "total_files": len(saved_paths)
        }
    
    return {"processed": 0, "errors": ["Failed to save files"]}
    
def process_query_context(query_text, context_files):
    """Process query with optional context files"""
    if not context_files:
        return query_text
    
    context_descriptions = []
    
    for file in context_files:
        file_ext = Path(file.name).suffix.lower()
        
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            # Process image
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
                f.write(file.getvalue())
                temp_path = f.name
            
            try:
                caption = get_image_caption(temp_path)
                context_descriptions.append(f"Image context: {caption}")
            finally:
                os.unlink(temp_path)
                
        elif file_ext in ['.mp3', '.wav', '.m4a']:
            # Process audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
                f.write(file.getvalue())
                temp_path = f.name
            
            try:
                transcription = transcribe_audio(temp_path)
                context_descriptions.append(f"Audio context: {transcription}")
            finally:
                os.unlink(temp_path)
                
        elif file_ext in ['.pdf', '.doc', '.docx', '.txt']:
            # For documents, we'll just mention they're included
            context_descriptions.append(f"Document reference: {file.name}")
    
    if context_descriptions:
        enhanced_query = f"{query_text}\n\nAdditional context:\n" + "\n".join(context_descriptions)
        return enhanced_query
    
    return query_text

# If you still get import errors, use this fallback:
try:
    from indexer import clear_index
except ImportError:
    # Define a fallback clear_index function
    def clear_index():
        import os
        files_to_delete = [
            "data/faiss_index.index",
            "data/faiss_metadata.json", 
            "data/id_map.json"
        ]
        deleted_files = []
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
        return {"deleted_files": deleted_files}

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

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'index_stats' not in st.session_state:
    st.session_state.index_stats = get_index_stats()

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
        background-color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .file-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8fafc;
    }
    .connection-item {
        background-color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    .success-message {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🔍 SmartSearch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal Document Intelligence • Search Across Documents, Images & Audio</div>', unsafe_allow_html=True)

# Sidebar - File Management
st.sidebar.header("📁 Document Management")

# File upload section with improved UX
st.sidebar.markdown("### Add Documents to Your Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop files here",
    type=['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'mp3', 'wav', 'm4a'],
    accept_multiple_files=True,
    help="Supported formats: PDF, Word, Text, Images, Audio files",
    label_visibility="collapsed"
)

if uploaded_files:
    st.sidebar.markdown(f"**Selected files:** {len(uploaded_files)}")
    for file in uploaded_files:
        st.sidebar.write(f"• {file.name}")

if st.sidebar.button("🚀 Process & Index Files", type="primary", use_container_width=True) and uploaded_files:
    with st.spinner("Processing your documents..."):
        result = process_uploaded_files(uploaded_files)
    
    if result["processed"] > 0:
        st.sidebar.markdown(f"""
        <div class="success-message">
            ✅ Successfully processed {result['processed']} items from {result['total_files']} files
        </div>
        """, unsafe_allow_html=True)
        st.session_state.processed_files.extend([f.name for f in uploaded_files])
        st.session_state.index_stats = get_index_stats()
    else:
        st.sidebar.error("❌ Failed to process files")

# Index statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Knowledge Base Status")

stats = st.session_state.index_stats
if stats["index_exists"] and stats["total_items"] > 0:
    st.sidebar.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 2rem; font-weight: bold;">{stats['total_items']}</div>
        <div>Total Documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Type distribution
    for file_type, count in stats.get('type_distribution', {}).items():
        icon = "📄" if file_type == "document" else "🖼️" if file_type == "image" else "🎵"
        st.sidebar.write(f"{icon} {file_type.title()}: **{count}**")
else:
    st.sidebar.info("💡 Add documents to build your knowledge base")

# Main content area
tab1, tab2 = st.tabs(["🔍 Smart Search", "📚 Your Documents"])

with tab1:
    st.markdown("""
    <div class="section-header">Ask Anything About Your Documents</div>
    <p style="color: #64748b; font-size: 1.1rem;">
    Ask questions in plain English. Upload additional files for context if needed.
    </p>
    """, unsafe_allow_html=True)
    
    # Simple query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_text = st.text_area(
            "Your question:",
            placeholder="e.g., 'What are the main points discussed in the meeting about the project timeline?' or 'Show me diagrams related to the system architecture'",
            height=100,
            key="main_query"
        )
    
    with col2:
        # Simple settings
        search_scope = st.selectbox(
            "Search scope:",
            ["All content", "Documents only", "Images only", "Audio only"]
        )

    
    # Search button
    if st.button("🔍 Search Knowledge Base", type="primary", use_container_width=True) and query_text:
        with st.spinner("Searching across your documents..."):
            try:
                # Map scope to filter type
                scope_map = {
                    "All content": None,
                    "Documents only": "document",
                    "Images only": "image", 
                    "Audio only": "audio"
                }
                
                # Perform search
                response = answer(query_text, k=6)
                
                # Display results
                st.markdown("### 💡 Answer")
                st.markdown(f"""
                <div class="answer-box">
                    {response["answer_text"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                if response.get("citations"):
                    st.markdown("### 📋 Sources & Evidence")
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
    All documents currently in your knowledge base
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
            ["All files", "Documents", "Images", "Audio"]
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
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div style="flex: 1;">
                            {icon} <strong>{file}</strong>
                        </div>
                        <div>
                            <button style="background: #3b82f6; color: white; border: none; padding: 0.3rem 0.8rem; border-radius: 5px; cursor: pointer;">
                                Search in this file
                            </button>
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
    "SmartSearch AI • Multimodal Document Intelligence • "
    "© 2024 All rights reserved"
    "</div>",
    unsafe_allow_html=True
)
