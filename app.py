import logging
import os
from pathlib import Path
from time import time

import streamlit as st
from pydub import AudioSegment
from pydub.utils import which

from utils.app_utils import (
    cleanup_old_sessions,
    get_index_stats,
    get_session_info,
    get_user_session_id,
    get_user_specific_paths,
    process_uploaded_files,
    remove_uploaded_file,
    render_citations,
    restore_original_paths,
    set_session_paths,
)
from core.vector_indexer import get_index_stats
from qa.qa_engine import answer

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Create directories
directories = ["uploads", "thumbnails", "figures", "data", "data/pdf_images", "user_data"]
for directory in directories:
    Path(directory).mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SmartSearch AI - Multimodal Document Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔮"
)

# Initialize user session
if 'user_session_id' not in st.session_state:
    cleanup_old_sessions()
    st.session_state.user_session_id = get_user_session_id()
    st.session_state.user_paths = get_user_specific_paths(st.session_state.user_session_id)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'index_stats' not in st.session_state:
    st.session_state.index_stats = {"total_items": 0, "index_exists": False}
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None
if 'show_toast' not in st.session_state:
    st.session_state.show_toast = False
if 'toast_message' not in st.session_state:
    st.session_state.toast_message = ""
if 'toast_type' not in st.session_state:
    st.session_state.toast_type = "success"
if 'carousel_index' not in st.session_state:
    st.session_state.carousel_index = 0

# Enhanced Modern CSS with Premium Design and Better Contrast
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Root variables for beautiful color scheme */
    :root {
        --primary-bg: #0a0e27;
        --secondary-bg: #141b34;
        --tertiary-bg: #1a2341;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #ec4899;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-error: #ef4444;
        --text-primary: #ffffff;
        --text-secondary: #e2e8f0;
        --text-muted: #94a3b8;
        --glass-bg: rgba(20, 27, 52, 0.75);
        --glass-border: rgba(99, 102, 241, 0.2);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.3);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #141b34 50%, #0a0e27 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Animated background with better visibility */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: backgroundPulse 15s ease-in-out infinite;
    }
    
    @keyframes backgroundPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    /* Main container */
    .main {
        position: relative;
        z-index: 1;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Main header with better contrast */
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        filter: drop-shadow(0 0 30px rgba(99, 102, 241, 0.5));
        animation: headerFloat 6s ease-in-out infinite;
    }

    .emoji {
        -webkit-background-clip: initial !important;
        -webkit-text-fill-color: initial !important;
        background: none !important;
        filter: none !important;
        font-size: 4.5rem;
        margin-right: 0.5rem;
    }
    
    @keyframes headerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        opacity: 0.95;
    }
    
    /* Premium glass card with high contrast */
    .glass-card {
        background: linear-gradient(135deg, rgba(20, 27, 52, 0.85) 0%, rgba(26, 35, 65, 0.75) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 20px;
        border: 1.5px solid rgba(99, 102, 241, 0.25);
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 
            var(--shadow-lg),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
        transition: left 0.6s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 16px 48px rgba(99, 102, 241, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            0 0 0 1px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Premium answer box with excellent contrast */
    .answer-box {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.9) 0%, rgba(20, 27, 52, 0.85) 100%);
        backdrop-filter: blur(30px) saturate(200%);
        -webkit-backdrop-filter: blur(30px) saturate(200%);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        border: 2px solid transparent;
        box-shadow: 
            var(--shadow-lg),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: -2px;
        right: -2px;
        bottom: -2px;
        left: -2px;
        z-index: -1;
        border-radius: 24px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
        background-size: 300% 300%;
        animation: gradientRotate 8s ease infinite;
        opacity: 0.7;
    }
    
    @keyframes gradientRotate {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .answer-box::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.12), transparent);
        animation: shimmer 4s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* High contrast metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.85) 0%, rgba(20, 27, 52, 0.8) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1.5px solid rgba(99, 102, 241, 0.3);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
        box-shadow: 
            var(--shadow-md),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 12px 40px rgba(99, 102, 241, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.25),
            0 0 0 1px rgba(99, 102, 241, 0.4);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-secondary);
        font-weight: 600;
    }
    
    /* Premium button styles with high contrast */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 14px;
        padding: 0.9rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 20px rgba(99, 102, 241, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 8px 30px rgba(99, 102, 241, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* File uploader with better visibility */
    .stFileUploader {
        background: rgba(26, 35, 65, 0.6);
        border-radius: 16px;
        border: 2px dashed rgba(99, 102, 241, 0.4);
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(99, 102, 241, 0.7);
        background: rgba(26, 35, 65, 0.8);
    }
    
    /* Text area with high contrast */
    .stTextArea textarea {
        background: rgba(26, 35, 65, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1.5px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(226, 232, 240, 0.5) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(99, 102, 241, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        background: rgba(26, 35, 65, 0.9) !important;
    }
    
    /* Snackbar/Toast notification with better visibility */
    .toast-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.98) 0%, rgba(20, 27, 52, 0.98) 100%);
        backdrop-filter: blur(20px);
        color: #ffffff;
        padding: 1.2rem 1.8rem;
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(99, 102, 241, 0.5);
        z-index: 9999;
        animation: slideInRight 0.4s ease, fadeOut 0.4s ease 2.8s;
        border: 1.5px solid rgba(99, 102, 241, 0.4);
        font-weight: 500;
        min-width: 320px;
        font-size: 1rem;
    }
    
    .toast-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.2) 100%);
        border-color: rgba(16, 185, 129, 0.6);
        color: #ffffff;
        margin-top: 2.5rem;
    }
    
    .toast-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.2) 100%);
        border-color: rgba(239, 68, 68, 0.6);
        color: #ffffff;
    }
    
    .toast-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.2) 100%);
        border-color: rgba(245, 158, 11, 0.6);
        color: #ffffff;
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    /* Session card with premium look */
    .session-card {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.85) 0%, rgba(20, 27, 52, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1.5px solid rgba(99, 102, 241, 0.3);
        box-shadow: 
            var(--shadow-md),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* File card with better contrast */
    .file-card {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.8) 0%, rgba(20, 27, 52, 0.7) 100%);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        border: 1px solid rgba(99, 102, 241, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .file-card:hover {
        transform: translateX(8px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
    }
    
    /* Accordion/Expander with high contrast */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.85) 0%, rgba(20, 27, 52, 0.8) 100%) !important;
        backdrop-filter: blur(15px);
        border-radius: 14px !important;
        border: 1.5px solid rgba(99, 102, 241, 0.35) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1.2rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        border-color: rgba(99, 102, 241, 0.55) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.7) 0%, rgba(20, 27, 52, 0.65) 100%);
        backdrop-filter: blur(15px);
        border: 1.5px solid rgba(99, 102, 241, 0.25);
        border-top: none;
        border-radius: 0 0 14px 14px;
        padding: 1.8rem;
    }
    
    /* Info box with better visibility */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-left: 4px solid #6366f1;
        padding: 1.3rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: var(--text-secondary);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    
    /* Alert messages with high contrast */
    .stSuccess, .stError, .stWarning, .stInfo {
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1.5px solid;
        font-weight: 500;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #10b981;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.4);
        color: #ef4444;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
        color: #f59e0b;
    }
    
    .stInfo {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #6366f1;
    }
    
    /* Carousel container */
    .carousel-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        margin: 2rem 0;
    }
    
    .carousel-slide {
        display: none;
        animation: fadeIn 0.6s ease;
    }
    
    .carousel-slide.active {
        display: block;
    }
    
    .carousel-nav {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 1.5rem;
    }
    
    .carousel-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    
    .carousel-dot.active {
        background: #6366f1;
        width: 32px;
        border-radius: 6px;
    }
    
    /* Feature card for carousel */
    .feature-card {
        background: linear-gradient(135deg, rgba(26, 35, 65, 0.8) 0%, rgba(20, 27, 52, 0.75) 100%);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1.5px solid rgba(99, 102, 241, 0.25);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        display: block;
        filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));
    }
    
    .feature-title {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    /* Custom scrollbar with better colors */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 27, 52, 0.6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 10px;
        border: 2px solid rgba(20, 27, 52, 0.6);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #8b5cf6, #ec4899);
    }
    
    /* Remove button with premium styling */
    .remove-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(239, 68, 68, 0.3) !important;
    }
    
    .remove-btn:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.5) !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        margin: 2.5rem 0;
        border-radius: 2px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }
    
    /* Enhanced contrast for all text elements */
    p, span, div, label {
        color: var(--text-secondary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
    }
    
    /* Ensure file uploader text is visible */
    .stFileUploader label, .stFileUploader p {
        color: var(--text-secondary) !important;
    }
    
    /* Better visibility for select boxes */
    .stSelectbox > div > div {
        background: rgba(26, 35, 65, 0.7);
        border: 1.5px solid rgba(99, 102, 241, 0.3);
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# Toast notification function
def show_toast(message, toast_type="success"):
    st.session_state.show_toast = True
    st.session_state.toast_message = message
    st.session_state.toast_type = toast_type

# Display toast if needed
if st.session_state.show_toast:
    toast_class = f"toast-{st.session_state.toast_type}"
    icon = "✅" if st.session_state.toast_type == "success" else "❌" if st.session_state.toast_type == "error" else "⚠️"
    st.markdown(f"""
    <div class="toast-notification {toast_class}">
        <strong>{icon} {st.session_state.toast_message}</strong>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.show_toast = False

# Main header with animated gradient
st.markdown(
    """
    <div style="text-align:center;">
        <span class="emoji">🔮</span>
        <span class="main-header fade-in">SmartSearch AI</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="sub-header fade-in">Multimodal Document Intelligence • AI-Powered Knowledge Discovery</div>', unsafe_allow_html=True)

# Carousel for app features
carousel_features = [
    {
        "icon": "🚀",
        "title": "Lightning Fast Search",
        "description": "Experience instant semantic search across all your documents with state-of-the-art AI technology"
    },
    {
        "icon": "🎯",
        "title": "Multimodal Intelligence",
        "description": "Process PDFs, images, audio files, and more with advanced multimodal understanding"
    },
    {
        "icon": "🔒",
        "title": "Privacy First",
        "description": "Your data stays completely isolated in your private session - automatically cleaned after 24 hours"
    },
    {
        "icon": "💡",
        "title": "Smart Citations",
        "description": "Get accurate answers with traceable sources and evidence from your documents"
    }
]

# Carousel navigation
col_prev, col_carousel, col_next = st.columns([1, 8, 1])

with col_prev:
    if st.button("◀", key="carousel_prev", help="Previous feature"):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(carousel_features)
        st.rerun()

with col_carousel:
    current_feature = carousel_features[st.session_state.carousel_index]
    st.markdown(f"""
    <div class="carousel-container">
        <div class="carousel-slide active">
            <div class="feature-card fade-in">
                <span class="feature-icon">{current_feature['icon']}</span>
                <div class="feature-title">{current_feature['title']}</div>
                <div class="feature-description">{current_feature['description']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Carousel dots
    dots_html = '<div class="carousel-nav">'
    for i in range(len(carousel_features)):
        active_class = 'active' if i == st.session_state.carousel_index else ''
        dots_html += f'<div class="carousel-dot {active_class}"></div>'
    dots_html += '</div>'
    st.markdown(dots_html, unsafe_allow_html=True)

with col_next:
    if st.button("▶", key="carousel_next", help="Next feature"):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(carousel_features)
        st.rerun()

# Auto-advance carousel every 5 seconds (optional)
import time

if 'last_carousel_update' not in st.session_state:
    st.session_state.last_carousel_update = time.time()

# App Info Accordion Section
with st.expander("📖 **How to Use SmartSearch AI** - Click to expand and learn", expanded=False):
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📤</span>
            <div class="feature-title">1. Upload Files</div>
            <div class="feature-description">
                Drag and drop your documents, images, or audio files. 
                Supports PDF, Word, images (JPG, PNG), and audio formats (MP3, WAV, M4A).
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">2. AI Processing</div>
            <div class="feature-description">
                Our AI automatically indexes your content with 
                multimodal understanding for intelligent retrieval.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">💬</span>
            <div class="feature-title">3. Ask Questions</div>
            <div class="feature-description">
                Query your knowledge base naturally and get 
                intelligent answers with source citations and original documents.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🔒 Privacy First:</strong> Your session is completely isolated. All your data is private and automatically cleaned up after 24 hours of inactivity.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Smart Features:</strong> Advanced multimodal AI, automatic chunking, semantic search, and citation tracking for reliable answers.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>⚡ Supported Formats:</strong> PDF, DOC/DOCX, TXT, JPG/JPEG/PNG images, MP3/WAV/M4A audio files. Multiple files can be processed simultaneously.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Unified Interface Layout
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;" class="fade-in">
    <h2 style="color: var(--text-primary); font-weight: 700; font-size: 2.2rem; margin-bottom: 0.5rem;">
        Ask Anything About Your Documents
    </h2>
    <p style="color: var(--text-secondary); font-size: 1.15rem; margin-top: 0.8rem; opacity: 0.95;">
        Upload files and ask questions in one place • Private session • Real-time indexing
    </p>
</div>
""", unsafe_allow_html=True)

# Main content area with unified interface
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    # Knowledge Base Panel
    st.markdown("### 📁 Your Knowledge Base")
    
    # Session info
    session_info = get_session_info()
    st.markdown(f"""
    <div class="session-card fade-in">
        <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.8rem; color: var(--text-primary);">
            👤 Private Session
        </div>
        <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.8rem; font-family: 'Courier New', monospace; background: rgba(99, 102, 241, 0.1); padding: 0.5rem; border-radius: 6px;">
            ID: {st.session_state.user_session_id[:16]}...
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #10b981;">●</span> Isolated & Secure • {session_info['current_session_size_mb']:.1f} MB used
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'mp3', 'wav', 'm4a'],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Text, Images, Audio",
        key="main_file_uploader"
    )
    
    # Process files immediately when uploaded
    if uploaded_files and st.session_state.get('last_uploaded_files') != [f.name for f in uploaded_files]:
        st.session_state.last_uploaded_files = [f.name for f in uploaded_files]
        with st.spinner("🔄 Indexing your files..."):
            result = process_uploaded_files(uploaded_files, st.session_state.user_paths)
            
            if result["processed"] > 0:
                show_toast(f"✨ Indexed {result['processed']} items from {len(uploaded_files)} files!", "success")
                st.success(f"✅ Successfully indexed {result['processed']} items from {len(uploaded_files)} files!")
                
                # Update processed files list
                new_files = [f.name for f in uploaded_files]
                st.session_state.processed_files.extend(new_files)
                st.session_state.processed_files = list(set(st.session_state.processed_files))
                
                # Update index stats
                try:
                    original_paths = {
                        "FAISS_INDEX_PATH": os.environ.get("FAISS_INDEX_PATH"),
                        "FAISS_META_PATH": os.environ.get("FAISS_META_PATH"),
                        "ID_MAP_PATH": os.environ.get("ID_MAP_PATH")
                    }
                    
                    set_session_paths(st.session_state.user_paths)
                    st.session_state.index_stats = get_index_stats()
                    restore_original_paths(original_paths)
                        
                except Exception as e:
                    logger.error(f"Failed to get index stats: {e}")
                    st.session_state.index_stats = {"total_items": result["processed"], "index_exists": True}
                
                st.rerun()
            else:
                error_msg = result.get("errors", ["Unknown error"])
                show_toast(f"Failed to process files", "error")
                st.error(f"❌ Failed to process files: {error_msg}")
    
    # Current files display with remove option
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("#### 📚 Your Documents")
        
        unique_files = list(set(st.session_state.processed_files))
        stats = st.session_state.index_stats
        
        # Quick stats with glossy cards
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown(f"""
            <div class="metric-card fade-in" style="height: 110px; padding: 1.2rem;">
                <div class="metric-value" style="font-size: 2.2rem;">{len(unique_files)}</div>
                <div class="metric-label" style="font-size: 0.8rem;">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown(f"""
            <div class="metric-card fade-in" style="height: 110px; padding: 1.2rem;">
                <div class="metric-value" style="font-size: 2.2rem;">{stats.get('total_items', 0)}</div>
                <div class="metric-label" style="font-size: 0.8rem;">Indexed Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        # File list with remove buttons
        st.markdown("##### Manage Your Files:")
        for file in sorted(unique_files):
            file_ext = Path(file).suffix.lower()
            icon = "📄" if file_ext in ['.pdf', '.doc', '.docx', '.txt'] else "🖼️" if file_ext in ['.jpg', '.jpeg', '.png'] else "🎵"
            
            col_file, col_remove = st.columns([4, 1])
            with col_file:
                st.markdown(f"""
                <div class="file-card">
                    <span style="font-size: 1.8rem;">{icon}</span>
                    <div>
                        <div style="color: var(--text-primary); font-size: 1rem; font-weight: 600; margin-bottom: 0.2rem;">
                            {file}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.8rem;">
                            {file_ext.upper()[1:]} File
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_remove:
                if st.button("🗑️", key=f"remove_{file}", help=f"Remove {file}"):
                    with st.spinner(f"Removing {file}..."):
                        result = remove_uploaded_file(file, st.session_state.user_paths)
                        if result >= 0:
                            st.session_state.processed_files = [f for f in st.session_state.processed_files if f != file]
                            st.session_state.index_stats["total_items"] = result
                            show_toast(f"🗑️ Removed {file}", "success")
                            st.success(f"✅ Removed {file}")
                            st.rerun()
                        else:
                            show_toast(f"Failed to remove {file}", "error")
                            st.error(f"❌ Failed to remove {file}")
    else:
        st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; padding: 3.5rem 2rem;">
            <div style="font-size: 5rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));">📚</div>
            <h3 style="color: var(--text-primary); margin-bottom: 0.8rem; font-weight: 700;">No documents yet</h3>
            <p style="color: var(--text-secondary); font-size: 1.05rem; line-height: 1.6;">
                Upload files above to build your AI-powered knowledge base
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Query Panel
    st.markdown("### 💬 Ask Your Question")
    
    # Check if user has files
    has_files = len(st.session_state.processed_files) > 0
    
    if not has_files:
        st.markdown("""
        <div class="glass-card fade-in" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));">📥</div>
            <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">Upload Files to Get Started</h3>
            <p style="color: var(--text-secondary); font-size: 1.1rem; line-height: 1.7;">
                Add documents to your knowledge base on the left to enable AI-powered intelligent search
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Query input
    query_text = st.text_area(
        "",
        placeholder="Ask anything about your uploaded documents...\nExamples:\n- 'What are the key findings in the reports?'\n",
        height=120,
        key="main_query",
        label_visibility="collapsed",
        disabled=not has_files
    )
    
    # Search button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        search_button = st.button("🔮 Generate Answer", 
                                  type="primary", 
                                  use_container_width=True,
                                  key="search_button",
                                  disabled=not has_files
        )
    
    # Display previous answer if exists
    if st.session_state.last_answer and st.session_state.last_question:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <h4 style="color: var(--text-primary); font-weight: 700; margin-bottom: 0.8rem; font-size: 1.3rem;">💡 AI Answer</h4>
            <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #6366f1;">
                <p style="color: var(--text-secondary); font-style: italic; font-size: 1.05rem; margin: 0;">"{st.session_state.last_question}"</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        response = st.session_state.last_answer
        
        # Display results with modern styling
        st.markdown(f"""
        <div class="answer-box fade-in">
            <div style="color: var(--text-primary); font-size: 1.15rem; line-height: 1.9; font-weight: 400;">
                {response["answer_text"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show citations with enhanced display
        if response.get("citations"):
            st.markdown("#### 📌 Sources & Evidence")
            st.markdown("""
            <div class="glass-card">
                <p style="color: var(--text-secondary); font-size: 1rem;">
                    💡 Click on citations below to view original sources and evidence
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            render_citations(response["citations"])
        else:
            st.info("ℹ️ No specific sources were referenced in this answer")
    
    # Handle new search
    if search_button and query_text.strip():
        with st.spinner("🔍 Searching across your knowledge base..."):
            try:
                # Store original environment variables
                original_paths = {
                    "FAISS_INDEX_PATH": os.environ.get("FAISS_INDEX_PATH"),
                    "FAISS_META_PATH": os.environ.get("FAISS_META_PATH"),
                    "ID_MAP_PATH": os.environ.get("ID_MAP_PATH")
                }
                
                # Set user-specific paths for search
                set_session_paths(st.session_state.user_paths)
                
                # Perform search
                response = answer(query_text, k=6)
                
                # Restore original paths
                restore_original_paths(original_paths)
                
                # Store in session state
                st.session_state.last_question = query_text
                st.session_state.last_answer = response
                
                show_toast("✨ Answer generated successfully!", "success")
                st.rerun()
                
            except Exception as e:
                show_toast("⚠️ Search failed. Please try again.", "error")
                st.error("❌ Search failed. Please try again.")
                logger.error(f"Search error: {e}")
                import traceback
                logger.error(f"Full search error traceback: {traceback.format_exc()}")

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: var(--text-muted); font-size: 0.95rem; margin-top: 3rem; padding: 2.5rem 0;'>
        <p style="font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 0.8rem; font-weight: 600;">
            <strong>🔮 SmartSearch AI</strong> • Unified Interface for Multi Modal Retrieval
        </p>
    </div>
    """,
    unsafe_allow_html=True
)