# indexer.py
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from constants import FAISS_INDEX_PATH, FAISS_META_PATH, ID_MAP_PATH
from embedding import embed_texts

logger = logging.getLogger(__name__)

def _ensure_dir(path):
    """Ensure directory exists for file path"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_meta(items):
    """Save metadata with error handling"""
    try:
        _ensure_dir(FAISS_META_PATH)
        # Convert non-serializable objects to strings
        serializable_items = []
        for item in items:
            serializable_item = {}
            for key, value in item.items():
                try:
                    # Try to serialize as-is
                    json.dumps(value)
                    serializable_item[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serializable_item[key] = str(value)
            serializable_items.append(serializable_item)
        
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"items": serializable_items}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata for {len(items)} items")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise

def load_meta():
    """Load metadata with error handling"""
    if not os.path.exists(FAISS_META_PATH):
        return []
    try:
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("items", [])
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return []

def save_id_map(id_map):
    """Save mapping from index positions to UUID"""
    try:
        _ensure_dir(ID_MAP_PATH)
        with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save ID map: {e}")

def load_id_map():
    """Load ID map"""
    if not os.path.exists(ID_MAP_PATH):
        return {}
    try:
        with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ID map: {e}")
        return {}

def load_index_and_meta():
    """Load both FAISS index and metadata"""
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            return None, []
        
        index = faiss.read_index(FAISS_INDEX_PATH)
        items = load_meta()
        
        # Validate that index size matches metadata
        if index.ntotal != len(items):
            logger.warning(f"Index size ({index.ntotal}) doesn't match metadata size ({len(items)})")
            # Use the minimum of both to avoid errors
            min_size = min(index.ntotal, len(items))
            if min_size == 0:
                return None, []
            
        logger.info(f"Loaded index with {index.ntotal} items")
        return index, items
        
    except Exception as e:
        logger.error(f"Failed to load index and metadata: {e}")
        return None, []

def build_index_from_items(items):
    """
    Build FAISS index from items with robust error handling
    """
    if not items:
        logger.warning("No items to index")
        return None, []
    
    # Ensure each item has required fields
    for item in items:
        if "uuid" not in item:
            item["uuid"] = str(uuid4())
        if "content" not in item:
            item["content"] = ""
        if "type" not in item:
            item["type"] = "unknown"
    
    # Generate embeddings
    texts = [item.get("content", "") for item in items]
    embeddings = embed_texts(texts)
    
    if len(embeddings) == 0 or embeddings.shape[0] != len(items):
        logger.error("Embedding generation failed or size mismatch")
        return None, []
    
    try:
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save index and metadata
        _ensure_dir(FAISS_INDEX_PATH)
        faiss.write_index(index, FAISS_INDEX_PATH)
        save_meta(items)
        
        # Create and save ID map
        id_map = {str(i): items[i]["uuid"] for i in range(len(items))}
        save_id_map(id_map)
        
        logger.info(f"Built index with {len(items)} items, dimension {dimension}")
        return index, items
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return None, []

def add_items(new_items):
    """Add new items by combining with existing and rebuilding"""
    try:
        existing_items = load_meta()
        combined_items = existing_items + new_items
        
        # Remove duplicates based on content hash
        seen_contents = set()
        unique_items = []
        for item in combined_items:
            content_hash = hash(item.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_items.append(item)
        
        return build_index_from_items(unique_items)
        
    except Exception as e:
        logger.error(f"Failed to add items: {e}")
        return None, []

def get_index_stats():
    """Get statistics about the index"""
    index, items = load_index_and_meta()
    if index is None:
        return {"total_items": 0, "index_exists": False}
    
    type_distribution = {}
    for item in items:
        item_type = item.get("type", "unknown")
        type_distribution[item_type] = type_distribution.get(item_type, 0) + 1
    
    return {
        "total_items": len(items),
        "index_exists": True,
        "index_size": index.ntotal,
        "type_distribution": type_distribution
    }

def clear_index():
    """Delete all index and metadata files"""
    deleted_files = []
    files_to_delete = [FAISS_INDEX_PATH, FAISS_META_PATH, ID_MAP_PATH]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                logger.info(f"Deleted {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
    
    # Clear various directories
    directories_to_clear = ["uploads", "thumbnails", "figures", "data/pdf_images"]
    
    for dir_path in directories_to_clear:
        if os.path.exists(dir_path):
            try:
                for file in Path(dir_path).glob("*"):
                    if file.is_file():
                        file.unlink()
                logger.info(f"Cleared {dir_path} directory")
            except Exception as e:
                logger.error(f"Failed to clear {dir_path}: {e}")
    
    return {
        "deleted_files": deleted_files,
        "message": f"Successfully cleared {len(deleted_files)} index files and directories"
    }

def remove_items_by_uuid(uuids_to_remove):
    """
    Remove items from metadata and rebuild index
    """
    try:
        items = load_meta()
        remaining_items = [item for item in items if item.get("uuid") not in uuids_to_remove]
        removed_count = len(items) - len(remaining_items)
        
        if removed_count > 0:
            build_index_from_items(remaining_items)
            return {
                "removed": removed_count, 
                "message": f"Successfully removed {removed_count} items"
            }
        return {
            "removed": 0, 
            "message": "No items removed (UUIDs not found)"
        }
        
    except Exception as e:
        logger.error(f"Failed to remove items: {e}")
        return {
            "removed": 0, 
            "message": f"Error removing items: {str(e)}"
        }

def list_indexed_items():
    """
    Return all items with minimal info for UI listing
    """
    items = load_meta()
    simplified_items = []
    for item in items:
        simplified_items.append({
            "uuid": item.get("uuid"),
            "type": item.get("type", "unknown"),
            "source": item.get("source", "Unknown"),
            "content_preview": item.get("content", "")[:100] + "..." if len(item.get("content", "")) > 100 else item.get("content", ""),
            "page": item.get("page"),
            "timestamp": item.get("start_sec") if item.get("type") == "audio" else None
        })
    return simplified_items

def get_item_by_uuid(uuid):
    """Get specific item by UUID"""
    items = load_meta()
    for item in items:
        if item.get("uuid") == uuid:
            return item
    return None

def rebuild_index():
    """Force rebuild index from existing metadata"""
    items = load_meta()
    if items:
        return build_index_from_items(items)
    return None, []