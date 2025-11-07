import os

import pymupdf4llm

from processors.image_processor import get_image_caption


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF using pymupdf4llm.
    Returns a list of dicts with page number and content.
    """
    pages = pymupdf4llm.to_markdown(pdf_path, write_images=False, page_chunks=True)
    chunks = []

    for page_data in pages:
        text = page_data.get("text", "").strip()
        if text:
            chunks.append({
                "content": text,
                "type": "document",
                "source": os.path.basename(pdf_path),
                "page": page_data.get("metadata", {}).get("page", None)
            })

    return chunks

def extract_pdf_images(path, out_dir="figures"):
    """Extract images from PDF and store them in out_dir"""
    out_dir = os.path.join(out_dir, os.path.basename(path).replace(".pdf", ""))
    os.makedirs(out_dir, exist_ok=True)

    pymupdf4llm.to_markdown(
        path,
        write_images=True,
        image_path=out_dir,
        image_format="png",
        dpi=300,
        page_chunks=True
    )

    images = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith(".png"):
            try:
                parts = fname.split("-")
                page = int(parts[-2]) + 1
                fig = int(parts[-1].split(".")[0]) + 1
                images.append({
                    "path": os.path.join(out_dir, fname),
                    "page": page,
                    "figure": fig,
                    "filename": fname
                })
            except Exception as e:
                print(f"Warning: could not parse metadata from {fname}: {e}")
    return images

def extract_text_chunks(file_path: str):
    """
    Main function to extract text and images from documents
    Supports PDF, DOC, DOCX, TXT files with Groq image captioning
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    docs = []

    try:
        if file_ext == '.pdf':
            # Extract text from PDF
            text_chunks = extract_pdf_text(file_path)
            for chunk in text_chunks:
                docs.append({
                    "content": chunk["content"],
                    "type": "document",
                    "source": os.path.basename(file_path),
                    "page": chunk.get("page")
                })

            # Extract and caption images from PDF
            try:
                image_chunks = extract_pdf_images(file_path)
                for img in image_chunks:
                    # Generate caption using Groq
                    caption = get_image_caption(img["path"])
                    docs.append({
                        "content": caption,
                        "type": "image",
                        "source": os.path.basename(file_path),
                        "page": img["page"],
                        "caption": caption,
                        "image_path": img["path"],
                        "figure": img["figure"]
                    })
            except Exception as e:
                print(f"Warning: Could not extract/caption images from PDF: {e}")

        elif file_ext in ['.docx', '.doc']:
            # For Word documents
            try:
                import docx
                doc = docx.Document(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                content = "\n".join(full_text)
                if content.strip():
                    docs.append({
                        "content": content,
                        "type": "document", 
                        "source": os.path.basename(file_path)
                    })
            except ImportError:
                # Fallback
                with open(file_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                if content.strip():
                    docs.append({
                        "content": content[:10000],
                        "type": "document",
                        "source": os.path.basename(file_path)
                    })

        elif file_ext == '.txt':
            # Simple text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content.strip():
                docs.append({
                    "content": content,
                    "type": "document",
                    "source": os.path.basename(file_path)
                })

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        docs.append({
            "content": f"Error processing file: {str(e)}",
            "type": "document",
            "source": os.path.basename(file_path),
            "error": True
        })

    return docs