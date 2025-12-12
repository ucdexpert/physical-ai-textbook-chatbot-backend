import os
import uuid
import logging
import frontmatter
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

COLLECTION_NAME = "physical_ai_book"

# ---------------------------
# Initialization
# ---------------------------

def init_services():
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    if not api_key_gemini:
        raise ValueError("GEMINI_API_KEY is missing in .env")
        
    api_key_qdrant = os.getenv("QDRANT_API_KEY")
    url_qdrant = os.getenv("QDRANT_URL")
    
    genai.configure(api_key=api_key_gemini)
    q_client = QdrantClient(
        url=url_qdrant, 
        api_key=api_key_qdrant,
        timeout=30
    )
    
    return q_client

# Default Docs Path (Change this if your folder is different)
DOCS_DIR = os.getenv("DOCS_DIR", str(Path(__file__).resolve().parent.parent / "docs"))

def get_docs_path() -> Path:
    logger.info(f"Looking for documents in: {DOCS_DIR}")
    return Path(DOCS_DIR)

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Splits text into manageable chunks preserving paragraph structure."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ---------------------------
# Main Pipeline
# ---------------------------

def main():
    logger.info("Starting Ingestion...")
    try:
        qdrant = init_services()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    docs_dir = get_docs_path()
    if not docs_dir.exists():
        logger.error(f"Docs directory not found at: {docs_dir}")
        return

    # 1. Read and Chunk files
    all_points = []
    files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))
    logger.info(f"Found {len(files)} markdown files.")

    for file_path in files:
        try:
            post = frontmatter.load(file_path)
            content = post.content
            metadata = post.metadata
            
            title = metadata.get("title", file_path.stem)
            slug = metadata.get("slug", file_path.stem)
            
            if not content.strip():
                continue

            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                # 2. Embed
                emb_result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                )
                vector = emb_result["embedding"]
                
                # 3. Create Point
                point = rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "title": title,
                        "slug": slug,
                        "heading": f"Part {i+1}",
                        "filename": file_path.name
                    }
                )
                all_points.append(point)
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    if not all_points:
        logger.warning("No points created. Exiting.")
        return

    # 4. Upsert to Qdrant
    logger.info(f"Upserting {len(all_points)} chunks to Qdrant...")
    
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
    )
    
    batch_size = 50
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i : i + batch_size]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Uploaded batch {i} - {i+len(batch)}")

    logger.info("Ingestion Complete!")

if __name__ == "__main__":
    main()