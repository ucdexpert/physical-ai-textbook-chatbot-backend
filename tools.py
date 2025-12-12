import os
import logging
from typing import List
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

from pydantic import BaseModel
from agents.tool import function_tool

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tools")

# ---------------------------
# Pydantic Models (STRICT)
# ---------------------------

class BookChunk(BaseModel):
    text: str
    title: str
    heading: str
    slug: str
    score: float


# ---------------------------
# Initialization
# ---------------------------

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30,
    )

def configure_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------
# Tool 1: Search
# ---------------------------

@function_tool
def search_book_content(query: str, top_k: int = 5) -> List[BookChunk]:
    """
    Search the Physical AI textbook for relevant content based on a query.
    """
    configure_gemini()
    client = get_qdrant_client()
    
    logger.info(f"Searching book for: {query}")

    # 1. Embed the query using Gemini
    try:
        embedding_resp = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = embedding_resp["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

    # 2. Search Qdrant
    try:
        search_results = client.query_points(
            collection_name="physical_ai_book",
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

    # 3. Convert to typed models
    results: List[BookChunk] = []
    for point in search_results:
        payload = point.payload or {}

        results.append(
            BookChunk(
                text=payload.get("text", ""),
                title=payload.get("title", "Unknown Chapter"),
                heading=payload.get("heading", "Section"),
                slug=payload.get("slug", ""),
                score=float(point.score)
            )
        )
    
    return results

# ---------------------------
# Tool 2: Format
# ---------------------------

@function_tool
def format_context_for_answer(results: List[BookChunk]) -> str:
    """
    Format the retrieved book content into a readable string with citations.
    """
    if not results:
        return "No relevant information found in the textbook."

    formatted_text = "REFERENCES FROM TEXTBOOK:\n\n"
    
    for i, item in enumerate(results, 1):
        formatted_text += (
            f"--- Source {i} ---\n"
            f"Chapter: {item.title}\n"
            f"Section: {item.heading}\n"
            f"Text: {item.text}\n\n"
        )
    
    return formatted_text