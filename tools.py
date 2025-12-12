import os
import logging
from typing import List
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from agents.tool import function_tool

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tools")

# ---------------------------
# Pydantic Models (STRICT) - Pydantic v2 compatible
# ---------------------------

class BookChunk(BaseModel):
    text: str = Field(..., description="The content text from the book")
    title: str = Field(..., description="The chapter title")
    heading: str = Field(..., description="The section heading")
    slug: str = Field(..., description="The URL-friendly identifier")
    score: float = Field(..., description="The relevance score")

    class Config:
        # For backward compatibility if needed
        from_attributes = True


# ---------------------------
# Initialization
# ---------------------------

def get_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=30,
    )

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment variables")
    genai.configure(api_key=api_key)

# ---------------------------
# Tool 1: Search
# ---------------------------

@function_tool
def search_book_content(query: str, top_k: int = 5) -> List[BookChunk]:
    """
    Search the Physical AI textbook for relevant content based on a query.
    """
    if not query or not query.strip():
        return []

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

        # Create BookChunk instance with validation
        book_chunk = BookChunk(
            text=payload.get("text", ""),
            title=payload.get("title", "Unknown Chapter"),
            heading=payload.get("heading", "Section"),
            slug=payload.get("slug", ""),
            score=float(point.score)
        )
        results.append(book_chunk)

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
        # Using model_dump() for Pydantic v2 compatibility
        item_dict = item.model_dump()
        formatted_text += (
            f"--- Source {i} ---\n"
            f"Chapter: {item_dict.get('title', 'N/A')}\n"
            f"Section: {item_dict.get('heading', 'N/A')}\n"
            f"Text: {item_dict.get('text', 'N/A')}\n\n"
        )

    return formatted_text