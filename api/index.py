from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from agent import textbook_agent  # make sure agent.py is at root
from agents.run import Runner       # ensure agents/run.py exists

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Physical AI RAG API...")
    yield
    print("Shutting down Physical AI RAG API...")

app = FastAPI(
    title="Physical AI RAG API",
    lifespan=lifespan,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://physical-ai-textbook-frontend-raz-c.vercel.app",
        "http://localhost:3000",  # For local development
        "http://localhost:3001",  # Alternative local dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user query to process")

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Welcome to Physical AI Textbook Chatbot API"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Physical AI Agent", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        # Run the agent with the user query
        response = await Runner.run(textbook_agent, request.query)

        # Handle different response formats from the agent
        if hasattr(response, "final_output"):
            final_answer = response.final_output
        elif hasattr(response, "output"):
            final_answer = response.output
        elif isinstance(response, str):
            final_answer = response
        elif hasattr(response, "__str__"):
            final_answer = str(response)
        else:
            final_answer = "I couldn't process your request. Please try again."

        return ChatResponse(answer=final_answer)
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        err = str(e)
        print("Agent Error:", err)
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        if "429" in err or "Resource exhausted" in err or "rate limit" in err.lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        elif "timeout" in err.lower() or "timed out" in err.lower():
            raise HTTPException(status_code=408, detail="Request timeout")
        else:
            raise HTTPException(status_code=500, detail=f"Internal server error: {err}")
