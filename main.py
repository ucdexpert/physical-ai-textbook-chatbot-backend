from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from agent import textbook_agent  # your Agent instance

from agents.run import Runner  # Import Runner

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Physical AI RAG API...")
    yield
    # Shutdown
    print("Shutting down Physical AI RAG API...")

app = FastAPI(title="Physical AI RAG API", lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3001",
    "https://physical-ai-textbook-frontend-raz-c.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://physical-ai-textbook-frontend-raz-c.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def main():
    return {"message": "wellcome to chatbot api"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Physical AI Agent"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Correct way to call the agent using Runner.run
        response = await Runner.run(textbook_agent, request.query)

        # Convert the agent response to text
        if hasattr(response, "final_output"):
            final_answer = response.final_output
        else:
            final_answer = str(response)

        return ChatResponse(answer=final_answer)

    except Exception as e:
        error_msg = str(e)
        print("Agent Error:", error_msg)
        
        if "429" in error_msg or "Resource exhausted" in error_msg:
             raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
             
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)