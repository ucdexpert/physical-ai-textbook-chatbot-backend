from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from agent import textbook_agent  # make sure agent.py is at root
from agents.run import Runner       # ensure agents/run.py exists

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Physical AI RAG API...")
    yield
    print("Shutting down Physical AI RAG API...")

app = FastAPI(title="Physical AI RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://physical-ai-textbook-frontend-raz-c.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Welcome to chatbot API"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Physical AI Agent"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        response = await Runner.run(textbook_agent, request.query)
        final_answer = getattr(response, "final_output", str(response))
        return ChatResponse(answer=final_answer)
    except Exception as e:
        err = str(e)
        print("Agent Error:", err)
        if "429" in err or "Resource exhausted" in err:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        raise HTTPException(status_code=500, detail=err)
