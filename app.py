from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from rag import RAG
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG pipeline...")
    app.state.rag_pipeline = RAG()
    print("RAG pipeline initialized.")
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "https://lakshyaworks.vercel.app",
    "http://localhost:3000"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

@app.get("/health-check")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = app.state.rag_pipeline.chat(request.query, request.chat_history)

    return ChatResponse(
        answer=result["answer"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
