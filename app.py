from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from rag import RAG
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG pipeline...")
    app.state.rag_pipeline = RAG()
    print("RAG pipeline initialized.")
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = app.state.rag_pipeline.chat(request.query, request.chat_history)

    return ChatResponse(
        answer=result["answer"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
