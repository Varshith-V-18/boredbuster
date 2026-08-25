import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import get_response
from database import init_db, save_message, get_history

# Secret key required to view chat history. Set HISTORY_KEY in your .env
# (locally) and in Render's Environment Variables (in production).
HISTORY_KEY = os.getenv("HISTORY_KEY")

# Create the FastAPI app
app = FastAPI()

# Allow the React frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure the chat_history table exists before we start serving requests
@app.on_event("startup")
def on_startup():
    init_db()

# Define what a chat message looks like coming in
class ChatRequest(BaseModel):
    message: str

# The endpoint the frontend calls
@app.post("/chat")
def chat(request: ChatRequest):
    reply = get_response(request.message)
    save_message(request.message, reply)
    return {"reply": reply}

# View your stored conversation history. Only you should be able to see this,
# so it now requires a secret key, e.g.:
# http://localhost:8000/history?key=YOUR_SECRET_KEY
@app.get("/history")
def history(limit: int = 100, key: str = ""):
    if not HISTORY_KEY or key != HISTORY_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"history": get_history(limit)}