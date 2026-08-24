from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import get_response
from database import init_db, save_message, get_history

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

# View your stored conversation history, e.g. http://localhost:8000/history
@app.get("/history")
def history(limit: int = 100):
    return {"history": get_history(limit)}