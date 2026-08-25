# BoredBuster 🎬🌍

A mood-based movie and places recommender chatbot. Tell it how you're
feeling, and it recommends something to watch or somewhere to go — powered
by an LLM agent with real tool-calling, not a hardcoded rules engine.

**Live app:** https://boredbuster.vercel.app

## What it does

- Chat interface where a user describes their mood ("funny," "scary,"
  "romantic," "I'm bored") or asks for movies in a specific language
  ("kannada movies," "hindi comedy")
- An LLM agent decides whether to call a movie-recommendation tool or a
  places-recommendation tool based on the conversation, then responds
  conversationally
- Every conversation is persisted to a Postgres database
- A protected admin endpoint lets the owner review chat history without
  exposing it publicly

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Render |
| Agent framework | LangChain / LangGraph (`create_react_agent`) |
| LLM | Llama (`openai/gpt-oss-20b`), served via Groq |
| Database | PostgreSQL (Neon, serverless Postgres) |
| Movie/place matching | Custom lightweight tag-based search (see below) |

## Architecture

```
User (browser)
   │
   ▼
React frontend (Vercel)
   │  POST /chat { message }
   ▼
FastAPI backend (Render)
   │
   ├─▶ LangGraph ReAct agent (Groq LLM)
   │       │
   │       ├─▶ recommend_movie(mood) tool ──▶ tag-based search over movies.txt
   │       └─▶ recommend_places(mood) tool ─▶ tag-based search over places.txt
   │
   └─▶ Postgres (Neon) — every message + reply saved to chat_history
```

## Notable engineering decisions

A few real bugs came up building and deploying this, each with a root
cause worth knowing:

**Memory crash on deploy (exit code 137).** The backend was OOM-killed on
Render's free tier (512MB RAM) shortly after every deploy. Root cause: an
unused `sentence_transformers` import was pulling in PyTorch, which alone
exceeded the memory limit before the app even finished starting up.
Fixed by removing the dead import.

**A misleading CORS error hiding a real bug.** The frontend showed a CORS
policy error in the browser console, which pointed away from the actual
problem. A crashed backend response doesn't include CORS headers, so any
server-side exception can *look* like a CORS issue from the client side.
The real cause, found via the Network tab and server logs, was a
corrupted `GROQ_API_KEY` environment variable that had accidentally been
set to the literal text of a shell command instead of the real key.

**An unauthenticated admin endpoint.** Before sharing the app publicly, a
review found that `/history` — which returns every user's full chat
history — had no authentication at all. Fixed by requiring a secret key
(`?key=...`) checked against a server-side environment variable, returning
403 Forbidden otherwise.

**A hidden cold-start cost.** After the app went live, cold starts (the
free tier sleeps after inactivity) were taking far longer than expected.
The cause: the vector database (ChromaDB) was using its default embedding
function, which silently downloads a ~79MB ML model from the internet the
first time it runs — and Render's free tier fully restarts the container
on every wake-up, so that download was happening on *every* cold start.
Fixed by replacing the ChromaDB/embedding-model approach entirely with a
simple tag-based matching function over the movie/place data — since the
dataset is small and already hand-tagged with genre/mood/language, a full
ML embedding pipeline was unnecessary overhead. This also removed a large
dependency (onnxruntime, numpy, etc.) from the deploy.

**Language handling.** Early versions tried to auto-detect the user's
language and reply in kind, first by asking the LLM to self-detect, then
via a statistical language-detection library with a confidence threshold.
Both approaches produced false positives on short, casual phrases (e.g.
"can you can you suggest a movie" was misdetected as French). The final
design separates two concerns that don't need to be coupled: the bot
always *replies* in English (simple, reliable, no more false positives),
while the movie *catalog* itself is genuinely multilingual (English,
Telugu, Hindi, Tamil, Kannada, Malayalam, Punjabi), each entry tagged with
its language — so the bot can honestly say "I don't have Korean movies
yet" instead of mislabeling what it does have.

## Running locally

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add .env with GROQ_API_KEY, GROQ_MODEL, DATABASE_URL, HISTORY_KEY
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | API key for the Groq-hosted LLM |
| `GROQ_MODEL` | Model name (e.g. `openai/gpt-oss-20b`) |
| `DATABASE_URL` | Postgres connection string (Neon) |
| `HISTORY_KEY` | Secret key required to access `/history` |
| `VITE_API_URL` | (Frontend) URL of the deployed backend |
