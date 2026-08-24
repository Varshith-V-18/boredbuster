import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# A standard Postgres connection string, e.g.:
# postgres://user:password@host:5432/dbname
# You'll get this from Vercel's Storage tab (or Neon/Supabase directly).
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Creates the chat_history table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_message TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def save_message(user_message: str, bot_reply: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_message, bot_reply) VALUES (%s, %s)",
        (user_message, bot_reply),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_history(limit: int = 100):
    """Returns the most recent conversations, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, user_message, bot_reply, created_at "
        "FROM chat_history ORDER BY id DESC LIMIT %s",
        (limit,),
    )
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows
