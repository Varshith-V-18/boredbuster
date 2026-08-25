import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langdetect import detect_langs, LangDetectException, DetectorFactory
from tools import recommend_movie, recommend_places

# Makes langdetect give the same result every time for the same input
DetectorFactory.seed = 0

load_dotenv()

# Set up the LLM (Llama, served by Groq)
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY")
)

# Put our tools in a list
tools = [recommend_movie, recommend_places]

# Build the agent: give it the LLM + the tools
agent = create_react_agent(llm, tools)

# Tells the agent to reply in whichever language it's told to use.
# (We detect the language ourselves in code below, instead of asking the
# model to guess — it's more reliable, especially once tool calls are
# happening in between the user's message and the final answer.)
SYSTEM_PROMPT = (
    "You are BoredBuster, a friendly assistant that recommends movies and "
    "places to go based on the user's mood. "
    "Each user message will start with a bracketed instruction telling you "
    "which language to reply in, for example '[Respond only in Spanish.]' — "
    "always follow that instruction for your ENTIRE reply, including "
    "greetings, explanations, and framing text. Keep movie and place names "
    "in their original title (don't translate proper names), but everything "
    "else should be in the specified language. Never mention, repeat, or "
    "quote the bracketed instruction itself in your reply."
)

# Common language codes -> names, so the model gets a clear word like
# "Spanish" instead of a code like "es"
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "hi": "Hindi", "te": "Telugu",
    "ta": "Tamil", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
    "ja": "Japanese", "ko": "Korean", "zh-cn": "Chinese", "ar": "Arabic",
    "ru": "Russian", "nl": "Dutch", "tr": "Turkish",
}


def detect_language(text: str) -> str:
    """Guess the language of the user's message. Falls back to English for
    short/ambiguous text, since single-word or very short messages ("hi",
    "ok") aren't reliably detectable and default English is the safest bet."""
    if len(text.strip().split()) < 3:
        return "English"

    try:
        top_guess = detect_langs(text)[0]
    except LangDetectException:
        return "English"

    # Only trust the detector when it's reasonably confident
    if top_guess.prob >= 0.85:
        return LANGUAGE_NAMES.get(top_guess.lang, "English")

    return "English"


# A function the server will call to get a response from the agent
def get_response(user_message: str) -> str:
    language = detect_language(user_message)
    result = agent.invoke({
        "messages": [
            ("system", SYSTEM_PROMPT),
            ("user", f"[Respond only in {language}.] {user_message}"),
        ]
    })
    # The agent returns a list of messages; the last one is the final answer
    return result["messages"][-1].content
