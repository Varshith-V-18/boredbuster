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


# Common words that show up constantly in this app's typical English
# messages. If a message contains a couple of these, it's treated as English
# right away without even asking the statistical detector — this is what
# stops short/casual phrases like "can you can you suggest a movie" from
# being misread as another language.
ENGLISH_HINT_WORDS = {
    "can", "you", "your", "suggest", "suggestion", "recommend",
    "recommendation", "movie", "movies", "film", "films", "place", "places",
    "mood", "bored", "boring", "feel", "feeling", "feels", "want", "wanna",
    "give", "tell", "me", "please", "funny", "scary", "sad", "happy",
    "tired", "good", "nice", "something", "anything", "what", "which",
    "yeah", "yes", "no", "hi", "hello", "hey", "thanks", "thank", "watch",
    "show", "shows", "go", "out", "today", "tonight", "now",
}


def detect_language(text: str) -> str:
    """Guess the language of the user's message. Falls back to English for
    short/ambiguous text, since single-word or very short messages ("hi",
    "ok") aren't reliably detectable and default English is the safest bet."""
    words = text.strip().lower().split()
    if len(words) < 3:
        return "English"

    # First check: does this look like a typical English message for this
    # app? If so, skip the detector entirely — it's the biggest source of
    # false positives on short phrases.
    hint_count = sum(1 for w in words if w.strip(".,!?¿¡") in ENGLISH_HINT_WORDS)
    if hint_count >= 2:
        return "English"

    try:
        top_guess = detect_langs(text)[0]
    except LangDetectException:
        return "English"

    # Only trust the detector when it's VERY confident. Genuine non-English
    # sentences score 99%+ confidence in practice, so a high bar filters out
    # false positives while still catching real non-English messages.
    if top_guess.prob >= 0.99:
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
