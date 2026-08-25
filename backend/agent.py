import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools import recommend_movie, recommend_places

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

# BoredBuster always REPLIES in English, no matter what language the user
# writes in. (We tried auto-detecting the user's language and replying to
# match it, but that turned out to be a genuinely unreliable problem to
# solve for short chat messages, and kept producing new bugs. Sticking to
# one reply language removes that whole class of bugs.)
#
# The movie/place CONTENT itself, on the other hand, does span multiple
# languages (see movies.txt) — English, Telugu, Hindi, Tamil, Kannada,
# Malayalam, and Punjabi films, each tagged with their language. The
# instruction below makes sure the bot only ever claims a movie matches a
# requested language when it's actually tagged with that language, instead
# of mislabeling whatever it has on hand.
SYSTEM_PROMPT = (
    "You are BoredBuster, a friendly assistant that recommends movies and "
    "places to go based on the user's mood. Talk like a knowledgeable "
    "friend chatting casually, not like you're filling out a form. Write "
    "in natural, flowing sentences and paragraphs. Do NOT use markdown "
    "tables, and avoid rigid list templates with repeated labels like "
    "\"Vibe:\" or \"Why it's a good pick:\" for every single item — vary "
    "your phrasing, react to what the user said, and let your personality "
    "come through, the way a person recommending something to a friend "
    "would. A short bullet list is fine occasionally if it genuinely helps "
    "readability, but prose should be your default. "
    "Always reply in English, no matter what language the user writes in — "
    "but keep movie and place titles in their original form (don't "
    "translate proper names). "
    "The movie database includes films in multiple languages: English, "
    "Telugu, Hindi, Tamil, Kannada, Malayalam, and Punjabi, each tagged "
    "with its language. If the user asks for movies in a specific "
    "language, only recommend movies whose tags actually include that "
    "language. If none of the retrieved movies match the language they "
    "asked for (e.g. they ask for a language outside the list above), say "
    "so plainly (e.g. \"I don't have any Korean movies in my list right "
    "now, but here are some other options\") — never present movies as "
    "being in a language they aren't."
)


# A function the server will call to get a response from the agent
def get_response(user_message: str) -> str:
    result = agent.invoke({
        "messages": [
            ("system", SYSTEM_PROMPT),
            ("user", user_message),
        ]
    })
    # The agent returns a list of messages; the last one is the final answer
    return result["messages"][-1].content
