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

# Tells the agent to always reply in whatever language the user wrote in
SYSTEM_PROMPT = (
    "You are BoredBuster, a friendly assistant that recommends movies and "
    "places to go based on the user's mood. "
    "Always detect the language the user's message is written in, and write "
    "your ENTIRE reply in that same language — including greetings, "
    "explanations, and framing text. Keep movie and place names in their "
    "original title (don't translate proper names), but everything else "
    "(the surrounding sentences) should be in the user's language. "
    "If you are honestly unsure what language a short or ambiguous message "
    "is in, default to English."
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