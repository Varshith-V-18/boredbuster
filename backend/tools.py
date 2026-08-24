from langchain_core.tools import tool
from rag import search, movie_collection, places_collection

@tool
def recommend_movie(mood: str) -> str:
    """Recommends movies based on the user's mood or the type of movie they want.
    Use this when the user wants to watch a movie or asks for movie suggestions."""
    results = search(mood, movie_collection, n=4)
    retrieved = "\n".join(results)
    return f"Here are movies from our database matching '{mood}':\n{retrieved}\n\nRecommend from ONLY these movies."

@tool
def recommend_places(mood: str) -> str:
    """Recommends places to go out based on the user's mood or what they feel like doing.
    Use this when the user wants to go out, visit somewhere, or asks for places to go."""
    results = search(mood, places_collection, n=4)
    retrieved = "\n".join(results)
    return f"Here are places from our database matching '{mood}':\n{retrieved}\n\nRecommend from ONLY these places."