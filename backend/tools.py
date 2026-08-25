import os
import json
import urllib.request
import urllib.error
from langchain_core.tools import tool
from rag import search, movie_collection, places_collection

# Set this in your .env (locally) and in Render's Environment Variables
# (in production) to enable real, live nearby-places search via Google.
# If it's not set, the app just falls back to the small curated places list
# below — nothing breaks, it just won't have real-world results.
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def _recommend_from_list(mood, collection, label):
    """Shared logic for recommending from our small hand-curated lists."""
    results = search(mood, collection, n=4)
    retrieved = "\n".join(results)
    singular = label[:-1] if label.endswith("s") else label
    return (
        f"Here are some candidate {label} from our database matching '{mood}':\n"
        f"{retrieved}\n\n"
        "Don't just list all of these — pick whichever ones actually fit best "
        f"and recommend those, with your own reasoning. Only recommend {label} "
        f"from this list; never invent a {singular} that isn't here."
    )


@tool
def recommend_movie(mood: str) -> str:
    """Recommends movies based on the user's mood or the type of movie they want.
    Use this when the user wants to watch a movie or asks for movie suggestions."""
    return _recommend_from_list(mood, movie_collection, "movies")


@tool
def recommend_places(mood: str) -> str:
    """Recommends places to go out based on the user's mood or what they feel like
    doing, from a small curated list. Use this when the user's real location is
    NOT known (no latitude/longitude given in the conversation)."""
    return _recommend_from_list(mood, places_collection, "places")


@tool
def recommend_nearby_places(mood: str, latitude: float, longitude: float) -> str:
    """Recommends REAL nearby places using the user's actual GPS location via
    Google Places. Use this INSTEAD of recommend_places whenever the user's
    latitude and longitude are known from the conversation context (they'll
    appear in a bracketed note in the message). Pass the exact latitude and
    longitude given to you — don't guess or round them."""
    if not GOOGLE_PLACES_API_KEY:
        # Not configured — fall back to the generic list rather than failing
        return _recommend_from_list(mood, places_collection, "places")

    url = "https://places.googleapis.com/v1/places:searchText"
    body = json.dumps({
        "textQuery": f"{mood} place to go",
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": 8000.0,  # ~8km — a reasonable "nearby" radius
            }
        },
        "maxResultCount": 5,
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            # Field mask is required by the Places API (New) — it limits
            # which fields (and therefore billing cost) come back
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,"
                "places.rating,places.types"
            ),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        # Google API unreachable, misconfigured, or timed out — degrade
        # gracefully instead of breaking the conversation
        return _recommend_from_list(mood, places_collection, "places")

    places = data.get("places", [])
    if not places:
        return (
            f"No real nearby places were found for '{mood}'. Let the user "
            "know honestly, and offer a general suggestion instead."
        )

    lines = []
    for place in places:
        name = place.get("displayName", {}).get("text", "Unknown place")
        address = place.get("formattedAddress", "address unavailable")
        rating = place.get("rating", "no rating yet")
        lines.append(f"{name} | address: {address} | rating: {rating}")

    retrieved = "\n".join(lines)
    return (
        f"Here are REAL nearby places matching '{mood}':\n{retrieved}\n\n"
        "These are real, live places actually near the user right now "
        "(from Google), not from a fixed list. Pick whichever fit best and "
        "recommend those, with your own reasoning, and mention the address "
        "so they can actually find it."
    )
