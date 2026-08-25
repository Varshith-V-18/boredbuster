import math
import json
import urllib.request
import urllib.parse
import urllib.error
from langchain_core.tools import tool
from rag import search, movie_collection, places_collection

# Real nearby-places search uses OpenStreetMap's Overpass API — it's free,
# community-run, and needs NO API key, NO signup, and NO billing/credit
# card at all. (We originally tried Google Places, but that requires
# enabling billing even for free-tier usage, which isn't an option here.)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Maps common mood/activity keywords to OpenStreetMap tag categories.
# OSM doesn't support free-text search like "romantic place" — it's a
# structured tag system — so we translate the user's mood into the right
# tags ourselves. (key, value) pairs, e.g. ("amenity", "cafe").
MOOD_TO_OSM_TAGS = {
    "relax": [("amenity", "cafe"), ("leisure", "park"), ("leisure", "garden")],
    "chill": [("amenity", "cafe"), ("leisure", "park")],
    "calm": [("leisure", "park"), ("leisure", "garden"), ("amenity", "cafe")],
    "peaceful": [("leisure", "park"), ("leisure", "garden")],
    "romantic": [("amenity", "restaurant"), ("amenity", "cafe"), ("leisure", "park"), ("tourism", "viewpoint")],
    "date": [("amenity", "restaurant"), ("amenity", "cafe"), ("leisure", "park")],
    "fun": [("amenity", "cinema"), ("leisure", "amusement_arcade"), ("leisure", "bowling_alley")],
    "exciting": [("amenity", "cinema"), ("leisure", "amusement_arcade"), ("amenity", "nightclub")],
    "energetic": [("leisure", "fitness_centre"), ("leisure", "sports_centre"), ("leisure", "amusement_arcade")],
    "active": [("leisure", "sports_centre"), ("leisure", "fitness_centre")],
    "social": [("amenity", "bar"), ("amenity", "pub"), ("amenity", "nightclub")],
    "night": [("amenity", "bar"), ("amenity", "nightclub"), ("amenity", "pub")],
    "party": [("amenity", "nightclub"), ("amenity", "bar")],
    "nature": [("leisure", "park"), ("leisure", "garden"), ("natural", "beach")],
    "outdoor": [("leisure", "park"), ("leisure", "garden"), ("tourism", "viewpoint")],
    "culture": [("tourism", "museum"), ("tourism", "gallery"), ("amenity", "theatre")],
    "art": [("tourism", "gallery"), ("tourism", "museum")],
    "museum": [("tourism", "museum")],
    "coffee": [("amenity", "cafe")],
    "food": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "hungry": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "drink": [("amenity", "bar"), ("amenity", "pub")],
    "shop": [("shop", "mall"), ("shop", "department_store")],
    "adventure": [("leisure", "sports_centre"), ("tourism", "attraction")],
}

DEFAULT_OSM_TAGS = [
    ("amenity", "cafe"), ("amenity", "restaurant"),
    ("leisure", "park"), ("tourism", "attraction"),
]


def _pick_osm_tags(mood: str):
    """Turn a free-text mood into a set of relevant OSM tags (deduped, in
    the order first matched)."""
    mood_lower = mood.lower()
    matched = []
    for keyword, tags in MOOD_TO_OSM_TAGS.items():
        if keyword in mood_lower:
            for tag in tags:
                if tag not in matched:
                    matched.append(tag)
    return matched or DEFAULT_OSM_TAGS


def _haversine_meters(lat1, lon1, lat2, lon2):
    """Straight-line distance between two coordinates, in meters."""
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


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
    """Recommends REAL nearby places using the user's actual GPS location, via
    OpenStreetMap. Use this INSTEAD of recommend_places whenever the user's
    latitude and longitude are known from the conversation context (they'll
    appear in a bracketed note in the message). Pass the exact latitude and
    longitude given to you — don't guess or round them."""
    print(f"[recommend_nearby_places] called with mood={mood!r} lat={latitude} lon={longitude}")
    tags = _pick_osm_tags(mood)

    # Group tags by OSM key so we can build one regex-alternation clause
    # per key, e.g. ["amenity"]["cafe","restaurant"] -> one clause
    by_key = {}
    for key, value in tags:
        by_key.setdefault(key, []).append(value)

    radius = 8000  # ~8km — a reasonable "nearby" radius
    clauses = []
    for key, values in by_key.items():
        pattern = "^(" + "|".join(values) + ")$"
        clauses.append(
            f'node["{key}"~"{pattern}"](around:{radius},{latitude},{longitude});'
        )

    query = f"[out:json][timeout:12];({''.join(clauses)});out body 8;"

    request = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "BoredBuster/1.0 (mood-based recommender app)",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        # OpenStreetMap unreachable or timed out — degrade gracefully
        # instead of breaking the conversation. Logged (not silently
        # swallowed) so we can see the real cause in Render's logs.
        print(f"[recommend_nearby_places] Overpass request failed: {type(exc).__name__}: {exc}")
        return _recommend_from_list(mood, places_collection, "places")

    elements = [e for e in data.get("elements", []) if e.get("tags", {}).get("name")]
    print(f"[recommend_nearby_places] Overpass returned {len(data.get('elements', []))} raw elements, {len(elements)} named")
    if not elements:
        return (
            f"No real nearby places with a listed name were found for "
            f"'{mood}'. Let the user know honestly, and offer a general "
            "suggestion instead."
        )

    lines = []
    for element in elements:
        name = element["tags"]["name"]
        category = element["tags"].get("amenity") or element["tags"].get("leisure") \
            or element["tags"].get("tourism") or element["tags"].get("shop") or ""
        distance_m = _haversine_meters(
            latitude, longitude, element["lat"], element["lon"]
        )
        distance_desc = (
            f"{distance_m:.0f}m away" if distance_m < 1000
            else f"{distance_m / 1000:.1f}km away"
        )
        lines.append(f"{name} | {category} | {distance_desc}")

    retrieved = "\n".join(lines)
    return (
        f"Here are REAL nearby places matching '{mood}':\n{retrieved}\n\n"
        "These are real, live places actually near the user right now "
        "(from OpenStreetMap), not from a fixed list. Pick whichever fit "
        "best and recommend those, with your own reasoning, and mention "
        "how far away each one is."
    )
