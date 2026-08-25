# A previous version of this file used ChromaDB for similarity search. The
# problem: ChromaDB's default embedder downloads a ~79MB machine learning
# model from the internet the first time it runs. Render's free tier fully
# restarts the server (with nothing saved) every time it wakes up from
# sleeping, so that download was happening on EVERY wake-up — adding a lot
# of delay on top of the normal cold start. Swapping in our own lightweight
# "embedding" function (see git history) fixed the download but gave worse
# recommendations, because a small hashed vector isn't very accurate for
# this small a vocabulary.
#
# The real fix: our movies.txt/places.txt files already come with a
# hand-written "tags" column (e.g. "comedy funny wild"). We don't need a
# machine learning model to compare text at all — just match the words in
# the user's mood against those tags (and the description, as a backup).
# This is simpler, needs no downloads, no external libraries, and is more
# accurate for this dataset than a generic embedding model would be.


def load_data(file_path, collection_name):
    # Read the file, one item per line
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# Load both knowledge bases
movie_collection = load_data("movies.txt", "movies")
places_collection = load_data("places.txt", "places")


def _score(query_words, line):
    """Score how well one line (title | tags | description) matches the
    words in the user's mood/query. Tag matches count more than
    description matches, since tags are the curated, reliable signal."""
    parts = line.split("|")
    tags = set(parts[1].strip().lower().split()) if len(parts) > 1 else set()
    description = parts[2].strip().lower() if len(parts) > 2 else ""

    score = 0
    for word in query_words:
        if word in tags:
            score += 3
        elif len(word) >= 4 and any(
            len(tag) >= 4 and word[:4] == tag[:4] for tag in tags
        ):
            # matches word stems, e.g. "romantic" ~ "romance", "scary" ~ "scared"
            score += 2
        if word in description:
            score += 1
    return score


# the search function
def search(query, collection, n=3):
    """Find the n items whose tags/description best match the query."""
    query_words = [w for w in query.lower().split() if len(w) > 2]

    scored = [(_score(query_words, line), line) for line in collection]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    top = [line for score, line in scored[:n] if score > 0]

    # If nothing matched at all (e.g. an unusual mood), fall back to
    # returning the first n items rather than an empty list, so the LLM
    # always has something to recommend from.
    if not top:
        top = collection[:n]

    return top
