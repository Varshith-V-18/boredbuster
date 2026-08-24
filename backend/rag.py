from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model (turns text into vectors)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB client (the vector database, stored in memory)
chroma_client = chromadb.Client()

#this function is to load the data in the db 
def load_data(file_path, collection_name):
    # Read the file, one item per line
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Create a collection (like a table) in ChromaDB
    collection = chroma_client.get_or_create_collection(collection_name)

    # Embed each line and store it, with an id
    collection.add(
        documents=lines,
        ids=[f"{collection_name}_{i}" for i in range(len(lines))]
    )
    return collection

# Load both knowledge bases
movie_collection = load_data("movies.txt", "movies")
places_collection = load_data("places.txt", "places")

# the search function

def search(query, collection, n=3):
    # Find the n most similar items to the query
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    # Return the matching documents as a list
    return results["documents"][0]