from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# Load transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone using the new API
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY", "pcsk_WQgQD_KavoYd4hHmkpZM6ezo8yQF4MwoKbsjzBaSW3EwahMUhGdu2Nw5psnZhf4VNfcgp")
)

# Connect to Pinecone Index
index = pc.Index("casanova-search")

def search_products(query):
    """Process a user query and retrieve the most relevant products."""
    query_vector = model.encode([query]).tolist()
    print("Query vector:", query_vector)  # Debug: see the generated vector
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    print("Pinecone results:", results)  # Debug: see the raw results
    return results["matches"]

