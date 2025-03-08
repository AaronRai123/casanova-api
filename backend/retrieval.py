
import os
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import re

# Load transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone using the new API
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY", "pcsk_WQgQD_KavoYd4hHmkpZM6ezo8yQF4MwoKbsjzBaSW3EwahMUhGdu2Nw5psnZhf4VNfcgp"))
# Connect to Pinecone Index
index = pc.Index("casanova-search")

def parse_query(query):
    """Detects and separates inclusion and exclusion terms in a query."""
    exclusion_pattern = r"\bnot\b\s+([\w\s]+)"
    excluded_terms = re.findall(exclusion_pattern, query, re.IGNORECASE)
    
    for term in excluded_terms:
        query = query.replace(f"not {term}", "").strip()
    
    return query, excluded_terms

def get_contrastive_embedding(query, excluded_terms):
    """Generates an embedding that contrasts the query with excluded terms."""
    positive_embed = model.encode(query)
    if excluded_terms:
        negative_embed = model.encode(" ".join(excluded_terms))
        return positive_embed - np.dot(positive_embed, negative_embed) * negative_embed
    return positive_embed

def search_products(query):
    """Retrieves products while handling semantic exclusions."""
    query, excluded_terms = parse_query(query)
    query_vector = get_contrastive_embedding(query, excluded_terms).tolist()

    results = index.query(vector=query_vector, top_k=50, include_metadata=True)
    
    filtered_results = []
    for product in results["matches"]:
        product_desc = product["metadata"].get("description", "").lower()
        
        # Exclude products containing the forbidden words
        if not any(term.lower() in product_desc for term in excluded_terms):
            filtered_results.append(product)

    return filtered_results
