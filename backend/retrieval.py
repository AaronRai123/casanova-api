import numpy as np
from backend.embeddings import index, model
from backend.query_processing import parse_query

def apply_exclusion_penalty(query_vector, excluded_vectors, lambda_penalty=0.5):
    """Subtracts the weighted average of excluded term vectors from the query vector."""
    if excluded_vectors:
        exclusion_vector = np.mean(excluded_vectors, axis=0)
        return query_vector - lambda_penalty * exclusion_vector
    return query_vector

def search_products(query):
    """Retrieves products while handling semantic exclusions."""
    # Parse the query to get the refined query and excluded terms.
    refined_query, excluded_terms = parse_query(query)
    # Get the query vector for the refined query.
    query_vector = model.encode(refined_query)

    # Encode each excluded term.
    excluded_vectors = [model.encode(term) for term in excluded_terms] if excluded_terms else []
    # Adjust the query vector by applying the exclusion penalty.
    adjusted_vector = apply_exclusion_penalty(query_vector, excluded_vectors, lambda_penalty=0.5)

    # Query the Pinecone index.
    results = index.query(vector=adjusted_vector.tolist(), top_k=50, include_metadata=True)

    # Filter out any candidate whose description includes an excluded term.
    filtered_results = []
    for product in results["matches"]:
        product_desc = product["metadata"].get("description", "").lower()
        if not any(term.lower() in product_desc for term in excluded_terms):
            filtered_results.append(product)

    return filtered_results
