import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fuzzywuzzy import fuzz

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rank_results(query, retrieved_products):
    query, excluded_terms = parse_query(query)  # Extract exclusions
    
    scored_products = []
    for product in retrieved_products:
        description = product["metadata"].get("description", "")
        inputs = tokenizer(query, description, return_tensors="pt", truncation=True, padding=True)
        score = model(**inputs).logits.item()
        if math.isnan(score):
            score = 0.0

        # Apply fuzzy exclusion penalty if the excluded term appears
        exclusion_penalty = max(
            fuzz.partial_ratio(term.lower(), description.lower()) / 100
            for term in excluded_terms
        ) if excluded_terms else 0

        score *= (1 - exclusion_penalty * 0.7)  # Reduce score if exclusion match found
        product["re_rank_score"] = score

        # Clean up the object before returning
        if isinstance(product, dict) and "vector" in product:
            del product["vector"]

        scored_products.append(product)

    # Sort by re-rank score
    return sorted(scored_products, key=lambda p: p["re_rank_score"], reverse=True)[:10]
