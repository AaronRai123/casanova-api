import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rank_results(query, retrieved_products):
    scored_products = []
    for product in retrieved_products:
        # Use the product's description from metadata (or empty string)
        description = product["metadata"].get("description", "")
        inputs = tokenizer(query, description, return_tensors="pt", truncation=True, padding=True)
        score = model(**inputs).logits.item()
        if math.isnan(score):
            score = 0.0
        product["re_rank_score"] = score

        # Optionally, remove a problematic key if it exists:
        if isinstance(product, dict) and "vector" in product:
            del product["vector"]

        scored_products.append(product)
    
    ranked_products = sorted(scored_products, key=lambda p: p["re_rank_score"], reverse=True)
    final_products = []
    for p in ranked_products[:5]:
        final_products.append({
            "id": p.get("id"),
            "metadata": p.get("metadata"),
            "score": p.get("score"),
            "re_rank_score": p.get("re_rank_score")
        })
    return final_products
