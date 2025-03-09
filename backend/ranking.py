import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the cross-encoder ranking model.
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rank_results(query, retrieved_products):
    """Re-ranks retrieved products using a cross-encoder model."""
    scored_products = []
    for product in retrieved_products:
        description = product["metadata"].get("description", "")
        inputs = tokenizer(query, description, return_tensors="pt", truncation=True, padding=True)
        score = model(**inputs).logits.item()
        if math.isnan(score):
            score = 0.0
        product["re_rank_score"] = score
        scored_products.append(product)

    # Return top 5 products sorted by re-rank score.
    return sorted(scored_products, key=lambda p: p["re_rank_score"], reverse=True)[:5]
