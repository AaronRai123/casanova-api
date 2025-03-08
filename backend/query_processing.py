import re
from transformers import pipeline

# Initialize NLU pipeline for intent detection.
nlu = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli")

def detect_exclusion_intent(query):
    # For simplicity, assume if "not" appears, there's exclusion intent.
    return "not" in query.lower()

def parse_query(query):
    """
    If exclusion intent is detected, use a text2text-generation model
    (like FLAN-T5) to extract excluded terms.
    Returns a tuple: (refined_query, list_of_excluded_terms)
    """
    if detect_exclusion_intent(query):
        extractor = pipeline("text2text-generation", model="google/flan-t5-xl")
        extraction = extractor(f"Extract the terms to exclude from: {query}")
        excluded_text = extraction[0]['generated_text']
        # Assume the output is a comma-separated list.
        excluded_terms = [term.strip() for term in excluded_text.split(",") if term.strip()]
        # Remove the excluded terms and the word "not" from the query.
        refined_query = query
        for term in excluded_terms:
            refined_query = refined_query.replace(term, "")
        refined_query = refined_query.replace("not", "").strip()
        return refined_query, excluded_terms
    return query, []
