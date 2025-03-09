import re
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

# Configuration
XLM_MODEL = "joeddav/xlm-roberta-large-xnli"
T5_MODEL = "google/flan-t5-base"
DEVICE = -1  # Use CPU
# Force the slow tokenizer by using use_fast=False
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_MODEL, use_fast=False)
xlm_model = XLMRobertaForSequenceClassification.from_pretrained(XLM_MODEL)
nlu = pipeline("text-classification", model=xlm_model, tokenizer=xlm_tokenizer, device=DEVICE)

t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL, use_fast=False)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL, torch_dtype=torch.float32)
extractor = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, device=DEVICE)

def detect_exclusion_intent(query: str) -> bool:
    return bool(re.search(r"\b(not|except|without)\b", query, re.IGNORECASE))

def parse_query(query: str) -> tuple[str, list]:
    """
    Uses a generative prompt to extract a JSON object with keys "include" and "exclude".
    For example, for the query:
       "I need pants for work not jeans"
    Expected output (if successful):
       refined_query: "pants for work"
       excluded_terms: ["jeans"]
    If extraction fails, a fallback using regex is used.
    """
    if not detect_exclusion_intent(query):
        return query, []

    try:
        prompt = (
            "Extract a JSON object from the following query with two keys: "
            "\"include\" (a list of terms to search for) and "
            "\"exclude\" (a list of terms to avoid). Return only valid JSON. "
            f"Query: \"{query}\""
        )
        result = extractor(prompt, max_new_tokens=100, num_beams=4, early_stopping=True)
        json_text = result[0]['generated_text'].strip()
        if not (json_text.startswith("{") and json_text.endswith("}")):
            raise ValueError("Output is not valid JSON.")
        structured = json.loads(json_text)
        include_terms = structured.get("include", [])
        exclude_terms = structured.get("exclude", [])
    except Exception as e:
        print(f"Structured extraction error: {str(e)}")
        # Fallback using regex
        match = re.search(r"\bnot\s+(\w+)", query, re.IGNORECASE)
        exclude_terms = [match.group(1)] if match else []
        include_terms = [w for w in query.split() if w.lower() != "not" and (not exclude_terms or w.lower() not in exclude_terms)]
    
    refined_query = " ".join(include_terms) if include_terms else query
    return refined_query, exclude_terms
