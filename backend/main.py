from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from backend.retrieval import search_products
from backend.ranking import rank_results

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Casanova API!"}

@app.get("/search")
async def search(query: str):
    try:
        retrieved_products = search_products(query)
        ranked_results = rank_results(query, retrieved_products)
        # Convert scores to native Python types if needed.
        final_results = []
        for product in ranked_results:
            final_results.append({
                "id": product.get("id"),
                "metadata": product.get("metadata"),
                "score": float(product.get("score")) if product.get("score") is not None else None,
                "re_rank_score": float(product.get("re_rank_score")) if product.get("re_rank_score") is not None else None,
            })
        return jsonable_encoder(final_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
