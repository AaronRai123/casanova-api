from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from backend.retrieval import search_products, parse_query
from backend.ranking import rank_results

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed
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
        query, excluded_terms = parse_query(query)
        retrieved_products = search_products(query)
        if not retrieved_products:
            return {"message": "No products found for the given query"}

        ranked_results = rank_results(query, retrieved_products)

        final_results = [
            {
                "id": product.get("id"),
                "metadata": product.get("metadata"),
                "score": product.get("score"),
                "re_rank_score": product.get("re_rank_score"),
            }
            for product in ranked_results
        ]

        return jsonable_encoder(final_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
