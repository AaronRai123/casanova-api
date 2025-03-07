from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from backend.retrieval import search_products
from backend.ranking import rank_results
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend to call backend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/search")
async def search(query: str):
    retrieved_products = search_products(query)
    ranked_results = rank_results(query, retrieved_products)

    # Build a clean list of results
    final_results = [
        {
            "id": product.get("id"),
            "metadata": product.get("metadata"),
            "score": product.get("score"),  # Original Pinecone score (if available)
            "re_rank_score": product.get("re_rank_score")  # Custom ranking score
        }
        for product in ranked_results
    ]

    # Use jsonable_encoder to ensure serialization
    return jsonable_encoder(final_results)
