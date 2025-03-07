import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Set your Pinecone API key here or via an environment variable
API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_WQgQD_KavoYd4hHmkpZM6ezo8yQF4MwoKbsjzBaSW3EwahMUhGdu2Nw5psnZhf4VNfcgp")

# Initialize Pinecone using the new API
pc = Pinecone(api_key=API_KEY)

# Define your index name and the embedding dimension (for "all-MiniLM-L6-v2", it's 384)
index_name = "casanova-search"
dimension = 384

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine"
        # You can specify serverless options here if needed:
        # spec=ServerlessSpec(cloud="gcp", region="us-west1")
    )
    print(f"✅ Pinecone index '{index_name}' created!")
else:
    print(f"✅ Pinecone index '{index_name}' already exists!")

# Connect to the index
index = pc.Index(index_name)

# Load product data from CSV (adjust the path if needed)
csv_path = os.path.join(os.path.dirname(__file__), "products.csv")
df = pd.read_csv(csv_path)

print("✅ Successfully loaded products.csv!")

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert each product's description into an embedding and upload to the index
for _, row in df.iterrows():
    vector = model.encode(row["description"]).tolist()
    # Upsert with metadata including the product description
    index.upsert(vectors=[(
        str(row["id"]), 
        vector, 
        {"name": row["name"], "price": row["price"], "description": row["description"]}
    )])

print("✅ Product embeddings stored in Pinecone!")
