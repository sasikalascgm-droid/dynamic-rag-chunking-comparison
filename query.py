
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Get the project folder reliably
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "python_pdf_medium"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DATABASE_PATH)

# Check whether the collection exists
existing_collections = [
    collection.name for collection in client.list_collections()
]

if COLLECTION_NAME not in existing_collections:
    print(f"\nCollection '{COLLECTION_NAME}' was not found.")
    print("Run compare_chunking.py first to create the database.")
    raise SystemExit(1)

collection = client.get_collection(name=COLLECTION_NAME)

print("Database connected!")
print("Total stored chunks:", collection.count())

if collection.count() == 0:
    print("The collection is empty. Run compare_chunking.py again.")
    raise SystemExit(1)

question = input("\nAsk a question about Python: ").strip()

if not question:
    print("Question cannot be empty.")
    raise SystemExit(1)

# Create a normalized question embedding
question_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# Retrieve the three most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=min(3, collection.count())
)

print("\n===== RELEVANT INFORMATION =====")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\nResult {i}:")
    print(document)

print("\n===== RETRIEVAL COMPLETED =====")
