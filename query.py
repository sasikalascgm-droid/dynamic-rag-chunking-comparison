
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Connect to the existing ChromaDB database
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="python_interview_questions"
)

print("Database connected!")
print("Total stored chunks:", collection.count())

# Step 3: Ask the user for a question
question = input("\nAsk a question about Python: ")

# Step 4: Convert the question into an embedding
question_embedding = model.encode(question).tolist()

# Step 5: Search for the most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Step 6: Display the retrieved information
print("\n===== RELEVANT INFORMATION =====")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\nResult {i}:")
    print(document)

print("\n===== RETRIEVAL COMPLETED =====")
