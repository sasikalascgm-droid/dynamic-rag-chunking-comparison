
import chromadb
from sentence_transformers import SentenceTransformer
import json
import urllib.request
import urllib.error

# 1. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(name="python_pdf_medium")

# 3. Ask a question
question = input("\nAsk a question about your PDF: ")

# 4. Convert the question into an embedding
question_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# 5. Retrieve the three most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

documents = results["documents"][0]
context = "\n\n".join(documents)

# 6. Prepare the prompt for the local AI model
prompt = f"""
Answer the question using only the context below.

If the answer is not available in the context,
say "I could not find the answer in the PDF."

Context:
{context}

Question: {question}

Answer:
"""

# 7. Send the prompt to Ollama
payload = json.dumps({
    "model": "llama3.2:3b",
    "prompt": prompt,
    "stream": False
}).encode("utf-8")

request = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("\nRAG Answer:")
    print(data["response"])

except urllib.error.URLError as error:
    print("\nCould not connect to Ollama.")
    print("Make sure the Ollama application is running.")
    print("Details:", error)

except TimeoutError:
    print("\nThe request took too long. Try again.")
