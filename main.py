
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import urllib.request
import urllib.error
import json

app = FastAPI(title="Dynamic RAG API")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing vector database
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="python_pdf_medium")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Dynamic RAG API is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # Convert question into an embedding
    embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve relevant PDF chunks
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    documents = results["documents"][0]
    context = "\n\n".join(documents)

    prompt = f"""
Answer the question using only the provided context.
If the answer is not available, say you could not
find it in the PDF.

Context:
{context}

Question: {question}

Answer:
"""

    payload = json.dumps({
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))

        return {"answer": data["response"]}

    except (urllib.error.URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Ollama: {error}"
        )
