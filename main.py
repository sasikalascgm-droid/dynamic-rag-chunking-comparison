
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ollama_client import generate_answer, OllamaClientError


# ----------------------------------------
# 1. PROJECT CONFIGURATION
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "python_pdf_medium"


# ----------------------------------------
# 2. INITIALIZE APPLICATION
# ----------------------------------------

app = FastAPI(
    title="Dynamic RAG API",
    description="Ask questions using information from a PDF.",
    version="1.0.0"
)

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)


# ----------------------------------------
# 3. GET CHROMADB COLLECTION
# ----------------------------------------

def get_collection():
    collection_names = [
        item.name for item in client.list_collections()
    ]

    if COLLECTION_NAME not in collection_names:
        return None

    return client.get_collection(
        name=COLLECTION_NAME
    )


# ----------------------------------------
# 4. REQUEST MODEL
# ----------------------------------------

class QuestionRequest(BaseModel):
    question: str


# ----------------------------------------
# 5. HOME ENDPOINT
# ----------------------------------------

@app.get("/")
def home():
    return {
        "message": "Dynamic RAG API is running"
    }


# ----------------------------------------
# 6. ASK ENDPOINT
# ----------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Retrieve the PDF collection
    collection = get_collection()

    if collection is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Collection '{COLLECTION_NAME}' is missing. "
                "Run compare_chunking.py first."
            )
        )

    total_chunks = collection.count()

    if total_chunks == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "The PDF collection is empty. "
                "Run compare_chunking.py again."
            )
        )

    # Create the question embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve the three most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, total_chunks)
    )

    documents = results["documents"][0] or []

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks were found in the PDF."
        )

    context = "\n\n".join(documents)

    # Build a strict PDF-based prompt
    prompt = f"""
You are a PDF question-answering assistant.

Follow these rules strictly:
1. Answer ONLY the user's exact question.
2. Use ONLY information explicitly available
   in the PDF context below.
3. Do NOT answer other questions found in the context.
4. Do NOT use general knowledge.
5. Do NOT guess or invent information.
6. If the context does not contain enough information
   to answer the question, reply exactly:
   I could not find the answer in the PDF.
7. Keep the answer clear and concise.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER ONLY THE USER QUESTION:
"""

    # Generate an answer using the shared Ollama client
    try:
        answer = generate_answer(prompt)

    except OllamaClientError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error)
        ) from error

    return {
        "question": question,
        "answer": answer
    }
