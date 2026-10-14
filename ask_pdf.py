
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ollama_client import generate_answer, OllamaClientError


# ----------------------------------------
# 1. CONFIGURATION
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "python_pdf_medium"


# ----------------------------------------
# 2. LOAD MODELS AND DATABASE
# ----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)


# ----------------------------------------
# 3. CHECK COLLECTION
# ----------------------------------------

collection_names = [
    item.name for item in client.list_collections()
]

if COLLECTION_NAME not in collection_names:
    print("PDF collection not found.")
    print("Run compare_chunking.py first.")
    raise SystemExit(1)

collection = client.get_collection(
    name=COLLECTION_NAME
)

if collection.count() == 0:
    print("PDF collection is empty.")
    print("Run compare_chunking.py again.")
    raise SystemExit(1)


# ----------------------------------------
# 4. ASK A QUESTION
# ----------------------------------------

question = input("\nAsk a question about your PDF: ").strip()

if not question:
    print("Question cannot be empty.")
    raise SystemExit(1)


# ----------------------------------------
# 5. RETRIEVE RELEVANT CHUNKS
# ----------------------------------------

question_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=min(3, collection.count())
)

documents = results["documents"][0] or []

if not documents:
    print("No relevant chunks were found in the PDF.")
    raise SystemExit(1)

context = "\n\n".join(documents)


# ----------------------------------------
# 6. BUILD THE PROMPT
# ----------------------------------------

prompt = f"""
You are a PDF question-answering assistant.

Rules:
1. Answer only the user's exact question.
2. Use only information explicitly available in the PDF context.
3. Do not use general knowledge or invent information.
4. If the answer is unavailable in the context, reply exactly:
   I could not find the answer in the PDF.
5. Keep the answer clear and concise.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


# ----------------------------------------
# 7. GENERATE THE ANSWER
# ----------------------------------------

try:
    answer = generate_answer(prompt)

    print("\nRAG Answer:")
    print(answer)

except OllamaClientError as error:
    print("\nCould not generate an answer.")
    print(error)
