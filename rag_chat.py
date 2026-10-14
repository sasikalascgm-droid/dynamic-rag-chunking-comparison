
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
# 2. LOAD MODEL AND DATABASE
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
# 4. RETRIEVE CONTEXT
# ----------------------------------------

def retrieve_context(question, top_k=3):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    total_chunks = collection.count()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks)
    )

    documents = results["documents"][0] or []

    return "\n\n".join(documents)


# ----------------------------------------
# 5. GENERATE ANSWER
# ----------------------------------------

def generate_rag_answer(question):
    context = retrieve_context(question)

    if not context.strip():
        return "I could not find the answer in the PDF."

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

    return generate_answer(prompt)


# ----------------------------------------
# 6. CHAT LOOP
# ----------------------------------------

print("\nDynamic RAG Chatbot")
print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Chatbot: Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:
        answer = generate_rag_answer(question)
        print(f"\nChatbot: {answer}\n")

    except OllamaClientError as error:
        print(f"\nCould not generate an answer: {error}\n")
