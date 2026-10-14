
import json
import urllib.request
import urllib.error

from sentence_transformers import SentenceTransformer
import chromadb


# STEP 1: Load the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 2: Connect to ChromaDB
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="python_interview_questions"
)

print("Database connected!")
print("Stored chunks:", collection.count())


# STEP 3: Ask a question
while True:
    question = input(
        "\nAsk a question about your PDF (type 'exit' to stop): "
    ).strip()

    if question.lower() in ["exit", "quit"]:
        print("Thank you for using Dynamic RAG!")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:
        # STEP 4: Convert question into an embedding
        question_embedding = embedding_model.encode(
            question
        ).tolist()

        # STEP 5: Retrieve relevant document chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        documents = results["documents"][0]

        if not documents:
            print("No relevant information was found.")
            continue

        context = "\n\n".join(documents)

        print("\nGenerating an answer from your PDF...")

        # STEP 6: Prepare the prompt for the local AI
        prompt = f"""
You are a helpful assistant for answering questions
about a Python interview questions document.

Answer the user's question using ONLY the context below.

Rules:
1. Give a clear, simple answer.
2. Use the context as your source.
3. Do not invent facts that are not in the context.
4. If the context does not contain the answer, say:
   "I could not find enough information about this
   question in the provided PDF."
5. Include a short example when the context supports it.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        # STEP 7: Send context and question to Ollama
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }

        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(
            request, timeout=180
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        # STEP 8: Display the generated answer
        print("\n========== RAG ANSWER ==========")
        print(data["response"].strip())
        print("================================")

    except urllib.error.URLError as error:
        print("\nCould not connect to Ollama.")
        print("Check that Ollama is running.")
        print("Details:", error)

    except Exception as error:
        print("\nAn error occurred:", error)
