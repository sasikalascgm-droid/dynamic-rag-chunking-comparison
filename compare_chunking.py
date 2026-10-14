
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Project configuration
PDF_PATH = Path(
    r"C:\Users\acer\OneDrive\Desktop\Python-Interview-Questions.pdf"
)

DATABASE_PATH = "chroma_db"
RESULTS_PATH = Path("chunking_results.txt")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


# 2. Chunking strategies
STRATEGIES = [
    {"name": "small", "chunk_size": 500, "overlap": 50},
    {"name": "medium", "chunk_size": 1000, "overlap": 200},
    {"name": "large", "chunk_size": 1500, "overlap": 300},
]


# 3. Evaluation questions
QUESTIONS = [
    "What is Python?",
    "What is a list in Python?",
    "What is the difference between a list and a tuple?",
    "What are the advantages of Python?",
    "What is a Python dictionary?",
    "What is a Python function?",
    "What is object-oriented programming?",
    "What is inheritance in Python?",
    "What is exception handling in Python?",
    "What is the difference between a list and a set?",
]


# 4. Extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    page_texts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text and page_text.strip():
            page_texts.append(page_text)

    extracted_text = "\n".join(page_texts)

    if not extracted_text.strip():
        raise ValueError("No readable text found in the PDF.")

    return extracted_text


# 5. Create chunks
def create_chunks(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# 6. Store chunks in ChromaDB
def ingest_chunks(client, model, chunks, strategy):
    collection_name = f"python_pdf_{strategy['name']}"

    # Recreate only this experiment's collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "strategy": strategy["name"],
                "chunk_index": i,
                "chunk_size": strategy["chunk_size"],
                "overlap": strategy["overlap"],
            }
            for i in range(len(chunks))
        ],
    )

    return collection


# 7. Retrieve the top matching chunks
def retrieve_chunks(collection, model, question):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    return collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "distances", "metadatas"],
    )


# 8. Automatic keyword-overlap evaluation
def evaluate_relevance(question, retrieved_documents):
    question_words = set(
        question.lower()
        .replace("?", "")
        .replace(",", "")
        .split()
    )

    stop_words = {
        "what", "is", "a", "an", "the", "in",
        "of", "and", "to", "between", "are",
        "how", "does", "difference", "advantages",
    }

    question_words -= stop_words

    if not question_words:
        return 0

    for document in retrieved_documents:
        document_words = set(
            document.lower()
            .replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .split()
        )

        overlap = len(question_words & document_words)
        overlap_ratio = overlap / len(question_words)

        if overlap_ratio >= 0.5:
            return 1

    return 0


# 9. Compare retrieval across strategies
def compare_retrieval(model, collections, output_file):
    evaluation_scores = {
        strategy["name"]: []
        for strategy, collection in collections
    }

    all_distances = {
        strategy["name"]: []
        for strategy, collection in collections
    }

    for question_number, question in enumerate(
        QUESTIONS, start=1
    ):
        print("\n" + "=" * 70)
        print(
            f"QUESTION {question_number}/{len(QUESTIONS)}: "
            f"{question}"
        )
        print("=" * 70)

        output_file.write(f"\n\nQUESTION: {question}\n")

        for strategy, collection in collections:
            results = retrieve_chunks(
                collection, model, question
            )

            documents = results["documents"][0]
            distances = results["distances"][0]

            print(
                f"\n--- {strategy['name'].upper()} STRATEGY ---"
            )
            print(
                f"Chunk size: {strategy['chunk_size']}, "
                f"Overlap: {strategy['overlap']}"
            )

            output_file.write(
                f"\n{strategy['name'].upper()} STRATEGY\n"
            )

            for rank, document in enumerate(
                documents, start=1
            ):
                distance = distances[rank - 1]
                preview = " ".join(document.split())[:500]

                print(
                    f"\nResult {rank} | "
                    f"Distance: {distance:.4f}"
                )
                print(preview)

                output_file.write(
                    f"\nResult {rank} | "
                    f"Distance: {distance:.4f}\n"
                )
                output_file.write(preview + "\n")

            if distances:
                all_distances[strategy["name"]].append(
                    distances[0]
                )

            # Automatically calculate a keyword-overlap score
            score = evaluate_relevance(question, documents)

            evaluation_scores[strategy["name"]].append(score)

            print(f"Automatic keyword-overlap score: {score}")

            output_file.write(
                f"Automatic keyword-overlap score: {score}\n"
            )

    # 10. Print evaluation summary
    print("\n" + "#" * 70)
    print("CHUNKING STRATEGY EVALUATION SUMMARY")
    print("#" * 70)

    output_file.write(
        "\n\nCHUNKING STRATEGY EVALUATION SUMMARY\n"
    )

    for strategy, collection in collections:
        name = strategy["name"]
        scores = evaluation_scores[name]
        distances = all_distances[name]

        hit_rate = (
            sum(scores) / len(scores) * 100
            if scores else 0
        )

        average_distance = (
            sum(distances) / len(distances)
            if distances else 0
        )

        summary = (
            f"{name.upper()}: "
            f"chunk_size={strategy['chunk_size']}, "
            f"overlap={strategy['overlap']}, "
            f"chunks={collection.count()}, "
            f"Keyword Hit@3={hit_rate:.1f}%, "
            f"average_top1_distance={average_distance:.4f}"
        )

        print(summary)
        output_file.write(summary + "\n")

    explanation = (
        "\nKeyword Hit@3 is the percentage of questions for which "
        "at least one of the top three retrieved chunks contains "
        "enough question keywords according to the simple threshold.\n"
        "This is an approximate keyword baseline, not a reliable "
        "measure of whether a chunk actually answers the question.\n"
        "Lower average top-1 cosine distance indicates greater "
        "embedding similarity, but does not guarantee answer correctness.\n"
    )

    print(explanation)
    output_file.write(explanation)

    return evaluation_scores


# 11. Main program
def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Update PDF_PATH to your actual PDF location."
        )

    print("STEP 1: Extracting PDF text...")
    text = extract_pdf_text(PDF_PATH)
    print(f"Extracted characters: {len(text)}")

    print("\nSTEP 2: Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nSTEP 3: Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DATABASE_PATH)

    collections = []

    print("\nSTEP 4: Ingesting each chunking strategy...")

    for strategy in STRATEGIES:
        chunks = create_chunks(
            text,
            strategy["chunk_size"],
            strategy["overlap"],
        )

        collection = ingest_chunks(
            client, model, chunks, strategy
        )

        collections.append((strategy, collection))

        print(
            f"\n{strategy['name'].upper()}: "
            f"{len(chunks)} chunks created and stored."
        )

    print("\nSTEP 5: Evaluating retrieval results...")

    with RESULTS_PATH.open(
        "w", encoding="utf-8"
    ) as output_file:
        output_file.write(
            "DYNAMIC RAG - CHUNKING STRATEGY COMPARISON\n"
        )
        output_file.write(f"PDF: {PDF_PATH.name}\n")
        output_file.write(f"Embedding model: {MODEL_NAME}\n")
        output_file.write(
            f"Questions evaluated: {len(QUESTIONS)}\n"
        )

        compare_retrieval(
            model, collections, output_file
        )

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
