
from pathlib import Path
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "Python-Interview-Questions.pdf"
DATABASE_PATH = str(BASE_DIR / "chroma_db")
RESULTS_PATH = BASE_DIR / "chunking_results.txt"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

STRATEGIES = [
    {"name": "small", "chunk_size": 500, "overlap": 50},
    {"name": "medium", "chunk_size": 1000, "overlap": 200},
    {"name": "large", "chunk_size": 1500, "overlap": 300},
]

EVALUATION_QUESTIONS = [
    "What is a Python dictionary?",
    "What is the difference between a list and a tuple?",
    "What is a Python function?",
    "What is a lambda function?",
    "What is object-oriented programming?",
    "What is inheritance in Python?",
    "What is exception handling?",
    "What is a Python module?",
    "What is the difference between == and is?",
    "What are Python decorators?",
]


# ==================================================
# 2. EXTRACT TEXT FROM PDF
# ==================================================

def extract_pdf_text(pdf_path):
    """Extract readable text from every page of a PDF."""

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            # Normalize Windows and Unix line endings.
            page_text = page_text.replace("\r\n", "\n")
            page_text = page_text.replace("\r", "\n")
            pages.append(page_text)

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError(
            "No readable text was extracted from the PDF."
        )

    return full_text


# ==================================================
# 3. SPLIT TEXT INTO QUESTION SECTIONS
# ==================================================

def split_into_sections(text):
    """
    Split numbered interview questions into separate sections.

    Example:
        1. What is Python?
        Answer: Python is a programming language.

        2. What is a list?
        Answer: A list stores multiple items.

    If numbered questions cannot be detected, use paragraphs.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Recognizes headings such as:
    # 1. What is Python?
    # 12. Explain inheritance.
    # 70.What is a local variable?
    question_pattern = re.compile(
        r"(?<!\d)\d{1,3}\s*[.)]\s*"
        r"(?=(?:What|How|Why|When|Where|Which|Who|"
        r"Explain|Describe|Define|Write|List|Name|"
        r"Can|Is|Are|Do|Does|Difference|Tell|Give)\b)",
        re.IGNORECASE,
    )

    matches = list(question_pattern.finditer(text))
    sections = []

    if matches:
        # Keep introductory text before the first question.
        introduction = text[:matches[0].start()].strip()

        if introduction:
            sections.append(introduction)

        for index, match in enumerate(matches):
            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            section = text[start:end].strip()

            if section:
                sections.append(section)

    else:
        # Fallback for PDFs without recognizable numbered questions.
        sections = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

    return [section for section in sections if section]


# ==================================================
# 4. SPLIT VERY LONG SECTIONS
# ==================================================

def split_long_section(section, chunk_size, overlap):
    """
    Keep question-and-answer sections together when possible.

    If a section is longer than the target size, split it
    by paragraphs or sentences. Use character splitting only
    for individual pieces that are still too long.
    """

    if len(section) <= chunk_size:
        return [section]

    # Prefer paragraph boundaries, then sentence boundaries.
    parts = [
        part.strip()
        for part in re.split(r"\n+|(?<=[.!?])\s+", section)
        if part.strip()
    ]

    if not parts:
        parts = [section]

    output = []
    current = ""

    for part in parts:
        if len(part) > chunk_size:
            if current:
                output.append(current.strip())
                current = ""

            step = chunk_size - overlap

            for start in range(0, len(part), step):
                piece = part[start:start + chunk_size].strip()

                if piece:
                    output.append(piece)

                if start + chunk_size >= len(part):
                    break

            continue

        candidate = f"{current}\n{part}".strip() if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                output.append(current.strip())

            current = part

    if current:
        output.append(current.strip())

    return output


# ==================================================
# 5. CREATE SECTION-AWARE CHUNKS
# ==================================================

def create_chunks(text, chunk_size, overlap):
    """
    Group question sections into chunks of different target sizes.

    Small: approximately 500 characters
    Medium: approximately 1000 characters
    Large: approximately 1500 characters

    Prefer section boundaries and preserve a limited overlap.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    sections = split_into_sections(text)

    if not sections:
        return []

    # Split sections that are too long for the chosen strategy.
    units = []

    for section in sections:
        units.extend(
            split_long_section(section, chunk_size, overlap)
        )

    chunks = []
    start = 0

    while start < len(units):
        current_parts = []
        current_length = 0
        end = start

        # Add complete sections until the target size is reached.
        while end < len(units):
            part = units[end]
            extra_length = len(part)

            if current_parts:
                extra_length += 1

            if (
                current_parts
                and current_length + extra_length > chunk_size
            ):
                break

            current_parts.append(part)
            current_length += extra_length
            end += 1

        if current_parts:
            chunk = "\n".join(current_parts).strip()

            if chunk:
                chunks.append(chunk)

        if end >= len(units):
            break

        # Find a small overlap using the last complete units.
        next_start = end
        overlap_length = 0

        for index in range(end - 1, start - 1, -1):
            part_length = len(units[index])

            if overlap_length > 0:
                part_length += 1

            if overlap_length + part_length <= overlap:
                overlap_length += part_length
                next_start = index
            else:
                break

        # Always move forward to prevent an infinite loop.
        if next_start <= start:
            start = end
        else:
            start = next_start

    return chunks


# ==================================================
# 6. CREATE OR REBUILD CHROMADB COLLECTIONS
# ==================================================

def ingest_chunks(client, model, text, strategy):
    """Create embeddings and store chunks in ChromaDB."""

    collection_name = f"python_pdf_{strategy['name']}"

    # Delete the previous collection before rebuilding it.
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = create_chunks(
        text=text,
        chunk_size=strategy["chunk_size"],
        overlap=strategy["overlap"],
    )

    if not chunks:
        raise ValueError(
            f"No chunks were created for {strategy['name']}."
        )

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    collection.add(
        ids=[f"chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "strategy": strategy["name"],
                "chunk_index": index,
                "chunk_size_target": strategy["chunk_size"],
                "overlap_target": strategy["overlap"],
            }
            for index in range(len(chunks))
        ],
    )

    print(
        f"{strategy['name'].capitalize()} strategy: "
        f"stored {collection.count()} chunks."
    )

    return collection


# ==================================================
# 7. RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_chunks(collection, model, question, top_k=TOP_K):
    """Retrieve the most relevant chunks for a question."""

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks),
        include=["documents", "distances", "metadatas"],
    )

    documents = results["documents"][0] or []
    distances = results["distances"][0] or []
    metadatas = results["metadatas"][0] or []

    retrieved = []

    for index, document in enumerate(documents):
        retrieved.append({
            "text": document,
            "distance": (
                distances[index]
                if index < len(distances)
                else None
            ),
            "metadata": (
                metadatas[index]
                if index < len(metadatas)
                else {}
            ),
        })

    return retrieved


# ==================================================
# 8. BASIC KEYWORD EVALUATION
# ==================================================

def normalize_words(text):
    """Convert text into a set of normalized words."""

    return {
        word.lower()
        for word in re.findall(r"\b\w+\b", text)
        if len(word) > 2
    }


def evaluate_relevance(question, retrieved_chunks):
    """
    Calculate keyword overlap for each retrieved chunk.

    This is a basic indicator, not a guarantee of answer accuracy.
    """

    question_words = normalize_words(question)
    scores = []

    for chunk in retrieved_chunks:
        chunk_words = normalize_words(chunk["text"])
        overlap = len(question_words & chunk_words)

        score = overlap / max(len(question_words), 1)
        scores.append(score)

    return scores


# ==================================================
# 9. COMPARE ALL THREE STRATEGIES
# ==================================================

def compare_retrieval(collections, model):
    """Evaluate and save the results of each strategy."""

    output_lines = [
        "DYNAMIC RAG - CHUNKING COMPARISON",
        "=" * 60,
        "Chunking method: question/paragraph-aware grouping",
        "Evaluation: keyword overlap and ChromaDB cosine distance",
        "",
        "Important: keyword Hit@3 is a simple metric. It does not "
        "prove that the retrieved chunk contains the correct answer.",
    ]

    for question in EVALUATION_QUESTIONS:
        print(f"\nQuestion: {question}")
        output_lines.append(f"\nQuestion: {question}")

        for strategy in STRATEGIES:
            collection = collections[strategy["name"]]

            retrieved = retrieve_chunks(
                collection=collection,
                model=model,
                question=question,
                top_k=TOP_K,
            )

            scores = evaluate_relevance(question, retrieved)
            hit_at_3 = int(any(score > 0 for score in scores))

            distances = [
                item["distance"]
                for item in retrieved
                if item["distance"] is not None
            ]

            top_distance = distances[0] if distances else None

            summary = (
                f"  {strategy['name'].upper()}: "
                f"chunks={collection.count()}, "
                f"keyword Hit@3={hit_at_3}, "
                f"top distance={top_distance}"
            )

            print(summary)
            output_lines.append(summary)

            for index, item in enumerate(retrieved, start=1):
                output_lines.append(
                    f"    Result {index} "
                    f"(distance={item['distance']}):"
                )
                output_lines.append(item["text"])
                output_lines.append("")

    RESULTS_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {RESULTS_PATH}")


# ==================================================
# 10. MAIN PROGRAM
# ==================================================

def main():
    print("Starting chunking comparison...")

    if not PDF_PATH.is_file():
        print("\nERROR: PDF file was not found.")
        print(f"Expected location: {PDF_PATH}")
        print("Place the PDF inside the project's data folder.")
        return

    print(f"Reading PDF: {PDF_PATH}")

    try:
        text = extract_pdf_text(PDF_PATH)
    except Exception as error:
        print(f"Could not read the PDF: {error}")
        return

    print("Detecting interview-question sections...")
    sections = split_into_sections(text)
    print(f"Detected {len(sections)} sections.")

    if not sections:
        print("No text sections were detected.")
        return

    print("Loading embedding model...")

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as error:
        print(f"Could not load the embedding model: {error}")
        return

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DATABASE_PATH)

    collections = {}

    for strategy in STRATEGIES:
        print(f"\nBuilding {strategy['name']} chunks...")

        try:
            collections[strategy["name"]] = ingest_chunks(
                client=client,
                model=model,
                text=text,
                strategy=strategy,
            )
        except Exception as error:
            print(
                f"Failed to build {strategy['name']} collection: "
                f"{error}"
            )
            raise

    compare_retrieval(collections, model)

    print("\nChunking comparison completed successfully!")


if __name__ == "__main__":
    main()
