
def retrieve_chunks(collection, model, question, top_k=3):
    """
    Retrieve relevant document chunks from ChromaDB.
    """

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # Generate a normalized query embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve the most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, total_chunks),
        include=["documents", "distances", "metadatas"]
    )

    documents = results["documents"][0] or []
    distances = results["distances"][0] or []
    metadatas = results["metadatas"][0] or []

    # Return structured results
    retrieved_chunks = []

    for i, document in enumerate(documents):
        retrieved_chunks.append({
            "text": document,
            "distance": distances[i] if i < len(distances) else None,
            "metadata": metadatas[i] if i < len(metadatas) else {}
        })

    return retrieved_chunks
