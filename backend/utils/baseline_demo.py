from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from logger import QueryLogger


def run_baseline_demo():
    print("=" * 60)
    print("BASELINE RAG PIPELINE DEMO (TERMINAL)")
    print("=" * 60)

    # Initialize components
    vector_store = VectorStore()
    logger = QueryLogger()

    rag = RAGPipeline(
        model_path="",
        vector_store=vector_store,
        logger=logger,
        max_tokens=200,
        temperature=0.7
    )

    # Minimal baseline documents
    documents = [
        "Retrieval Augmented Generation combines document retrieval with language models.",
        "FAISS is used for fast similarity search over vector embeddings.",
        "Sentence Transformers generate dense embeddings for text.",
        "Large Language Models generate answers using contextual information."
    ]

    metadata = [
        {"file_path": "rag_intro.txt", "type": "text"},
        {"file_path": "faiss.txt", "type": "text"},
        {"file_path": "embeddings.txt", "type": "text"},
        {"file_path": "llm.txt", "type": "text"}
    ]

    # Add documents (baseline ingestion)
    vector_store.add_documents(documents, metadata)

    # Single baseline query
    question = "What is Retrieval Augmented Generation?"

    print("\nQuestion:")
    print(question)

    result = rag.query(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved Sources:")
    for src in result["sources"]:
        print(f"- {src['file']} | similarity={src['similarity']}")

    print("\nProcessing Time (seconds):")
    print(result["processing_time"])

    print("\nBaseline RAG pipeline executed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_baseline_demo()
