from embedder import Embedder
from vectorstore import VectorStore
from retriever import Retriever


DOCUMENTS = [
    "RAG stands for Retrieval Augmented Generation.",
    "FAISS is used for vector similarity search.",
    "Embeddings convert text into numbers.",
    "PyCharm is a Python IDE.",
    "Infosys is an IT services company."
]


def main():
    embedder = Embedder()

    doc_vectors = embedder.embed(DOCUMENTS)

    vector_store = VectorStore(dimension=doc_vectors.shape[1])
    vector_store.add(doc_vectors, DOCUMENTS)

    retriever = Retriever(embedder, vector_store)

    print("\nSimple RAG Retriever Ready (type 'exit')\n")

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break

        results = retriever.retrieve(query)
        print("Retrieved documents:")
        for r in results:
            print("-", r)
        print()


if __name__ == "__main__":
    main()
