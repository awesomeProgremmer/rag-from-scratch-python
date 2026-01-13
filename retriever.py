class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, k=2):
        query_vector = self.embedder.embed([query])
        return self.vector_store.search(query_vector, k)
