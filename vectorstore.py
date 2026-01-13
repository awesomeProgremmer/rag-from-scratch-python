import faiss


class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, vectors, documents):
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_vector, k=2):
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[i] for i in indices[0]]

