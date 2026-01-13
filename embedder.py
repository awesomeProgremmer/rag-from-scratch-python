from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        vectors = self.model.encode(texts)
        return np.array(vectors).astype("float32")
