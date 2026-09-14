from langchain_community.vectorstores import FAISS
from src.local_embedding import LocalEmbedding
from langchain_community.vectorstores.utils import DistanceStrategy

class VectorStore:

    def __init__(self):

        self.embedding = LocalEmbedding().embedding_model
        self.db = None

    def create_index(self, documents):

        self.db = FAISS.from_documents(
            documents,
            self.embedding,
            distance_strategy=DistanceStrategy.COSINE
        )

    def save(self, path="./vector_db"):

        self.db.save_local(path)

    def load(self, path="./vector_db"):

        self.db = FAISS.load_local(
            path,
            self.embedding,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE
        )

    def search(self, query, k=4):

        return self.db.similarity_search(query, k=k)