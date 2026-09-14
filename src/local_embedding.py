from langchain_huggingface import HuggingFaceEmbeddings

class LocalEmbedding:

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}   
        )

    def embed_documents(self, documents):

        texts = [doc.page_content for doc in documents]

        embeddings = self.embedding_model.embed_documents(texts)

        return embeddings

    def embed_query(self, query):

        return self.embedding_model.embed_query(query)