from src.pdf_reader import PdfReader
from src.textchunker import Chunker
from src.vector_store import VectorStore

def build_index():

    reader = PdfReader("./data/pdf_files/graph.pdf")

    text = reader.extract_text()

    chunker = Chunker()

    docs = chunker.chunk_text(text)

    store = VectorStore()

    store.create_index(docs)

    store.save()

    print("Index Saved Successfully")

if __name__=="__main__":
    build_index()