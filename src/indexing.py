from src.pdf_reader import PdfReader
from src.textchunker import Chunker
from src.vector_store import VectorStore

def build_index(pdf_path , source=None):

    reader = PdfReader(pdf_path)

    text = reader.extract_text()

    chunker = Chunker()

    docs = chunker.chunk_text(
    text,
    source=source
)

    store = VectorStore()

    store.create_index(docs)

    store.save()

    return store

