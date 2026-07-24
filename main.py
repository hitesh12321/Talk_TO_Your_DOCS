from src.pdf_reader import PdfReader
from src.textchunker import Chunker


def main():
    print("Hello from rag-1!")
    pdf = PdfReader("./data/pdf_files/graph.pdf")

    text = pdf.extract_text()

    chunker = Chunker(
    chunk_size=500,
    chunk_overlap=100
)

    docs = chunker.chunk_text(text)

    print(len(docs))

    print(docs[0].page_content)


if __name__ == "__main__":
    main()
