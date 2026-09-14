from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunker:

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text, source=None):

        chunks = self.splitter.split_text(text)

        documents = []

        for i, chunk in enumerate(chunks):

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "chunk_id": i,
                        "chunk_size": len(chunk)
                    }
                )
            )

        return documents