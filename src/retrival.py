from src.vector_store import VectorStore

def retrieve():

    store = VectorStore()
    store.load()

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        docs = store.search(query, k=3)

        print("\nTop Retrieved Chunks:\n")

        for i, doc in enumerate(docs, 1):
            print(f"----- Result {i} -----")
            print(doc.page_content)
            print()

if __name__ == "__main__":
    retrieve()