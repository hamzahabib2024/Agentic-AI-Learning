
import chromadb
import ollama

EMBED_MODEL = "embeddinggemma"

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="university_notes"
)


def search_documents(question):

    # Check database
    total = collection.count()

    if total == 0:
        print("Database is empty. Run ingest.py first.")
        return

    # Convert question into an embedding
    response = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )

    question_vector = response["embeddings"][0]

    # Search similar vectors
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(4, total),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\nRelevant information:\n")

    # Display results
    for i, (document, metadata, distance) in enumerate(
        zip(   # zip function is used to combine the three lists into a single iterable, where each element is a tuple containing one item from each list. In this case, it combines the first document, metadata, and distance from the results of the query.
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        print(f"\n--- Result {i} ---")

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])

        print("Distance:", round(distance, 4))

        print("Content:")
        print(document)

        print("-" * 50)


if __name__ == "__main__":   #this is the main entry point of the script, it means that if this script/python_file is run directly, the code inside this block will be executed. If the script is imported as a module in another script, the code inside this block will not be executed.

    while True:

        question = input(
            "\nAsk your university notes (or type exit): "
        )

        if question.lower().strip() == "exit":
            break

        if question.strip():
            search_documents(question)
