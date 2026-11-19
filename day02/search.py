import chromadb
import ollama

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K
)

# connect to chromaDB

client = chromadb.PersistentClient(
    path = str(DB_DIR)
)

collection = client.get_or_create_collection(
    name = COLLECTION_NAME
)


# Retrieve relevant documents chunks

def retrieve_context(question):

    # convert question into an embedding
    response = ollama.embed(
        model = EMBED_MODEL,
        input = question
    )

    query_embedding = response["embeddings"][0]

    # Search chromaDB
    if collection.count() == 0:
        return "", []
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = min(TOP_K, collection.count()),
        include = ["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]


    context = ""
    sources = []

    for document, metadata in zip(documents, metadatas):

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        context += f"""
        
source: {source}
page: {page}

content: {document}

-------------------------
"""
        sources.append({
            "source": source,
            "page": page
        })

    return context, sources

# Test retrieval independently

if __name__ == "__main__":

    question = input("Enter your question: ")

    context, sources = retrieve_context(question)

    print("\nRetrieved Context:")
    print(context)

    print("\nRetrieved Sources:")
    for source in sources:
        print(source)

