from pypdf import PdfReader
import chromadb
import ollama


from config import (
    DATA_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE
)

# Connect to datbase : chromaDB
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name = COLLECTION_NAME
)


# Split the Text into chunks
def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP
    
    return chunks


# extract text from pdf
def extract_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))
    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": pdf_path.name,
                "page": page_number,
                "chunk": chunk_index
            })
    return all_chunks


# Generate embeddings and store in ChromaDB
def store_chunks(chunks):
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        documents = [
            item["text"] for item in batch
        ]
        response = ollama.embed(
            model = EMBED_MODEL,
            input = documents
        )
        embeddings = response["embeddings"]

        ids = [
            f"{item['source']}_page_{item['page']}_chunk_{item['chunk']}"
            for item in batch
        ]

        metadatas = [
            {
                "source": item["source"],
                "page": item["page"],
                "chunk": item["chunk"]
            }
            for item in batch
        ]

        collection.upsert(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas
            
        )

        print(f"Stored {len(batch)} chunks")


# Main ingestion function

def main():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data directory.")
        return

    total_chunks = []

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}")
        chunks = extract_pdf(pdf_path)
        total_chunks.extend(chunks)

    if not total_chunks:
        print("No text extracted from the PDF files.")
        return

    print(f"Total chunks: {len(total_chunks)}")
    store_chunks(total_chunks)
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()