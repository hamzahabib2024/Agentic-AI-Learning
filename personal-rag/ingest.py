
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

# Configuration
DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")

EMBED_MODEL = "embeddinggemma"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30
BATCH_SIZE = 32

# Create local persistent database
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="university_notes",
    metadata={"hnsw:space": "cosine"}
)


# 1. Split text into overlapping word chunks
def split_text(text):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# 2. Read PDFs and extract text
def ingest_documents():

    DATA_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found inside data/")
        return

    for pdf_path in pdf_files:

        print(f"\nReading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        documents = []
        metadatas = []
        ids = []

        for page_num, page in enumerate(
            reader.pages, start=1
        ):

            text = page.extract_text() or ""

            chunks = split_text(text)

            for chunk_num, chunk in enumerate(chunks, start=1):

                documents.append(chunk)

                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_num
                })

                ids.append(
                    f"{pdf_path.stem}-p{page_num}-c{chunk_num}"
                )

        if not documents:
            print("No extractable text found.")
            continue

        # Remove previous version of this PDF
        collection.delete(
            where={"source": pdf_path.name}
        )

        # 3. Generate embeddings in batches
        for start in range(0, len(documents), BATCH_SIZE):

            batch_docs = documents[start:start + BATCH_SIZE]

            batch_meta = metadatas[start:start + BATCH_SIZE]

            batch_ids = ids[start:start + BATCH_SIZE]

            response = ollama.embed(
                model=EMBED_MODEL,
                input=batch_docs
            )

            embeddings = response["embeddings"]

            # 4. Store text, vectors and metadata
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings
            )

        print(f"Stored {len(documents)} chunks.")

    print("\nDocument ingestion completed!")


if __name__ == "__main__":
    ingest_documents()
