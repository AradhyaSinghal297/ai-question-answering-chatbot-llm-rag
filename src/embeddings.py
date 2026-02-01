from sentence_transformers import SentenceTransformer
import faiss
import os

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    text_path = "data/processed/cleaned_text.txt"
    text = load_text(text_path)

    chunks = split_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss_index")

    with open("vector_store/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    print("✅ Embeddings created and stored in FAISS vector database.")
