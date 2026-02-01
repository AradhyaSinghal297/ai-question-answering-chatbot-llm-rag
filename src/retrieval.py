from sentence_transformers import SentenceTransformer
import faiss

def load_chunks():
    with open("vector_store/chunks.txt", "r", encoding="utf-8") as file:
        return file.readlines()

def search(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    index = faiss.read_index("vector_store/faiss_index")
    distances, indices = index.search(query_embedding, top_k)

    chunks = load_chunks()
    results = [chunks[i] for i in indices[0]]

    return results


if __name__ == "__main__":
    query = input("Enter your question: ")
    results = search(query)

    print("\n🔍 Relevant Information Retrieved:\n")
    for res in results:
        print("-", res.strip())
