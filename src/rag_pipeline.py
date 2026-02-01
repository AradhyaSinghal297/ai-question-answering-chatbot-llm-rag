from transformers import pipeline
from retrieval import search

# Load Question Answering pipeline (only once)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

def generate_answer(question, context_chunks):
    """
    Generate answer using QA model with retrieved context.
    """
    context = " ".join(context_chunks)

    if not context.strip():
        return "No relevant information found."

    result = qa_pipeline(
        question=question,
        context=context
    )

    return result["answer"]


if __name__ == "__main__":
    query = input("Enter your question: ")

    retrieved_chunks = search(query)

    if not retrieved_chunks:
        print("\n❌ No relevant context retrieved.")
    else:
        answer = generate_answer(query, retrieved_chunks)

        print("\n🤖 Final Answer:\n")
        print(answer)
