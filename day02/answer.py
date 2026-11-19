import ollama

from config import LLM_MODEL
from search import retrieve_context


# Generate an answer using the LLM

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI tutor.

Answer the user's question using ONLY
the information provided in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context, say:
"I could not find this information in the provided docs."
4. Explain concepts in simple language.
5. Be clear and concise.
6. Do not invent source references.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model = LLM_MODEL,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        options = {
            "temperature": 0.2
        }
    )
    return response["messages"]["content"]


# main application

def main():
    print("=" * 45)
    print("My RAG AI Assistant")
    print("=" * 45)

    print("\nAsk questions about the pdf documents.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue
        try:

            #step 01: retrieve relevant context

            context, sources = retrieve_context(question)

            if not context.strip():
                print("No relevant context found.")
                continue

            #step 02: generate answer using LLM

            answer = generate_answer(question, context)

            #step 03: display answer
            print("\nAI: " + answer)

            #step 04: display actual sources
            print("\nSources:")

            unique_sources = {
                (source["source"], source["page"])
                for source in sources
            }

            for source, page in sorted(unique_sources):
                print(f"- {source} | page {page}")

        except Exception as error:
            print(f"\nError: {error}")

if __name__ == "__main__":
    main()
