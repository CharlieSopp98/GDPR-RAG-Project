from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import FAISS
import os
from data_loader import main as create_db
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import argparse

FAISS_PATH = "FAISS_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    get_llm_response(query_text)

def load_db():
    embeddings = get_embedding_function()
    if os.path.exists(FAISS_PATH):
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Need to first create database, now running script to create FAISS database...")
        create_db()
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_query_context(query_text: str):
    db = load_db()
    results = db.similarity_search(query=query_text,k=5)
    sources = [doc.metadata['id'] for doc in results]

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context_text, sources

def get_llm_response(query_text: str):
    context_text, sources = retrieve_query_context(query_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print('\n --- \n Prompt: \n')
    print(prompt)

    model = Ollama(model="mistral")

    print("\nLoading response from mistral LLM (time taken will depend on hardware available)...")
    response_text = model.invoke(prompt)

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text

if __name__ == "__main__":
    main()
