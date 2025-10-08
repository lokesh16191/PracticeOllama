import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# Load DB
embeddings = OllamaEmbeddings(model="turingdance/gte-large-zh:latest")
db = FAISS.load_local("vector_db_gte_large", embeddings, allow_dangerous_deserialization=True)

def ollama_generate(prompt, model="qwen3:4b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# Example query
query = "What is Lokesh wife name"

# Retrieve top 3 relevant chunks
docs = db.similarity_search(query, k=3)
context = "\n\n".join([d.page_content for d in docs])

final_prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

answer = ollama_generate(final_prompt)
print(answer)
