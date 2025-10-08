import streamlit as st

st.title("RAG with GTE-Large and Ollama")

query = st.text_input("Ask a question:")

if st.button("Ask"):
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = f"Answer using context:\n{context}\n\nQuestion: {query}"
    answer = ollama_generate(final_prompt)
    st.write(answer)
