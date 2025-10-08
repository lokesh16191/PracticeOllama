import streamlit as st
import requests
import json

st.title("Chatbot By Lokesh")
st.write("Interact with your local Ollama model using Streamlit.")

# User input
prompt = st.text_area("Enter your prompt:", "write an email for sick Leave")

if st.button("Run"):
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen3:4b", "prompt": prompt},
                stream=True,
                timeout=60
            )

            output = ""
            placeholder = st.empty()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            output += data["response"]
                            placeholder.write(output)
                    except json.JSONDecodeError:
                        continue

            st.success("Done!")
            st.text_area("Full Output:", output, height=200)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama API: {e}")
