import streamlit as st

# Page title
st.title("💬 Simple Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Text input for user message
user_input = st.text_input("You:", "")

# When user submits
if st.button("Send") and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Dummy bot response (you can replace this with an LLM or API call)
    bot_reply = f"I heard you say: '{user_input}'"

    # Add bot message
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Clear input and rerun
    st.rerun()
