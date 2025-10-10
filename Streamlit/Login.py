import streamlit as st

# 🌈 Page Configuration
st.set_page_config(page_title="User Registration", page_icon="🧍", layout="centered")


# 🧭 App title
st.markdown("<div class='title'>🧍 User Registration Form</div>", unsafe_allow_html=True)

# 🧾 Input Form
with st.container():
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    gender = st.radio("⚧ Gender", ["Male", "Female", "Other"], horizontal=True)
    age = st.selectbox("🎂 Age", list(range(10, 101)))
    privacy = st.checkbox("I agree to the Privacy Policy")

    # 🧩 Two Buttons: Register & Reset
    col1, col2 = st.columns(2)

    with col1:
        register = st.button("Register", key="register", use_container_width=True)
    with col2:
        reset = st.button("Reset", key="reset", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# 🧠 Logic
if register:
    if not username:
        st.warning("⚠️ Please enter your username.")
    elif not privacy:
        st.error("🚫 You must agree to the Privacy Policy before registering.")
    else:
        st.success(f"✅ Registration successful! Welcome **{username}** ({gender}, {age} years old).")

if reset:
    # This just refreshes the app
    st.rerun()
