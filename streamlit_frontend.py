# streamlit_frontend.py

import streamlit as st
from rag_llm_backend import rag

# st.markdown("<h1 style='text-align: center;'>Corporate Tax AI🖨️</h1>", unsafe_allow_html=True)
st.title("Corporate Tax AI🖨️")

st.markdown("""
<style>
.stApp {
    background-color: #D8BFD8;
}

</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_avatar = "👩‍💻"
assistant_avatar = "🤖"

if st.button("Clear Chat"):
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I help you?"):

    st.session_state["messages"].append({"role": "user", "content": prompt})


    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)


    response = rag(prompt)


    with st.chat_message("assistant", avatar=assistant_avatar):
        st.markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
