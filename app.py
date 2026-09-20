import streamlit as st
import requests

st.title("Agentic RAG Chatbot")

query = st.text_input("Ask me anything:")
if st.button("Send"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"query": query})
    st.write(response.json()["response"])
