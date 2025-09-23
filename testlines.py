import streamlit as st

st.set_page_config(page_title="Test CSS", layout="wide")

# Simple visible UI
st.title("Hello World 👋")
st.header("Header text – should appear")
st.write("Body text visible here")

st.sidebar.title("Sidebar Title")
st.sidebar.write("Sidebar content visible here")
