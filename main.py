import streamlit as st
from helper import create_llm
import os

# Custom CSS to style the app
st.markdown("""
<style>
.main-container {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("GitHub Document Summarizer")

# Sidebar inputs
st.sidebar.header("Input Parameters")
query = st.sidebar.text_input("Query:", value="Enter your search query")
repo = st.sidebar.text_input("GitHub Repository:", value="username/repository")

# Main functionality
if st.sidebar.button("Generate Summary"):
    with st.spinner('Processing...'):
        try:
            # Assuming create_llm returns a string summary
            summary = create_llm(query, repo)
            st.markdown(f"### Summary for *{query}* in *{repo}*", unsafe_allow_html=True)
            st.markdown(summary, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Enter your search query.")
st.sidebar.markdown("2. Enter the GitHub repository in the format `username/repository`.")
st.sidebar.markdown("3. Click on 'Generate Summary' to process.")

# About
st.sidebar.markdown("### About")
st.sidebar.info("This app summarizes documents from a specified GitHub repository based on a search query.")