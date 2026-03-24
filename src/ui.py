import streamlit as st
import requests

st.set_page_config(page_title="Omni-Support AI", layout="wide")
st.title("🤖 FastAPI Support Engineer (GraphRAG)")

# Sidebar for "Under the Hood" details
with st.sidebar:
    st.header("System Status")
    st.info("Model: Llama 3 (via Groq)\n\nDatabases: Neo4j + Qdrant")

user_input = st.chat_input("Describe your FastAPI issue...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing docs and GitHub history..."):
        # Calling our FastAPI backend
        try:
            api_response = requests.post(
                "http://localhost:8000/v1/solve-ticket", json={"user_query": user_input}
            )
            try:
                response = api_response.json()
            except ValueError:
                response = {"status": "error", "message": api_response.text}
        except requests.exceptions.RequestException as exc:
            response = {"status": "error", "message": f"API connection failed: {exc}"}

        if response.get("status") == "success":
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            # Show the reasoning in the sidebar
            metadata = response.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            with st.sidebar:
                st.success(f"Feature Detected: {metadata.get('detected_feature')}")
                st.write(f"📚 Documents Searched: {metadata.get('docs_retrieved')}")
                st.write(
                    f"🐞 GitHub Issues Linked: {metadata.get('github_issues_found')}"
                )
        else:
            error_message = (
                response.get("detail")
                or response.get("message")
                or "Unexpected error from API."
            )
            with st.chat_message("assistant"):
                st.error(error_message)
