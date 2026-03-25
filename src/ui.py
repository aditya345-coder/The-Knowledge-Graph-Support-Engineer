import os
import re
import streamlit as st
import requests

from utils.logging_config import setup_logging

st.set_page_config(page_title="Omni-Support AI", layout="wide")
st.title("🤖 FastAPI Support Engineer (GraphRAG)")

logger = setup_logging(__name__)


def format_citations(text: str) -> str:
    repo = os.getenv("TARGET_REPO")

    def replace_tag(match: re.Match[str]) -> str:
        source = match.group(1).strip()
        if source.startswith("Issue #"):
            issue_id = source.replace("Issue #", "").strip()
            if repo and issue_id.isdigit():
                url = f"https://github.com/{repo}/issues/{issue_id}"
                return f"🔗[Source: Issue #{issue_id}]({url})"
            return f"[Source: Issue #{issue_id}]"
        return f"📖[Source: {source}]"

    # return re.sub(r"\[Source: ([^\]]+)\]", replace_tag, text)
    return re.sub(r"\[Source: (.*?)\]", replace_tag, text)


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
                "http://127.0.0.1:8000/v1/solve-ticket", json={"user_query": user_input}
            )
            try:
                response = api_response.json()
            except ValueError:
                logger.warning(
                    "Non-JSON API response", extra={"status": api_response.status_code}
                )
                response = {"status": "error", "message": api_response.text}
        except requests.exceptions.RequestException as exc:
            logger.error("API connection failed", exc_info=exc)
            response = {"status": "error", "message": f"API connection failed: {exc}"}

        if response.get("status") == "success":
            st.write("DEBUG: Raw LLM Output:", response["answer"]) # <--- ADD THIS
            with st.chat_message("assistant"):
                # st.markdown(format_citations(response["answer"]))
                formatted_text = format_citations(response["answer"])
                st.markdown(formatted_text)

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
