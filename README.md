# Omni-Support GraphRAG: FastAPI Support Engineer 🤖

An agentic AI system designed to act as a world-class support engineer for FastAPI. It uses a **Hybrid GraphRAG** approach, combining semantic documentation search (Vector DB) with historical issue/bug relationship mapping (Graph DB) to provide grounded, expert-level answers.

---

## 🏗️ Architecture Overview

The system follows a multi-layered agentic architecture built with **LangGraph**:

1.  **The Ingestion Pipeline (ETL):**
    *   **Vector DB (Qdrant):** Stores embeddings of the official FastAPI documentation.
    *   **Graph DB (Neo4j):** Stores a knowledge graph of GitHub issues, linked features, and version history.
2.  **The Agentic Core:**
    *   **Router:** Analyzes the user query to identify which FastAPI features are involved.
    *   **Hybrid Retriever:** Simultaneously queries Qdrant (for "How-to") and Neo4j (for "Known Bugs").
    *   **The Critic Node:** A specialized verification step that checks for hallucinations and ensures the answer is grounded in the retrieved data.
3.  **Interaction Layer:**
    *   **FastAPI Backend:** Orchestrates the LangGraph state machine.
    *   **Streamlit UI:** Provides a premium, chat-based interface for the user.

---

## 🚀 Setup Instructions

### 1. Prerequisites
*   Python 3.13+
*   [Qdrant](https://qdrant.tech/) Account (Cloud or Local)
*   [Neo4j](https://neo4j.com/cloud/aura24/) Aura Account (Free Tier works great)
*   [Groq](https://console.groq.com/) API Key (for high-speed Llama 3 inference)

### 2. Clone and Install
```bash
git clone https://github.com/your-username/The-Knowledge-Graph-Support-Engineer.git
cd The-Knowledge-Graph-Support-Engineer

# Install dependencies
pip install -e .
```

### 3. Environment Configuration
Create a `.env` file in the root directory and populate it with your keys:
```env
# LLM
LLM_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key

# Logging
LOG_LEVEL=INFO
# Local only: write logs to file (stdout always enabled)
LOG_TO_FILE=true
LOG_FILE=logs/app.log

# Qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Neo4j
NEO4J_URI=neo4j+s://your_instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# GitHub (for ingestion)
GITHUB_TOKEN=your_github_pat
TARGET_REPO=tiangolo/fastapi
```

---

## 🛠️ How to Run

### Phase 1: Ingest Data (Do this once)
First, populate your databases with the documentation and GitHub issues:
```bash
# Ingest Documentation into Qdrant (creates fastapi_docs collection)
python src/ingestion/docs_loader.py

# Ingest GitHub Issues into Neo4j
python src/ingestion/github_loader.py
```

### Phase 2: Start the System
You need to run both the backend API and the frontend UI:

1.  **Start the Backend API:**
    ```bash
    python src/main.py
    ```
2.  **Start the Streamlit UI:**
    ```bash
    streamlit run src/ui.py
    ```

Notes:
- If the API returns `Collection fastapi_docs not found`, run the docs ingestion step above.

---

## 📺 Demo
*A video demo or screenshot gallery will be placed here soon.*

---

## 🛡️ Key Features
*   **Zero-Hallucination Design:** The LangGraph "Critic" node loops back if the answer isn't grounded.
*   **LiteLLM Integration:** Swap between Llama 3, GPT-4, or Claude with a single `.env` change.
*   **Relational Context:** Knows not just *how* a feature works, but *what bugs* are currently affecting it.
