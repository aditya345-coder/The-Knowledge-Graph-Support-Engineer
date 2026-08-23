from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    # ── Ingestion Mode ──────────────────────────────────────────
    LOCAL_MODE: bool = os.getenv("LOCAL_MODE", "F").upper() == "T"

    # ── Limits ──────────────────────────────────────────────────
    REPO_FILE_LIMIT: int = int(os.getenv("REPO_FILE_LIMIT", "500"))
    RATE_LIMIT_MAX: int = int(os.getenv("RATE_LIMIT_MAX", "50"))
    CLEANUP_RATE_LIMIT: int = int(os.getenv("CLEANUP_RATE_LIMIT", "5"))
    MAX_ISSUES_FETCHED: int = int(os.getenv("MAX_ISSUES_FETCHED", "20"))
    MAX_VERIFY_RETRIES: int = int(os.getenv("MAX_VERIFY_RETRIES", "3"))
    DOCS_UPSERT_BATCH: int = int(os.getenv("DOCS_UPSERT_BATCH", "64"))
    ONNX_THREADS: int = int(os.getenv("ONNX_THREADS", "1"))

    # ── AST Code Indexing ───────────────────────────────────────
    AST_ENABLED: bool = os.getenv("AST_ENABLED", "T").upper() == "T"
    AST_EXCLUDE_DIRS: list[str] = (
        os.getenv("AST_EXCLUDE_DIRS", "tests,examples,docs,node_modules,__pycache__,.git")
        .split(",")
    )

    # ── Server ──────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list[str] = (
        os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        .split(",")
    )
    CORS_METHODS: list[str] = (
        os.getenv("CORS_METHODS", "GET,POST").split(",")
    )
    CORS_HEADERS: list[str] = (
        os.getenv("CORS_HEADERS", "Content-Type,Authorization,X-Session-Id").split(",")
    )

    # ── LLM ─────────────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "z-ai/glm4.7")
    NVIDIA_API_KEY: str | None = os.getenv("NVIDIA_API_KEY")

    # ── Qdrant ──────────────────────────────────────────────────
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "docs_default")

    # ── Neo4j ───────────────────────────────────────────────────
    NEO4J_URI: str | None = os.getenv("NEO4J_URI")
    NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD")

    # ── Redis ───────────────────────────────────────────────────
    REDIS_URL: str | None = os.getenv("REDIS_URL")

    # ── Auth0 ───────────────────────────────────────────────────
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "F").upper() == "T"
    AUTH0_DOMAIN: str | None = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE: str | None = os.getenv("AUTH0_AUDIENCE")

    # ── Webhook ─────────────────────────────────────────────────
    WEBHOOK_SESSION_ID: str | None = os.getenv("WEBHOOK_SESSION_ID")
    WEBHOOK_SECRET: str | None = os.getenv("WEBHOOK_SECRET")

    # ── Dev Fallbacks (user-provided at runtime, optional here) ──
    GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
    TARGET_REPO: str | None = os.getenv("TARGET_REPO")
    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

    # ── Evaluation ─────────────────────────────────────────────────
    EVAL_SESSION_ID: str = os.getenv("EVAL_SESSION_ID", "eval-v1")
    EVAL_REPO_URL: str | None = os.getenv("TARGET_REPO")

    # ── Paths ───────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DOCS_DIR: Path = DATA_DIR / "raw_docs"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # ── Logging ─────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    LOG_FILE: str = os.getenv(
        "LOG_FILE",
        str(Path(__file__).resolve().parent.parent / "logs" / "app.log"),
    )

    # ── LangSmith ───────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT: str | None = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str | None = os.getenv("LANGCHAIN_PROJECT")

    # ── Checkpointer ────────────────────────────────────────────
    CHECKPOINTER: str = os.getenv("CHECKPOINTER", "memory")
    SQLITE_PATH: str = os.getenv(
        "SQLITE_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"),
    )
    POSTGRES_URI: str | None = os.getenv("POSTGRES_URI")


settings = Settings()
