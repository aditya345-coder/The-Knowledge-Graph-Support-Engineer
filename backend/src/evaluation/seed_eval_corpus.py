"""Seed a fixed evaluation corpus in Qdrant + Neo4j.

This script ingests a repository under a fixed session_id, creating the
persistent evaluation corpus that `evaluate.py --session-id` will query.

Usage:
    python -m evaluation.seed_eval_corpus \\
        --repo-url https://github.com/user/repo.git \\
        --session-id eval-v1 \\
        --confirm
"""

import argparse
import sys
from pathlib import Path

from settings import settings


def seed_corpus(
    repo_url: str,
    github_token: str | None,
    session_id: str,
    skip_code: bool = False,
    skip_graph: bool = False,
    force: bool = False,
    batch_size: int | None = None,
) -> None:
    """Ingest a repository into Qdrant and Neo4j under a fixed session_id."""
    local_path = str(settings.RAW_DOCS_DIR / f"eval_{session_id}")

    # Check if collection already exists
    if not force:
        from database.vector_store import VectorStore
        vs = VectorStore()
        if vs.has_session_collection(session_id):
            print(f"Warning: Collection docs_{session_id} already exists.")
            print("Re-seeding will duplicate data. Run with --force to proceed.")
            return

    # Clean up existing data if --force
    if force:
        from database.vector_store import VectorStore
        from database.graph_store import GraphStore
        vs = VectorStore()
        gs = GraphStore()
        vs.cleanup_session(session_id)
        gs.cleanup_session(session_id)
        print(f"Cleaned up existing data for session {session_id}")

    # Override batch size if provided
    if batch_size is not None:
        settings.DOCS_UPSERT_BATCH = batch_size

    # 1. Clone + parse docs
    from ingestion.docs_loader import DocsLoader
    docs_loader = DocsLoader(
        repo_url=repo_url,
        github_token=github_token,
        local_path=local_path,
        session_id=session_id,
    )
    chunks = docs_loader.load_and_split()
    docs_loader.upload_to_qdrant(chunks)
    print(f"Indexed {len(chunks)} document chunks into docs_{session_id}")

    # 2. Index code (optional)
    if not skip_code:
        from ingestion.code_indexer import CodeIndexer
        from database.vector_store import VectorStore
        from qdrant_client.models import Distance, VectorParams, PointStruct

        code_indexer = CodeIndexer()
        repo_root = Path(local_path)
        src_root = repo_root
        for candidate in [repo_root / "src", repo_root / "lib", repo_root]:
            if candidate.is_dir():
                src_root = candidate
                break

        code_chunks = code_indexer.index_directory(
            src_root,
            exclude_dirs=settings.AST_EXCLUDE_DIRS,
        )
        print(f"Found {len(code_chunks)} code symbols")

        if code_chunks:
            vector_store = VectorStore()
            collection_name = f"code_{session_id}"

            try:
                vector_store.client.get_collection(collection_name)
            except Exception:
                vector_store.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            try:
                vector_store.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="session_id",
                    field_schema="keyword",
                )
            except Exception:
                pass

            batch_size = batch_size or 64
            total = len(code_chunks)
            for start in range(0, total, batch_size):
                end = min(total, start + batch_size)
                batch = code_chunks[start:end]

                texts = [c.docstring or c.signature or c.symbol_name for c in batch]
                vectors = vector_store.embeddings.embed_documents(texts)

                points = []
                for j, chunk in enumerate(batch):
                    idx = start + j
                    points.append(PointStruct(
                        id=idx,
                        vector=vectors[j],
                        payload={
                            "text": texts[j],
                            "symbol_name": chunk.symbol_name,
                            "symbol_type": chunk.symbol_type,
                            "signature": chunk.signature,
                            "file_path": chunk.file_path,
                            "line_start": chunk.line_start,
                            "line_end": chunk.line_end,
                            "source_code": chunk.source_code,
                            "type": "code",
                            "session_id": session_id,
                        },
                    ))

                vector_store.client.upsert(collection_name=collection_name, points=points)
                print(f"Indexed code symbols ({end}/{total})")

            print(f"Indexed {len(code_chunks)} code symbols into code_{session_id}")

    # 3. Build Neo4j graph (optional)
    if not skip_graph:
        from ingestion.github_loader import GitHubGraphLoader
        gh_loader = GitHubGraphLoader(
            repo_url=repo_url,
            token=github_token,
            session_id=session_id,
        )
        gh_loader.run()
        print(f"Built Neo4j graph for session {session_id}")

    print(f"\nEval corpus '{session_id}' seeded with data from {repo_url}")
    print(f"Run: python -m evaluation.evaluate --session-id {session_id}")


def clean_corpus(session_id: str) -> None:
    """Drop all data for a session from Qdrant and Neo4j."""
    from database.vector_store import VectorStore
    from database.graph_store import GraphStore
    vs = VectorStore()
    gs = GraphStore()
    vs.cleanup_session(session_id)
    gs.cleanup_session(session_id)
    print(f"Cleaned up session {session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed evaluation corpus")
    parser.add_argument("--repo-url", default=settings.TARGET_REPO,
        help="Repository URL to ingest (default: TARGET_REPO)")
    parser.add_argument("--session-id", default=settings.EVAL_SESSION_ID,
        help="Session ID for the eval corpus (default: eval-v1)")
    parser.add_argument("--skip-code", action="store_true",
        help="Skip code indexing phase")
    parser.add_argument("--skip-graph", action="store_true",
        help="Skip Neo4j graph building phase")
    parser.add_argument("--confirm", action="store_true", required=True,
        help="Confirm you want to ingest data. Required for safety.")
    parser.add_argument("--force", action="store_true",
        help="Clean existing data before re-seeding")
    parser.add_argument("--clean", action="store_true",
        help="Drop all data for the session and exit")
    parser.add_argument("--batch-size", type=int, default=None,
        help="Override upsert batch size (default: use settings value, usually 64)")
    args = parser.parse_args()

    if args.clean:
        clean_corpus(args.session_id)
        sys.exit(0)

    if not args.repo_url:
        print("Error: --repo-url or TARGET_REPO environment variable is required.")
        sys.exit(1)

    seed_corpus(
        repo_url=args.repo_url,
        github_token=settings.GITHUB_TOKEN,
        session_id=args.session_id,
        skip_code=args.skip_code,
        skip_graph=args.skip_graph,
        force=args.force,
        batch_size=args.batch_size,
    )
