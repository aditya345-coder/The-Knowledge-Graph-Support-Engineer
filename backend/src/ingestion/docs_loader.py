# src/ingestion/docs_loader.py
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from git import Repo
from git.exc import GitCommandError
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from database.graph_store import GraphStore
from database.vector_store import VectorStore
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils.logging_config import setup_logging

from settings import settings

logger = setup_logging(__name__)


class DocsLoader:
    def __init__(
        self,
        repo_url: str | None = None,
        github_token: str | None = None,
        local_path: str | None = None,
        session_id: str | None = None,
        progress_cb: Callable[[str, int, int, str], None] | None = None,
    ):
        # Prefer explicit input; fall back to env to stay repo-agnostic.
        self.repo_url = (repo_url or settings.TARGET_REPO or "").strip() or None
        self.github_token = github_token or settings.GITHUB_TOKEN
        self.local_path = (
            Path(local_path)
            if local_path
            else (settings.RAW_DOCS_DIR / self._default_repo_dir_name(self.repo_url))
        )
        self.vector_store = VectorStore()
        self.session_id = session_id
        self.progress_cb = progress_cb
        self._api_contents: dict[str, str] = {}
        self._feature_embeddings: list[tuple[str, list[float]]] | None = None
        self._graph_store_instance: GraphStore | None = None

    def _default_repo_dir_name(self, repo_url: str | None) -> str:
        """Derive a stable, filesystem-safe directory name for the repo."""
        raw = (repo_url or "").strip()
        if not raw:
            return "repo"

        # Handle full URLs like https://github.com/owner/name(.git)
        if raw.startswith("http://") or raw.startswith("https://"):
            parts = [p for p in raw.rstrip("/").split("/") if p]
            if len(parts) >= 2:
                raw = "/".join(parts[-2:])

        raw = raw.removesuffix(".git")
        # Filesystem-safe: keep alnum, dot, dash, underscore; map others to underscore.
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw)
        return safe or "repo"

    def _progress(self, phase: str, current: int, total: int, message: str) -> None:
        cb = self.progress_cb
        if cb:
            try:
                cb(phase, current, total, message)
            except Exception:
                # Best-effort only; ingestion must continue.
                pass

    @staticmethod
    def _strip_github_suffixes(url: str) -> str:
        """Strip /tree/branch, /blob/path, /pull/N etc from GitHub URLs."""
        url = url.removesuffix(".git")
        for marker in ("/tree/", "/blob/", "/pull/", "/issues/"):
            idx = url.find(marker)
            if idx != -1:
                url = url[:idx]
        return url

    def _clone_url(self) -> str:
        repo = (self.repo_url or "").strip()
        if not repo:
            raise ValueError("repo_url must be provided (or set TARGET_REPO)")
        if repo.startswith("http://") or repo.startswith("https://"):
            return self._strip_github_suffixes(repo)
        # Accept owner/name (UI default) and convert to a cloneable HTTPS URL.
        return f"https://github.com/{repo}.git"

    def _repo_owner_name(self) -> str:
        url = self.repo_url or settings.TARGET_REPO or ""
        url = self._strip_github_suffixes(url)
        if url.startswith("http"):
            parts = [p for p in url.rstrip("/").split("/") if p]
            if len(parts) >= 2:
                return "/".join(parts[-2:])
        return url

    def fetch_via_api(self) -> list[dict]:
        from github import Github

        gh_token = self.github_token or settings.GITHUB_TOKEN
        gh = Github(gh_token)
        repo_name = self._repo_owner_name()
        if repo_name.count("/") != 1 or not all(part.strip() for part in repo_name.split("/")):
            raise ValueError(
                f"Invalid GitHub repo format: '{repo_name}'. "
                "Expected 'owner/repo' (e.g. 'pytorch/pytorch')."
            )
        repo = gh.get_repo(repo_name)

        def walk_contents(contents) -> list[dict]:
            files = []
            for item in contents:
                if item.type == "dir":
                    try:
                        sub = repo.get_contents(item.path)
                        files.extend(walk_contents(sub))
                    except Exception:
                        continue
                elif item.name.endswith((".md", ".rst", ".mdx")):
                    try:
                        content = item.decoded_content.decode("utf-8")
                        files.append({
                            "path": item.path,
                            "name": item.name,
                            "content": content,
                        })
                    except Exception:
                        continue
            return files

        md_files = []

        for candidate_dir in ["docs", "documentation"]:
            try:
                contents = repo.get_contents(candidate_dir)
                md_files = walk_contents(contents)
                if md_files:
                    break
            except Exception:
                continue
        else:
            try:
                contents = repo.get_contents("")
                for item in ([contents] if not isinstance(contents, list) else contents):
                    if item.name.endswith((".md", ".rst", ".mdx")) and item.type == "file":
                        content = item.decoded_content.decode("utf-8")
                        md_files.append({
                            "path": item.path,
                            "name": item.name,
                            "content": content,
                        })
            except Exception:
                pass

        return md_files

    def prepare_local_repo(self) -> str:
        """Ensure repository exists locally (clone if missing).

        Uses shallow clone (depth=1) for faster initial clones.
        Caches the clone for 1 hour — skips fetch if the repo was cloned recently.
        """
        local = self.local_path
        cache_marker = local / ".cloned_at"
        cache_ttl_seconds = 3600  # 1 hour

        if local.is_dir() and (local / ".git").is_dir():
            # Skip fetch if cloned within the last hour
            if cache_marker.is_file():
                try:
                    cloned_at = float(cache_marker.read_text().strip())
                    if (time.time() - cloned_at) < cache_ttl_seconds:
                        logger.info("Repo cache hit, skipping fetch", extra={"local_path": str(local)})
                        return str(local)
                except (ValueError, OSError):
                    pass
            try:
                repo = Repo(str(local))
                repo.remotes.origin.fetch()
                cache_marker.write_text(str(time.time()))
            except Exception:
                pass
            return str(local)

        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            Repo.clone_from(self._clone_url(), str(local), depth=1)
            cache_marker.write_text(str(time.time()))
        except GitCommandError as e:
            logger.error(
                "Git clone failed",
                extra={
                    "repo_url": self.repo_url,
                    "clone_url": self._clone_url(),
                    "local_path": str(local),
                    "stderr": getattr(e, "stderr", ""),
                },
            )
            raise
        except Exception as e:
            logger.error("Failed to clone repository", extra={"repo_url": self.repo_url, "err": str(e)})
            raise
        return str(local)

    def discover_docs_path(self, local_path: str) -> tuple[str, list[str]]:
        """Find docs/README/root markdown to ingest.

        Returns:
        - base_dir: directory to walk
        - include_files: explicit files to include if we can't walk a docs tree
        """
        # Common docs directories (in priority order)
        base = Path(local_path)
        candidates = [base / "docs", base / "documentation"]
        for c in candidates:
            if c.is_dir():
                return str(c), []

        include_files: list[str] = []
        for ext in (".md", ".rst", ".mdx"):
            readme = base / f"README{ext}"
            if readme.is_file():
                include_files.append(str(readme))
                break

        # Fallback: ingest documentation files in repo root.
        try:
            for p in base.iterdir():
                if p.is_file() and p.name.lower().endswith((".md", ".rst", ".mdx")):
                    include_files.append(str(p))
        except Exception:
            pass

        return str(base), include_files

    def _load_feature_embeddings(self) -> None:
        """Fetch features from Neo4j and pre-compute their embeddings."""
        if self._feature_embeddings is not None:
            return

        try:
            gs = self._graph_store_instance or GraphStore()
            self._graph_store_instance = gs
            feature_names = gs.get_all_feature_names(session_id=self.session_id)
        except Exception:
            logger.debug("Could not connect to Neo4j for feature names", exc_info=True)
            self._feature_embeddings = []
            return

        if not feature_names:
            self._feature_embeddings = []
            return

        vectors = self.vector_store.embeddings.embed_documents(feature_names)
        self._feature_embeddings = list(zip(feature_names, vectors))
        logger.info(
            "Loaded feature embeddings",
            extra={"feature_count": len(feature_names), "features": feature_names},
        )

    def _best_matching_feature(self, text: str) -> str:
        """Find the closest Neo4j feature name for a doc chunk text.

        Returns the best matching feature name, or 'General' if no good match.
        """
        if self._feature_embeddings is None:
            self._load_feature_embeddings()

        if not self._feature_embeddings:
            return "General"

        text_vec = self.vector_store.embeddings.embed_query(text)
        text_vec = np.array(text_vec)

        best_feature = "General"
        best_score = 0.0
        threshold = 0.5

        for name, feat_vec in self._feature_embeddings:
            feat_vec = np.array(feat_vec)  # type: ignore[assignment]
            cos_sim = np.dot(text_vec, feat_vec) / (
                np.linalg.norm(text_vec) * np.linalg.norm(feat_vec)
            )
            if cos_sim > best_score:
                best_score = cos_sim
                best_feature = name

        return best_feature if best_score >= threshold else "General"

    def _best_matching_features_batch(self, texts: list[str]) -> list[str]:
        """Batch version of _best_matching_feature. Embeds all texts at once
        and computes vectorized cosine similarity against feature embeddings.

        Returns a list of best-matching feature names, one per input text.
        """
        if self._feature_embeddings is None:
            self._load_feature_embeddings()

        if not self._feature_embeddings:
            return ["General"] * len(texts)

        if not texts:
            return []

        # Batch embedding to avoid ONNX Runtime BFC arena OOM
        batch_size = settings.DOCS_UPSERT_BATCH
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(self.vector_store.embeddings.embed_documents(batch))
        text_vecs = np.array(all_vectors)

        # Build feature matrix: shape (n_features, embedding_dim)
        feat_names = [name for name, _ in self._feature_embeddings]
        feat_matrix = np.array([vec for _, vec in self._feature_embeddings])

        # Vectorized cosine similarity: (n_texts, n_features)
        text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
        feat_norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        text_norms = np.where(text_norms == 0, 1, text_norms)
        feat_norms = np.where(feat_norms == 0, 1, feat_norms)
        similarity = (text_vecs @ feat_matrix.T) / (text_norms * feat_norms.T)

        threshold = 0.5
        results = []
        for i in range(len(texts)):
            best_idx = int(np.argmax(similarity[i]))
            best_score = float(similarity[i, best_idx])
            if best_score >= threshold:
                results.append(feat_names[best_idx])
            else:
                results.append("General")
        return results

    def identify_feature(self, text: str, source_path: str = "") -> str:
        """Identify the most likely feature name for a doc chunk.

        Uses embedding similarity to match against existing Neo4j Feature nodes first.
        Falls back to path-based inference from directory structure.
        Falls back to 'General' if no good match exists.
        """
        feature = self._best_matching_feature(text)
        if feature != "General":
            return feature

        if source_path:
            path = Path(source_path)
            parts = path.parts

            doc_root_idx = -1
            for i, part in enumerate(parts):
                if part.lower() in ("docs", "documentation"):
                    doc_root_idx = i
                    break
            if doc_root_idx == -1:
                doc_root_idx = 0

            skip_words = {
                "docs", "documentation", "guides", "guide", "reference",
                "tutorials", "tutorial", "how-to", "examples", "example",
                "getting-started", "introduction", "overview", "index",
                "readme", "src", "source", "assets", "images", "img",
                "common", "general", "misc", "miscellaneous",
            }

            relevant_parts = parts[doc_root_idx + 1:]
            for part in relevant_parts:
                stem = Path(part).stem
                if stem.lower() not in skip_words and not stem.startswith("_"):
                    return stem.lower()

        return "General"

    def resolve_neo4j_id(self, feature_name: str) -> str:
        """Maps a feature name to its Neo4j identifier."""
        return feature_name

    def load_and_split(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        all_chunks = []

        if settings.LOCAL_MODE:
            if not self.local_path.is_dir():
                self.prepare_local_repo()
            else:
                try:
                    self.prepare_local_repo()
                except Exception:
                    pass

            base_dir, include_files = self.discover_docs_path(str(self.local_path))

            files_to_process: list[tuple[str, str]] = []
            if include_files:
                for p in include_files:
                    files_to_process.append((p, os.path.basename(p)))
            else:
                for root, _, files in os.walk(base_dir):
                    for file in files:
                        if file.endswith((".md", ".rst", ".mdx")):
                            files_to_process.append((os.path.join(root, file), file))
        else:
            raw_files = self.fetch_via_api()
            files_to_process = [(f["path"], f["name"]) for f in raw_files]
            self._api_contents = {f["path"]: f["content"] for f in raw_files}

        if len(files_to_process) > settings.REPO_FILE_LIMIT:
            raise ValueError(
                f"Repository has {len(files_to_process)} documentation files, "
                f"which exceeds the limit of {settings.REPO_FILE_LIMIT}."
            )

        total_files = max(1, len(files_to_process))
        self._progress(
            "parsing_docs",
            0,
            total_files,
            f"Parsing documentation files (0/{total_files})",
        )

        # Phase 1: Split all files into chunks (no feature identification yet)
        chunk_sources: list[tuple[str, str]] = []  # (source_label, source_path)
        for i, (path, label) in enumerate(files_to_process, start=1):
            self._progress(
                "parsing_docs",
                i - 1,
                total_files,
                f"Parsing documentation files ({i-1}/{total_files})",
            )
            if settings.LOCAL_MODE:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = self._api_contents.get(path, "")
            chunks = splitter.split_text(content)
            if len(chunks) <= 1 and path.endswith(".rst"):
                rst_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200,
                )
                chunks = [Document(page_content=t) for t in rst_splitter.split_text(content)]
            for chunk in chunks:
                chunk_sources.append((label, path))
            all_chunks.extend(chunks)
            self._progress(
                "parsing_docs",
                i,
                total_files,
                f"Parsing documentation files ({i}/{total_files})",
            )

        # Phase 2: Batch feature identification for all chunks at once
        if all_chunks:
            texts = [chunk.page_content for chunk in all_chunks]
            batch_features = self._best_matching_features_batch(texts)
            for chunk, (label, path), feat in zip(all_chunks, chunk_sources, batch_features):
                neo4j_id = self.resolve_neo4j_id(feat)
                chunk.metadata["source"] = label
                chunk.metadata["feature_name"] = feat
                chunk.metadata["feature"] = feat
                chunk.metadata["neo4j_id"] = neo4j_id

        return all_chunks

    def upload_to_qdrant(self, chunks):
        collection_name = (
            f"docs_{self.session_id}" if self.session_id else self.vector_store.collection_name
        )
        try:
            self.vector_store.client.get_collection(collection_name)
        except Exception:
            self.vector_store.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        try:
            self.vector_store.client.create_payload_index(
                collection_name=collection_name,
                field_name="session_id",
                field_schema="keyword",
            )
        except Exception:
            logger.debug("Payload index for session_id already exists")

        unique_features = set()
        for chunk in chunks:
            feat = chunk.metadata.get("neo4j_id", "General")
            if feat and feat != "General":
                unique_features.add(feat)

        if unique_features:
            try:
                gs = GraphStore()
                existing = set(gs.get_all_feature_names(session_id=self.session_id))
                missing = unique_features - existing
                for feat_name in missing:
                    gs.ensure_feature(feat_name, session_id=self.session_id)
                if missing:
                    logger.info("Backfilled Feature nodes in Neo4j", extra={"features": list(missing)})
            except Exception:
                logger.debug("Could not backfill Feature nodes", exc_info=True)

        total = len(chunks)
        if total == 0:
            self._progress("indexing_docs", 1, 1, "No docs found to index")
            logger.info("Documentation indexed with feature tags")
            return

        batch_size = settings.DOCS_UPSERT_BATCH
        done = 0
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch = chunks[start:end]
            texts = [c.page_content for c in batch]
            vectors = self.vector_store.embeddings.embed_documents(texts)

            points = []
            for j, chunk in enumerate(batch):
                i = start + j
                points.append(
                    PointStruct(
                        id=i,
                        vector=vectors[j],
                        payload={
                            "text": chunk.page_content,
                            "source": chunk.metadata.get("source", "unknown"),
                            "feature_name": chunk.metadata.get("feature_name", "General"),
                            "feature": chunk.metadata.get("feature", "General"),
                            "neo4j_id": chunk.metadata.get(
                                "neo4j_id",
                                chunk.metadata.get("feature_name", "General"),
                            ),
                            "session_id": self.session_id,
                        },
                    )
                )

            self.vector_store.client.upsert(collection_name=collection_name, points=points)
            done = end
            self._progress(
                "indexing_docs",
                done,
                total,
                f"Indexing doc chunks ({done}/{total})",
            )
        logger.info("Documentation indexed with feature tags")


if __name__ == "__main__":
    loader = DocsLoader()
    chunks = loader.load_and_split()
    loader.upload_to_qdrant(chunks)
