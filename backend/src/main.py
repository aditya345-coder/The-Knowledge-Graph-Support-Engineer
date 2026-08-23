import asyncio
import hashlib
import hmac
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import Depends, FastAPI, HTTPException, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Callable, cast

from urllib.parse import urlparse

from database.redis_store import get_redis_store
from middleware.rate_limit import check_rate_limit, decrement_rate_limit
from middleware.feedback import FeedbackStore
from middleware.auth import get_current_user
from utils.logging_config import setup_logging
from utils.validators import validate_session_id
from settings import settings

logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: resume interrupted sessions + start periodic cleanup. Shutdown: nothing to do."""
    running = await redis_store.get_running_sessions()
    for session in running:
        sid = session.get("session_id", "")
        logger.info("Resuming interrupted session", extra={"session_id": sid})
        repo_url = session.get("repo_url", "")
        github_token = session.get("github_token") or settings.GITHUB_TOKEN
        completed = session.get("completed_phases", [])
        user_sub = session.get("user_sub", "anonymous")
        # Run in thread pool since _prepare_repo_task is sync
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _prepare_repo_task, repo_url, github_token, sid, completed, user_sub)

    # Start periodic orphan cleanup (every hour)
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(3600)
            try:
                await _cleanup_orphaned_sessions()
            except Exception as e:
                logger.warning("Periodic cleanup failed: %s", e)

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()


app = FastAPI(title="Hybrid Support Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

_feedback_store = FeedbackStore()

# Redis-backed session state store (replaces in-memory _SESSION_STATUS)
redis_store = get_redis_store()

# Per-session locks to prevent concurrent resume tasks
_session_locks: dict[str, threading.Lock] = {}
_session_locks_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a lock for a specific session."""
    with _session_locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _release_session_lock(session_id: str) -> None:
    """Release a session lock when done."""
    with _session_locks_lock:
        _session_locks.pop(session_id, None)

_agent: Any = None


def _get_agent() -> Any:
    """Lazy-initialize SupportAgent on first use."""
    global _agent
    if _agent is None:
        from agents.support_agent import SupportAgent
        logger.info("Initializing SupportAgent...")
        try:
            _agent = SupportAgent()
            logger.info("SupportAgent initialized successfully.")
        except Exception as e:
            logger.warning("SupportAgent init failed (external services unavailable): %s", e)
            raise
    return _agent

# Startup security check
env = os.getenv("ENV", "development").lower()
if env == "production" and not settings.AUTH_ENABLED:
    logger.critical(
        "AUTH_ENABLED is False in production — this is a security risk"
    )


class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=2000)
    # Optional session isolation key; can also be provided via `X-Session-Id` header.
    session_id: str | None = Field(None, pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    # Optional flags/plumbing for future roadmap items.
    allow_web_search: bool = False
    repo_url: str | None = Field(None, max_length=500)
    github_token: str | None = Field(None, max_length=200)


class PrepareRepoRequest(BaseModel):
    repo_url: str = Field(..., min_length=1, max_length=500)
    github_token: str | None = Field(None, max_length=200)
    session_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,128}$")


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)
    feature_detected: str = Field("General", max_length=100)
    thumbs_up: bool
    session_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,128}$")


class ResumeRepoRequest(BaseModel):
    session_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,128}$")


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    raw_body = await request.body()
    
    if not settings.WEBHOOK_SECRET:
        env = os.getenv("ENV", "development").lower()
        if env == "production":
            logger.critical("WEBHOOK_SECRET not configured in production - refusing webhook")
            return JSONResponse(status_code=503, content={"detail": "Webhook authentication not configured"})
        elif os.getenv("WEBHOOK_AUTH_DISABLED", "false").lower() != "true":
            logger.warning("WEBHOOK_SECRET not configured - webhook authentication disabled")
            return JSONResponse(status_code=503, content={"detail": "Webhook authentication not configured"})
        else:
            # Only allow bypass in development
            if env == "production":
                logger.critical("WEBHOOK_AUTH_DISABLED=true in production - refusing webhook (bypass not allowed)")
                return JSONResponse(status_code=503, content={"detail": "Webhook authentication not configured"})
            logger.warning("WEBHOOK_SECRET not configured - authentication explicitly disabled via WEBHOOK_AUTH_DISABLED")
    else:
        signature_header = request.headers.get("X-Hub-Signature-256", "")
        if not signature_header:
            logger.warning("Webhook missing signature header")
            raise HTTPException(status_code=403, detail="Missing signature")
        
        expected_signature = "sha256=" + hmac.new(
            settings.WEBHOOK_SECRET.encode(),
            raw_body,
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, signature_header):
            logger.warning("Webhook signature verification failed")
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    event = request.headers.get("X-GitHub-Event", "")
    session_id = settings.WEBHOOK_SESSION_ID or ""

    logger.info("Webhook received", extra={"event": event, "session_id": session_id})

    from middleware.webhook_handler import handle_push, handle_issue_event, handle_pr_event

    if event == "push":
        background_tasks.add_task(handle_push, payload, session_id)
    elif event == "issues":
        background_tasks.add_task(handle_issue_event, payload, session_id)
    elif event == "pull_request":
        background_tasks.add_task(handle_pr_event, payload, session_id)
    else:
        logger.info("Unhandled webhook event", extra={"event": event})

    return {"status": "ok"}


@app.post("/v1/solve-ticket", dependencies=[Depends(get_current_user)])
async def solve_ticket(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    try:
        current_agent = _get_agent()
    except Exception:
        raise HTTPException(status_code=503, detail="Agent not initialized — external services unavailable")
    try:
        # Running our LangGraph State Machine
        logger.info("Solving ticket", extra={"query": request.user_query})
        session_id = x_session_id or request.session_id
        if session_id:
            session_id = validate_session_id(session_id)
            if not session_id:
                logger.warning("Invalid session_id format in header", extra={"session_id": x_session_id[:50] if x_session_id else None})
                raise HTTPException(status_code=400, detail="Invalid session_id format")
        repo_name = request.repo_url or settings.TARGET_REPO or "this repository"

        # Enforce rate limit using authenticated user identity
        user_id = user.get("sub", "anonymous")
        if not check_rate_limit(user_id, max_queries=settings.RATE_LIMIT_MAX):
            return JSONResponse(
                status_code=429,
                content={"detail": f"Daily query limit of {settings.RATE_LIMIT_MAX} reached. Try again tomorrow."},
            )

        # If a session_id is provided, require that the repo was prepared (docs collection exists).
        if session_id:
            try:
                ready = current_agent.retriever.vector_store.has_session_collection(session_id)
            except Exception:
                ready = False
            if not ready:
                return {
                    "status": "needs_ingestion",
                    "message": "Repository is not prepared for this session_id. Call /v1/prepare-repo first.",
                    "metadata": {"session_id": session_id},
                }
        initial_state: dict[str, Any] = {
            "query": request.user_query,
            "original_query": request.user_query,
            "rewritten_query": "",
            "repo_name": repo_name,
            "session_id": session_id or "",
            "detected_feature": "",
            "documents": [],
            "code_results": [],
            "github_issues": [],
            "web_results": [],
            "response": "",
            "is_relevant": True,
            "is_hallucination": False,
            "iteration": 0,
            "allow_web_search": bool(request.allow_web_search),
        }
        config = {"configurable": {"thread_id": session_id or "default"}}
        result = cast(Any, current_agent.app).invoke(initial_state, config=config)

        current_agent.prune_history(session_id or "default")

        return {
            "status": "success",
            "answer": result["response"],
            "metadata": {
                "detected_feature": result["detected_feature"],
                "docs_retrieved": len(result["documents"]),
                "github_issues_found": len(result["github_issues"]),
                "session_id": session_id,
                "is_relevant": bool(result.get("is_relevant", True)),
            },
        }
    except Exception as e:
        # Decrement rate limit counter on failure so quota isn't consumed
        try:
            decrement_rate_limit(user_id)
        except Exception:
            logger.warning("Failed to decrement rate limit on error", extra={"user_id": user_id})
        logger.exception("Error while solving ticket")
        raise HTTPException(status_code=500, detail=str(e))


def _update_session(
    session_id: str,
    stage: str,
    message: str = "",
    *,
    percent: int | None = None,
    current: int | None = None,
    total: int | None = None,
    eta_seconds: int | None = None,
    created: bool = False,
) -> None:
    """Update session state in Redis. Use created=True for initial creation."""
    fields: dict[str, Any] = {"stage": stage, "message": message}
    if percent is not None:
        fields["percent"] = str(int(max(0, min(100, percent))))
    if current is not None:
        fields["current"] = str(int(current))
    if total is not None:
        fields["total"] = str(int(total))
    if eta_seconds is not None:
        fields["eta_seconds"] = str(int(max(0, eta_seconds)))
    if created:
        fields["created_at"] = datetime.now().isoformat()
        fields["completed_phases"] = "[]"
    redis_store.save_session_sync(session_id, fields)


def _index_code_chunks(
    chunks: list,
    session_id: str,
    report: Callable,
    vector_store: Any = None,
) -> None:
    from database.vector_store import VectorStore
    from qdrant_client.models import Distance, VectorParams, PointStruct

    if vector_store is None:
        vector_store = VectorStore()
    collection_name = f"code_{session_id}" if session_id else "code_default"

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
        logger.debug("Payload index for session_id already exists")

    total = len(chunks)
    batch_size = 64

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = chunks[start:end]

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
        report("indexing_code", end, total, f"Indexing code ({end}/{total})")


def _prepare_repo_task(
    repo_url: str, github_token: str | None, session_id: str,
    completed_phases: list[str] | None = None,
    user_sub: str = "anonymous",
) -> None:
    """Prepare repository for a session. Supports resume via completed_phases."""
    # Validate session_id to prevent path traversal
    validated_session_id = validate_session_id(session_id)
    if not validated_session_id:
        logger.error(
            "Invalid session_id in _prepare_repo_task",
            extra={"session_id": session_id[:50]},
        )
        _update_session(session_id, "error", "Invalid session_id format", percent=100)
        return
    session_id = validated_session_id

    # Acquire session lock to prevent concurrent resume
    lock = _get_session_lock(session_id)
    if not lock.acquire(blocking=False):
        logger.warning("Session already being processed", extra={"session_id": session_id})
        _update_session(session_id, "error", "Session already being processed", percent=100)
        return

    try:
        from ingestion.docs_loader import DocsLoader
        from ingestion.github_loader import GitHubGraphLoader

        # Get or initialize completed phases list
        if completed_phases is None:
            completed_phases = []

        # Overall progress weights — phases 3/4/5 run in parallel after phase 2.
        PHASES = {
            "cloning": 5,
            "parsing_docs": 20,
            "indexing_docs": 35,
            "indexing_code": 10,
            "building_graph": 30,
        }
        phase_order = list(PHASES.keys())
        phase_base: dict[str, int] = {}
        acc = 0
        for p in phase_order:
            phase_base[p] = acc
            acc += PHASES[p]

        phase_started_at: dict[str, float] = {}

        def report(phase: str, current: int, total: int, message: str) -> None:
            w = PHASES.get(phase, 0)
            base = phase_base.get(phase, 0)
            if phase not in phase_started_at:
                phase_started_at[phase] = time.time()
            pct_in_phase = 0
            if total > 0:
                pct_in_phase = int((current / total) * 100)
            overall = base + int((pct_in_phase / 100) * w)

            eta = None
            started = phase_started_at.get(phase)
            if started and total > 0 and current > 0:
                elapsed = time.time() - started
                rate = current / max(0.001, elapsed)
                remaining = max(0, total - current)
                eta = int(remaining / max(0.001, rate))

            _update_session(
                session_id,
                phase,
                message,
                percent=overall,
                current=current,
                total=total,
                eta_seconds=eta,
            )

        # ── Phase 1: Clone (skip if completed) ──────────────────────
        if "cloning" not in completed_phases:
            report("cloning", 0, 1, f"Preparing local repo for {repo_url}")
            local_path = str(settings.RAW_DOCS_DIR / session_id)

            docs_loader = DocsLoader(
                repo_url=repo_url,
                github_token=github_token,
                local_path=local_path,
                session_id=session_id,
                progress_cb=report,
            )
            report("cloning", 1, 1, "Repository available locally")
            completed_phases.append("cloning")
            redis_store.mark_phase_complete_sync(session_id, "cloning")
        else:
            # Re-initialize docs_loader for subsequent phases
            local_path = str(settings.RAW_DOCS_DIR / session_id)
            docs_loader = DocsLoader(
                repo_url=repo_url,
                github_token=github_token,
                local_path=local_path,
                session_id=session_id,
                progress_cb=report,
            )

        # ── Phase 2: Parse docs (skip if completed) ─────────────────
        if "parsing_docs" not in completed_phases:
            report("parsing_docs", 0, 1, "Loading and splitting docs")
            chunks = docs_loader.load_and_split()
            completed_phases.append("parsing_docs")
            redis_store.mark_phase_complete_sync(session_id, "parsing_docs")
        else:
            chunks = []

        # ── Phases 3/4/5: Index docs, index code, build graph (parallel) ──

        def _run_index_docs():
            if "indexing_docs" in completed_phases:
                return
            report("indexing_docs", 0, max(1, len(chunks)), "Indexing docs into Qdrant")
            docs_loader.upload_to_qdrant(chunks)
            completed_phases.append("indexing_docs")
            redis_store.mark_phase_complete_sync(session_id, "indexing_docs")

        def _run_index_code():
            if "indexing_code" in completed_phases:
                return
            if not (settings.AST_ENABLED and settings.LOCAL_MODE):
                return
            from ingestion.code_indexer import CodeIndexer

            report("indexing_code", 0, 1, "Indexing Python source code")
            code_indexer = CodeIndexer()

            repo_root = Path(str(settings.RAW_DOCS_DIR / session_id))
            src_root = repo_root
            for candidate in [repo_root / "src", repo_root / "lib", repo_root]:
                if candidate.is_dir():
                    src_root = candidate
                    break

            code_chunks = code_indexer.index_directory(src_root, exclude_dirs=settings.AST_EXCLUDE_DIRS)
            report("indexing_code", 0, max(1, len(code_chunks)), f"Found {len(code_chunks)} code symbols")

            if code_chunks:
                _index_code_chunks(
                    code_chunks, session_id, report,
                    vector_store=_get_agent().retriever.vector_store,
                )
            completed_phases.append("indexing_code")
            redis_store.mark_phase_complete_sync(session_id, "indexing_code")

        def _run_build_graph():
            if "building_graph" in completed_phases:
                return
            report("building_graph", 0, 20, "Building Neo4j issue graph")
            gh_loader = GitHubGraphLoader(
                repo_url=repo_url,
                token=github_token,
                session_id=session_id,
                progress_cb=report,
            )
            gh_loader.run()
            completed_phases.append("building_graph")
            redis_store.mark_phase_complete_sync(session_id, "building_graph")

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_run_index_docs): "indexing_docs",
                pool.submit(_run_index_code): "indexing_code",
                pool.submit(_run_build_graph): "building_graph",
            }
            for future in as_completed(futures):
                phase_name = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(f"Parallel phase {phase_name} failed")
                    raise

        _update_session(session_id, "complete", "Repository prepared", percent=100)

        # Auto-save to user's repo list so it survives browser switches
        try:
            parsed = urlparse(repo_url)
            repo_path = parsed.path.strip("/")
            repo_name = repo_path.removesuffix(".git")
            entry = {
                "sessionId": session_id,
                "repoUrl": repo_url,
                "name": repo_name,
                "preparedAt": datetime.utcnow().isoformat(),
            }
            current_list = redis_store.get_repo_list_sync(user_sub) or []
            existing = [r for r in current_list if r.get("sessionId") != session_id]
            updated = existing + [entry]
            redis_store.save_repo_list_sync(user_sub, updated)
        except Exception as e:
            logger.warning("Failed to auto-save repo list: %s", e)
    except Exception as e:
        logger.exception("Repo preparation failed", extra={"session_id": session_id, "error_type": type(e).__name__})
        error_msg = str(e) or "Repository preparation failed"
        if len(error_msg) > 300:
            error_msg = error_msg[:297] + "..."
        _update_session(session_id, "error", error_msg, percent=100)
    finally:
        _release_session_lock(session_id)


@app.post("/v1/prepare-repo", dependencies=[Depends(get_current_user)])
async def prepare_repo(
    request: PrepareRepoRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    session_id = x_session_id or request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session_id = validate_session_id(session_id)
    if not session_id:
        logger.warning("Invalid session_id format in header", extra={"session_id": x_session_id[:50] if x_session_id else None})
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    user_sub = user.get("sub", "anonymous")

    # Check for existing session
    existing = await redis_store.get_session(session_id)
    if existing and existing.get("stage") == "running":
        return {
            "status": "interrupted",
            "session_id": session_id,
            "completed_phases": existing.get("completed_phases", []),
            "message": "Previous session was interrupted. Use /v1/resume-repo to continue.",
        }

    # Start fresh
    await redis_store.save_session(session_id, {
        "repo_url": request.repo_url,
        "stage": "running",
        "completed_phases": [],
        "created_at": datetime.now().isoformat(),
        "user_sub": user_sub,
    })
    background_tasks.add_task(_prepare_repo_task, request.repo_url, request.github_token, session_id, None, user_sub)
    return {"status": "processing", "metadata": {"session_id": session_id}}


@app.get("/v1/status/{session_id}", dependencies=[Depends(get_current_user)])
async def get_status(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    session_id = validate_session_id(session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    # Check Redis first
    session = await redis_store.get_session(session_id)
    if session:
        return {
            "status": "ok",
            "data": {
                "session_id": session_id,
                "stage": session.get("stage", "unknown"),
                "message": session.get("message", ""),
                "percent": int(session.get("percent", 0)) if session.get("percent") else 0,
                "completed_phases": session.get("completed_phases", []),
                "eta_seconds": int(session.get("eta_seconds", 0)) if session.get("eta_seconds") else None,
                "repo_url": session.get("repo_url", ""),
            },
        }

    return {"status": "needs_ingestion", "metadata": {"session_id": session_id}}


@app.post("/v1/cleanup/{session_id}", dependencies=[Depends(get_current_user)])
async def cleanup_session(session_id: str, user: dict = Depends(get_current_user)):
    session_id = validate_session_id(session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    # Rate limit cleanup operations (configurable, default 5 per day per user)
    user_id = user.get("sub", "anonymous")
    if not await redis_store.check_rate_limit_namespace(user_id, "cleanup", settings.CLEANUP_RATE_LIMIT):
        logger.warning("Cleanup rate limit exceeded", extra={"user_id": user_id})
        return JSONResponse(
            status_code=429,
            content={"detail": f"Daily cleanup limit of {settings.CLEANUP_RATE_LIMIT} reached. Try again tomorrow."},
        )

    try:
        await _cleanup_session_data(session_id)
        return {"status": "success"}
    except Exception as e:
        logger.exception("Cleanup failed", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail=str(e))


async def _cleanup_session_data(session_id: str) -> None:
    """Delete all session-scoped data: Redis, Qdrant, Neo4j, disk."""
    # 1. Delete Redis session state
    await redis_store.delete_session(session_id)

    # 2. Delete Qdrant collections
    try:
        current_agent = _get_agent()
        current_agent.retriever.vector_store.cleanup_session(session_id)
    except Exception as e:
        logger.warning("Qdrant cleanup failed: %s", e)

    # 3. Delete Neo4j session nodes
    try:
        current_agent = _get_agent()
        current_agent.retriever.graph_store.cleanup_session(session_id)
    except Exception as e:
        logger.warning("Neo4j cleanup failed: %s", e)

    # 4. Delete cloned repo from disk
    repo_path = settings.settings.RAW_DOCS_DIR / session_id
    if repo_path.exists():
        shutil.rmtree(repo_path, ignore_errors=True)


@app.post("/v1/resume-repo", dependencies=[Depends(get_current_user)])
async def resume_repo(
    request: ResumeRepoRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    session_id = x_session_id or request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session_id = validate_session_id(session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await redis_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="No session found. Start a new preparation.")

    if session.get("stage") == "complete":
        return {"status": "already_complete", "session_id": session_id}

    if session.get("stage") != "running":
        raise HTTPException(status_code=400, detail=f"Session is in '{session.get('stage')}' state. Cannot resume.")

    # Resume in background
    repo_url = session.get("repo_url", "")
    github_token = session.get("github_token") or settings.GITHUB_TOKEN
    completed = session.get("completed_phases", [])
    user_sub = user.get("sub", "anonymous")
    background_tasks.add_task(_prepare_repo_task, repo_url, github_token, session_id, completed, user_sub)
    return {"status": "resuming", "completed_phases": completed}


@app.post("/v1/fresh-repo", dependencies=[Depends(get_current_user)])
async def fresh_repo(
    request: PrepareRepoRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    session_id = x_session_id or request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session_id = validate_session_id(session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    user_sub = user.get("sub", "anonymous")

    # Clean up old session data
    await _cleanup_session_data(session_id)

    # Start fresh
    await redis_store.save_session(session_id, {
        "repo_url": request.repo_url,
        "stage": "running",
        "completed_phases": [],
        "created_at": datetime.now().isoformat(),
        "user_sub": user_sub,
    })
    background_tasks.add_task(_prepare_repo_task, request.repo_url, request.github_token, session_id, None, user_sub)
    return {"status": "processing", "metadata": {"session_id": session_id}}


@app.post("/v1/feedback", dependencies=[Depends(get_current_user)])
async def submit_feedback(request: FeedbackRequest):
    try:
        _feedback_store.store_feedback(
            query=request.query,
            answer=request.answer,
            feature=request.feature_detected,
            thumbs_up=request.thumbs_up,
            session_id=request.session_id,
        )
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Feedback storage failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Repo List Persistence ────────────────────────────────────────────

@app.get("/v1/repo-list", dependencies=[Depends(get_current_user)])
async def get_repo_list(user: dict = Depends(get_current_user)):
    user_id = user.get("sub", "anonymous")
    repos = await redis_store.get_repo_list(user_id)
    return {"repos": repos}


@app.post("/v1/repo-list", dependencies=[Depends(get_current_user)])
async def save_repo_list(request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    user_id = user.get("sub", "anonymous")
    await redis_store.save_repo_list(user_id, body.get("repos", []))
    return {"status": "ok"}


# ── Orphan Cleanup (Phase 6) ────────────────────────────────────────

async def _cleanup_orphaned_sessions() -> int:
    """Find and clean up Qdrant/Neo4j/disk data for sessions no longer in Redis.
    Returns the number of orphaned sessions cleaned."""
    cleaned = 0
    try:
        current_agent = _get_agent()
        # Scan Qdrant collections for docs_{sid} and code_{sid} patterns
        collections = current_agent.retriever.vector_store.client.get_collections().collections
        for col in collections:
            name = col.name
            sid = None
            if name.startswith("docs_"):
                sid = name[5:]
            elif name.startswith("code_"):
                sid = name[6:]
            if sid and not await redis_store.get_session(sid):
                logger.info("Cleaning orphaned Qdrant collection", extra={"collection": name, "session_id": sid})
                try:
                    current_agent.retriever.vector_store.client.delete_collection(name)
                except Exception as e:
                    logger.warning("Failed to delete orphaned Qdrant collection %s: %s", name, e)
                # Also clean Neo4j and disk for this session
                try:
                    current_agent.retriever.graph_store.cleanup_session(sid)
                except Exception as e:
                    logger.warning("Failed to clean orphaned Neo4j data for %s: %s", sid, e)
                repo_path = settings.settings.RAW_DOCS_DIR / sid
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                cleaned += 1
    except Exception as e:
        logger.warning("Orphan cleanup scan failed: %s", e)
    return cleaned


@app.post("/v1/admin/cleanup-orphans", dependencies=[Depends(get_current_user)])
async def cleanup_orphans(user: dict = Depends(get_current_user)):
    """Manually trigger orphaned session data cleanup."""
    cleaned = await _cleanup_orphaned_sessions()
    return {"status": "ok", "cleaned": cleaned}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
