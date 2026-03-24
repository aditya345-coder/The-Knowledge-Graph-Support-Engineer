from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, cast

from agents.support_agent import SupportAgent
from utils.logging_config import setup_logging

logger = setup_logging(__name__)
app = FastAPI(title="Omni-Support GraphRAG API")

logger.info("Initializing SupportAgent...")
agent: Any = SupportAgent()
logger.info("SupportAgent initialized successfully.")


class QueryRequest(BaseModel):
    user_query: str


@app.post("/v1/solve-ticket")
async def solve_ticket(request: QueryRequest):
    try:
        # Running our LangGraph State Machine
        logger.info("Solving ticket", extra={"query": request.user_query})
        initial_state = {
            "query": request.user_query,
            "detected_feature": "",
            "documents": [],
            "github_issues": [],
            "response": "",
            "is_hallucination": False,
            "iteration": 0,
        }
        result = cast(Any, agent.app).invoke(initial_state)

        return {
            "status": "success",
            "answer": result["response"],
            "metadata": {
                "detected_feature": result["detected_feature"],
                "docs_retrieved": len(result["documents"]),
                "github_issues_found": len(result["github_issues"]),
            },
        }
    except Exception as e:
        logger.exception("Error while solving ticket")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)
