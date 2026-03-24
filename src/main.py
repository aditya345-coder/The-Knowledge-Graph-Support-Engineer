from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.support_agent import SupportAgent

app = FastAPI(title="Omni-Support GraphRAG API")
agent = SupportAgent()

class QueryRequest(BaseModel):
    user_query: str

@app.post("/v1/solve-ticket")
async def solve_ticket(request: QueryRequest):
    try:
        # Running our LangGraph State Machine
        result = agent.app.invoke({"query": request.user_query})
        
        return {
            "status": "success",
            "answer": result["response"],
            "metadata": {
                "detected_feature": result["detected_feature"],
                "docs_retrieved": len(result["documents"]),
                "github_issues_found": len(result["github_issues"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)