from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.supervisor import run_qualification

app = FastAPI(title="RevOps Multi-Agent Qualification API")


class QualifyLeadRequest(BaseModel):
    domain: str


@app.post("/api/v1/qualify-lead")
def qualify_lead(req: QualifyLeadRequest):
    state = run_qualification(req.domain)
    if state.get("blocked"):
        raise HTTPException(status_code=400, detail=state.get("block_reason", "blocked input"))
    return {
        "domain": state["domain"],
        "intent": (state.get("intent_data") or {}).get("top_intent"),
        "pytorch_score": state.get("pytorch_score"),
        "is_qualified": state.get("is_qualified"),
        "icp_reasoning": state.get("icp_reasoning"),
        "rag_context": state.get("rag_context"),
        "media_asset_info": state.get("media_asset_info"),
        "audit_logs": state.get("audit_logs"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}