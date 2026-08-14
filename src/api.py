from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.agents.supervisor import build_revops_graph
from src.agents.state import RevOpsState
from src.utils.guardrails import SafetyAndEvalGuard

app = FastAPI(
    title="RevOps Multi-Agent Sales Orchestrator API",
    version="1.0.0",
    description="Enterprise API endpoint running 5-Agent State Graph pipeline for sales qualification."
)

guard = SafetyAndEvalGuard()
graph = build_revops_graph()


class LeadRequest(BaseModel):
    domain: str
    raw_features: List[float]  # [company_size, revenue, tech_match, activity_score]


class LeadResponse(BaseModel):
    domain: str
    intent: Optional[str] = None
    pytorch_score: Optional[float] = None
    is_qualified: Optional[bool] = None
    reasoning: Optional[str] = None
    rag_context: Optional[List[str]] = None
    media_prompt: Optional[str] = None


@app.post("/api/v1/qualify-lead", response_model=LeadResponse)
async def qualify_lead(request: LeadRequest):
    # 1. Guardrail Safety Check
    if guard.check_prompt_injection(request.domain):
        raise HTTPException(status_code=400, detail="Security Warning: Malicious input detected.")

    # 2. Build Initial State
    initial_state: RevOpsState = {
        "domain": request.domain,
        "raw_features": request.raw_features,
        "scraped_text": None,
        "intent_data": None,
        "pytorch_score": None,
        "icp_reasoning": None,
        "is_qualified": None,
        "rag_context": None,
        "media_asset_info": None,
        "current_step": "START",
        "audit_logs": []
    }

    # 3. Execute LangGraph Workflow
    final_state = graph.invoke(initial_state)

    # 4. Log to MLflow
    guard.log_agent_run(final_state)

    # 5. Format Response
    intent_data = final_state.get("intent_data") or {}
    media_info = final_state.get("media_asset_info") or {}

    return LeadResponse(
        domain=final_state["domain"],
        intent=intent_data.get("top_intent"),
        pytorch_score=final_state.get("pytorch_score"),
        is_qualified=final_state.get("is_qualified"),
        reasoning=final_state.get("icp_reasoning"),
        rag_context=final_state.get("rag_context"),
        media_prompt=media_info.get("diffusion_prompt")
    )
