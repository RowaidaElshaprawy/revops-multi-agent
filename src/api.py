import uuid
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.database.config import get_db_session, init_db
from src.database.models import LeadModel, AuditLogModel
from src.config.tracing import setup_langsmith_tracing
from src.workers.tasks import process_lead_async

# Initialize LangSmith Tracing
setup_langsmith_tracing()

app = FastAPI(
    title="RevOps Enterprise Multi-Agent Engine",
    version="2.0.0",
    description="Production-grade asynchronous lead qualification microservice with PostgreSQL, Redis, Celery, and LangSmith."
)

@app.on_event("startup")
def startup_event():
    """Ensure database tables and vector extensions are initialized on boot."""
    try:
        init_db()
    except Exception as e:
        print(f"Database initialization warning: {e}")

class LeadPayload(BaseModel):
    domain: str = Field(..., example="stripe.com")
    raw_features: Dict[str, Any] = Field(
        default_factory=lambda: {
            "company_size": 250,
            "revenue_tier": 3,
            "tech_match": 1,
            "engagement": 8
        }
    )

@app.post("/api/v1/qualify-lead-async", status_code=202)
async def qualify_lead_async_endpoint(payload: LeadPayload, db: Session = Depends(get_db_session)):
    """Production Endpoint: Instantly returns HTTP 202 and delegates processing to Celery background task."""
    lead_id = str(uuid.uuid4())
    
    # 1. Immediate Persistence into PostgreSQL
    db_lead = LeadModel(
        id=lead_id,
        domain=payload.domain,
        raw_features=payload.raw_features,
        status="PROCESSING"
    )
    db.add(db_lead)
    db.commit()

    # 2. Push to Redis Task Queue for Asynchronous Processing
    process_lead_async.delay(lead_id, {"domain": payload.domain, "raw_features": payload.raw_features, "lead_id": lead_id})

    # 3. Return Instant HTTP 202 Accepted Response
    return {
        "status": "ACCEPTED",
        "message": "Lead payload queued for asynchronous processing.",
        "lead_id": lead_id,
        "check_status_url": f"/api/v1/lead-status/{lead_id}"
    }

@app.get("/api/v1/lead-status/{lead_id}")
async def get_lead_status(lead_id: str, db: Session = Depends(get_db_session)):
    """Fetch lead status and final evaluation results from PostgreSQL."""
    lead = db.query(LeadModel).filter(LeadModel.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead ID not found.")
        
    logs = db.query(AuditLogModel).filter(AuditLogModel.lead_id == lead_id).all()
    
    return {
        "lead_id": lead.id,
        "domain": lead.domain,
        "status": lead.status,
        "pytorch_score": lead.pytorch_score,
        "is_qualified": lead.is_qualified,
        "intent": lead.intent,
        "audit_logs": [log.action_taken for log in logs]
    }