import os
import uuid
from celery import Celery
from src.database.config import SessionLocal
from src.database.models import LeadModel, AuditLogModel

celery_app = Celery(
    "revops_tasks", 
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_lead_async(self, lead_id: str, payload: dict):
    """Executes the LangGraph Multi-Agent pipeline asynchronously."""
    db = SessionLocal()
    try:
        # Import inside task to prevent circular imports
        from src.agents.supervisor import build_revops_graph
        
        graph = build_revops_graph()
        
        # Add metadata for LangSmith tracing
        config = {
            "configurable": {"thread_id": lead_id},
            "tags": ["production", "celery-worker"],
            "metadata": {"lead_id": lead_id, "domain": payload.get("domain")}
        }
        
        # Execute Multi-Agent Graph
        final_state = graph.invoke(payload, config=config)
        
        # Retrieve lead record from PostgreSQL
        lead = db.query(LeadModel).filter(LeadModel.id == lead_id).first()
        if lead:
            lead.pytorch_score = final_state.get("pytorch_score")
            lead.is_qualified = final_state.get("is_qualified", False)
            lead.intent = final_state.get("intent_data", {}).get("top_intent")
            lead.status = "COMPLETED"
            
            # Write Audit Logs to Postgres
            audit_entry = AuditLogModel(
                id=str(uuid.uuid4()),
                lead_id=lead_id,
                agent_name="LangGraphEngine",
                action_taken=f"Qualified: {lead.is_qualified} | PyTorch Score: {lead.pytorch_score}"
            )
            db.add(audit_entry)
            db.commit()
            
        return {"status": "SUCCESS", "lead_id": lead_id, "is_qualified": final_state.get("is_qualified")}
        
    except Exception as exc:
        db.rollback()
        lead = db.query(LeadModel).filter(LeadModel.id == lead_id).first()
        if lead:
            lead.status = "FAILED"
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()