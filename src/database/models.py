import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.database.config import Base

class LeadModel(Base):
    """Relational table storing lead records, scores, and status."""
    __tablename__ = "leads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    domain = Column(String, index=True, nullable=False)
    intent = Column(String, nullable=True)
    pytorch_score = Column(Float, nullable=True)
    is_qualified = Column(Boolean, default=False)
    status = Column(String, default="PENDING", index=True)  # PENDING, PROCESSING, COMPLETED, FAILED
    raw_features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    audit_logs = relationship("AuditLogModel", back_populates="lead", cascade="all, delete-orphan")

class AuditLogModel(Base):
    """Audit table logging every agent execution step for observability."""
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id = Column(String, ForeignKey("leads.id"), nullable=False)
    agent_name = Column(String, nullable=False)
    action_taken = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    lead = relationship("LeadModel", back_populates="audit_logs")

class CaseStudyVectorModel(Base):
    """PostgreSQL pgvector table storing case studies and embeddings for RAG."""
    __tablename__ = "case_studies"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    industry = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # Dimensions for embeddings
    created_at = Column(DateTime, default=datetime.utcnow)