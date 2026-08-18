import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings # High-speed local embeddings for dev
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from src.database.models import CaseStudyVectorModel

def retrieve_case_studies_postgres(db_session: Session, industry: str, query_embedding: list[float], limit: int = 2):
    """Retrieves top matching case studies using PostgreSQL pgvector cosine similarity."""
    results = (
        db_session.query(CaseStudyVectorModel)
        .filter(CaseStudyVectorModel.industry == industry)
        .order_by(CaseStudyVectorModel.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )
    
    return [doc.content for doc in results]

class RAGKnowledgeAgent:
    def __init__(self):
        # Initialize Vector Store (Module 5)
        self.embeddings = FakeEmbeddings(size=384)
        self.vector_db = None
        self._seed_knowledge_base()

    def _seed_knowledge_base(self):
        """Populates ChromaDB with product case studies and objection battlecards."""
        sample_docs = [
            Document(
                page_content="Case Study: Stripe increased enterprise pipeline by 40% using our RevOps automation tool. ROI achieved in 90 days.",
                metadata={"topic": "FinTech", "type": "case_study"}
            ),
            Document(
                page_content="Objection Battlecard: Security & Compliance. We are SOC2 Type II certified and HIPAA compliant with end-to-end encryption.",
                metadata={"topic": "Security", "type": "battlecard"}
            ),
            Document(
                page_content="Case Study: B2B SaaS platform automated 75% of SDR qualification calls, cutting customer acquisition cost (CAC) by 30%.",
                metadata={"topic": "SaaS", "type": "case_study"}
            )
        ]
        
        self.vector_db = Chroma.from_documents(
            documents=sample_docs,
            embedding=self.embeddings,
            collection_name="revops_knowledge"
        )

    def retrieve_context(self, query_text: str, top_k: int = 2) -> List[str]:
        """Performs hybrid/semantic search in ChromaDB (Module 5)"""
        if not self.vector_db:
            return ["No knowledge base available."]

        results = self.vector_db.similarity_search(query_text, k=top_k)
        return [doc.page_content for doc in results]