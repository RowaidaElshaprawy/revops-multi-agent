import os
from typing import List

import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_store")
COLLECTION_NAME = "revops_battlecards"

SEED_DOCS = [
    {"id": "case-1", "text": "Case study: 50-person SaaS company cut lead qualification time "
                               "from 3 days to 20 minutes after automating inbound scoring."},
    {"id": "battlecard-pricing", "text": "Objection: pricing seems high. Response: point to "
                                          "usage-based tiers and offer a scoped pilot."},
    {"id": "battlecard-enterprise", "text": "Objection: need SOC 2 and SSO before trial. "
                                             "Response: share compliance one-pager, offer sandbox SSO demo."},
    {"id": "case-2", "text": "Case study: enterprise buyer with strong SSO/compliance signals "
                              "converted 3x faster when outreach led with security posture."},
]


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        collection.add(ids=[d["id"] for d in SEED_DOCS], documents=[d["text"] for d in SEED_DOCS])
    return collection


def retrieve(query_text: str, k: int = 2) -> List[str]:
    if not query_text.strip():
        return []
    collection = _get_collection()
    results = collection.query(query_texts=[query_text], n_results=min(k, collection.count()))
    docs = results.get("documents", [[]])
    return docs[0] if docs else []


def rag_node(state):
    scraped_text = state.get("scraped_text") or state["domain"]
    context = retrieve(scraped_text, k=2)
    logs = state.get("audit_logs", [])
    logs.append(f"[rag_agent] retrieved {len(context)} context docs from ChromaDB")
    state["rag_context"] = context
    state["current_step"] = "RAG_RETRIEVED"
    state["audit_logs"] = logs
    return state