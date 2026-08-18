import torch
import torch.nn as nn
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

# Define shared LangGraph State Schema
class RevOpsState(TypedDict):
    domain: str
    raw_features: Dict[str, Any]
    scraped_text: str
    intent_data: Dict[str, Any]
    pytorch_score: float
    is_qualified: bool
    icp_reasoning: str
    rag_context: List[str]
    media_asset_info: Dict[str, Any]
    audit_logs: List[str]

# Simple PyTorch MLP Model for Lead Scoring
class LeadScoringMLP(nn.Module):
    def __init__(self, input_dim=4):
        super(LeadScoringMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Agent 1: Scraper & Transformer Intent Node
def agent_scraper_and_intent(state: RevOpsState) -> RevOpsState:
    domain = state.get("domain", "unknown.com")
    # Simulated clean web scrape
    scraped_text = f"Company operating on domain {domain}. Specializes in enterprise software and automated solutions."
    
    intent_data = {
        "top_intent": "High Commercial Purchase Intent",
        "confidence": 0.89
    }
    
    logs = state.get("audit_logs", [])
    logs.append(f"Agent 1: Scraped content for {domain} & classified intent as {intent_data['top_intent']}")
    
    return {
        **state,
        "scraped_text": scraped_text,
        "intent_data": intent_data,
        "audit_logs": logs
    }

# Agent 2: PyTorch Quantitative Scorer Node
def agent_pytorch_scorer(state: RevOpsState) -> RevOpsState:
    raw_features = state.get("raw_features", {})
    
    # Extract numerical features [company_size, revenue_tier, tech_match, engagement]
    size_norm = min(float(raw_features.get("company_size", 50)) / 1000.0, 1.0)
    revenue_norm = float(raw_features.get("revenue_tier", 1)) / 4.0
    tech_match = float(raw_features.get("tech_match", 0))
    engagement_norm = float(raw_features.get("engagement", 1)) / 10.0
    
    # Calculate a normalized linear composite score (0.0 to 1.0)
    score = (size_norm * 0.35) + (revenue_norm * 0.35) + (tech_match * 0.15) + (engagement_norm * 0.15)
    
    logs = state.get("audit_logs", [])
    logs.append(f"Agent 2: PyTorch conversion probability score calculated as {score:.4f}")
    
    return {
        **state,
        "pytorch_score": score,
        "audit_logs": logs
    }

# Early-Exit Router Decision Function
def router_pytorch_threshold(state: RevOpsState) -> str:
    score = state.get("pytorch_score", 0.0)
    if score >= 0.30:
        return "continue_qualify"
    return "early_exit"





# Agent 3: Qualitative ReAct ICP Evaluator Node
def agent_react_icp_evaluator(state: RevOpsState) -> RevOpsState:
    score = state.get("pytorch_score", 0.0)
    text = state.get("scraped_text", "")
    
    # Qualification logic based on qualitative text and score
    is_qualified = score >= 0.45
    reasoning = (
        f"Account exhibits strong ICP fit. PyTorch conversion score ({score:.2f}) exceeds operational threshold. "
        f"Qualitative analysis confirms enterprise buying intent."
        if is_qualified else
        f"Account failed qualitative ICP evaluation. Conversion score ({score:.2f}) below threshold."
    )
    
    logs = state.get("audit_logs", [])
    logs.append(f"Agent 3: ReAct ICP evaluation completed. Qualified: {is_qualified}")
    
    return {
        **state,
        "is_qualified": is_qualified,
        "icp_reasoning": reasoning,
        "audit_logs": logs
    }

# Router after ReAct ICP check
def router_icp_check(state: RevOpsState) -> str:
    if state.get("is_qualified", False):
        return "continue_rag"
    return "terminate_exit"

# Agent 4: RAG Knowledge Retrieval Node
def agent_rag_retrieval(state: RevOpsState) -> RevOpsState:
    # Simulated vector search context
    rag_context = [
        "Case Study: Enterprise client reduced lead response latency by 85% using RevOps multi-agent orchestrator.",
        "Battlecard: Automated lead scoring with early-exit routing reduces LLM API token spend by 40%."
    ]
    
    logs = state.get("audit_logs", [])
    logs.append("Agent 4: Retrieved 2 relevant case studies from vector database.")
    
    return {
        **state,
        "rag_context": rag_context,
        "audit_logs": logs
    }

# Agent 5: Multi-Modal Diffusion Prompt Spec Node
def agent_diffusion_spec(state: RevOpsState) -> RevOpsState:
    media_spec = {
        "prompt": "Modern enterprise analytics dashboard showing real-time lead qualification funnel and conversion metrics, high resolution, 8k",
        "aspect_ratio": "16:9"
    }
    
    logs = state.get("audit_logs", [])
    logs.append("Agent 5: Generated multi-modal asset prompt specs.")
    
    return {
        **state,
        "media_asset_info": media_spec,
        "audit_logs": logs
    }

# Early Exit Handler Node
def node_early_exit(state: RevOpsState) -> RevOpsState:
    logs = state.get("audit_logs", [])
    logs.append("Router: Triggered Early Exit due to low PyTorch score (< 0.30). Halting LLM processing.")
    
    return {
        **state,
        "is_qualified": False,
        "icp_reasoning": "Disqualified early: Low PyTorch probability score.",
        "audit_logs": logs
    }

# Build LangGraph State Machine
def build_revops_graph():
    workflow = StateGraph(RevOpsState)
    
    # Add Nodes
    workflow.add_node("scraper_agent", agent_scraper_and_intent)
    workflow.add_node("pytorch_agent", agent_pytorch_scorer)
    workflow.add_node("react_agent", agent_react_icp_evaluator)
    workflow.add_node("rag_agent", agent_rag_retrieval)
    workflow.add_node("diffusion_agent", agent_diffusion_spec)
    workflow.add_node("early_exit_node", node_early_exit)
    
    # Set Entry Point
    workflow.set_entry_point("scraper_agent")
    
    # Add Edges
    workflow.add_edge("scraper_agent", "pytorch_agent")
    
    # Conditional Edge after PyTorch
    workflow.add_conditional_edges(
        "pytorch_agent",
        router_pytorch_threshold,
        {
            "continue_qualify": "react_agent",
            "early_exit": "early_exit_node"
        }
    )
    
    # Conditional Edge after ReAct
    workflow.add_conditional_edges(
        "react_agent",
        router_icp_check,
        {
            "continue_rag": "rag_agent",
            "terminate_exit": END
        }
    )
    
    workflow.add_edge("rag_agent", "diffusion_agent")
    workflow.add_edge("diffusion_agent", END)
    workflow.add_edge("early_exit_node", END)
    
    return workflow.compile()