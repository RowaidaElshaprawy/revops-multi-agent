from langgraph.graph import StateGraph, END
from src.agents.state import RevOpsState
from src.agents.scraper_agent import ScraperIntentAgent
from src.agents.scoring_agent import PyTorchScoringAgent
from src.agents.icp_agent import ICPReasoningAgent
from src.agents.rag_agent import RAGKnowledgeAgent
from src.agents.media_agent import MediaGenAgent

# Initialize agents
scraper_agent = ScraperIntentAgent()
scoring_agent = PyTorchScoringAgent()
icp_agent = ICPReasoningAgent()
rag_agent = RAGKnowledgeAgent()
media_agent = MediaGenAgent()

# Existing Nodes
def node_scrape_and_intent(state: RevOpsState) -> RevOpsState:
    domain = state["domain"]
    scrape_res = scraper_agent.scrape_domain(domain)
    if scrape_res["success"]:
        intent_res = scraper_agent.classify_intent(scrape_res["raw_text"])
        state["scraped_text"] = scrape_res["raw_text"]
        state["intent_data"] = intent_res
        state["audit_logs"].append(f"[Agent 1] Scraped domain and detected intent: {intent_res['top_intent']}")
    else:
        state["scraped_text"] = "Scraping failed or limited content."
        state["intent_data"] = {"top_intent": "Unknown", "confidence": 0.0}
        state["audit_logs"].append("[Agent 1] Web scraping fallback executed.")
    state["current_step"] = "SCRAPED"
    return state

def node_pytorch_scoring(state: RevOpsState) -> RevOpsState:
    score = scoring_agent.score_lead(state["raw_features"])
    state["pytorch_score"] = score
    state["audit_logs"].append(f"[Agent 2] Computed PyTorch conversion score: {score}")
    state["current_step"] = "SCORED"
    return state

def node_icp_reasoning(state: RevOpsState) -> RevOpsState:
    eval_res = icp_agent.evaluate_qualitative_fit(state)
    state["icp_reasoning"] = eval_res["icp_reasoning"]
    state["is_qualified"] = eval_res["is_qualified"]
    state["audit_logs"].append(f"[Agent 3] Qualitative ReAct evaluation complete. Qualified: {eval_res['is_qualified']}")
    state["current_step"] = "EVALUATED"
    return state

# Phase 3 Nodes
def node_rag_retrieval(state: RevOpsState) -> RevOpsState:
    query = state.get("scraped_text", "") or state["domain"]
    context = rag_agent.retrieve_context(query)
    state["rag_context"] = context
    state["audit_logs"].append(f"[Agent 4] Retrieved {len(context)} ChromaDB battlecards/case studies.")
    state["current_step"] = "RAG_RETRIEVED"
    return state

def node_media_generation(state: RevOpsState) -> RevOpsState:
    domain = state["domain"]
    intent = state.get("intent_data", {}).get("top_intent", "Automation")
    asset_info = media_agent.generate_personalized_asset(domain, intent)
    state["media_asset_info"] = asset_info
    state["audit_logs"].append(f"[Agent 5] Generated Stable Diffusion prompt & visual asset spec.")
    state["current_step"] = "MEDIA_GENERATED"
    return state

# Conditional Edge Routers
def route_after_scoring(state: RevOpsState) -> str:
    if state.get("pytorch_score", 0.0) < 0.3:
        state["is_qualified"] = False
        state["audit_logs"].append("[Router] Lead score below threshold (< 0.3). Exiting.")
        return "disqualified"
    return "qualify_further"

def route_after_icp(state: RevOpsState) -> str:
    if state.get("is_qualified"):
        return "qualified"
    state["audit_logs"].append("[Router] Lead not ICP qualified. Exiting.")
    return "disqualified"

# Build Graph
def build_revops_graph():
    workflow = StateGraph(RevOpsState)

    # Add Nodes
    workflow.add_node("scrape_and_intent", node_scrape_and_intent)
    workflow.add_node("pytorch_scoring", node_pytorch_scoring)
    workflow.add_node("icp_reasoning", node_icp_reasoning)
    workflow.add_node("rag_retrieval", node_rag_retrieval)
    workflow.add_node("media_generation", node_media_generation)

    # Add Edges
    workflow.set_entry_point("scrape_and_intent")
    workflow.add_edge("scrape_and_intent", "pytorch_scoring")

    workflow.add_conditional_edges(
        "pytorch_scoring",
        route_after_scoring,
        {"qualify_further": "icp_reasoning", "disqualified": END}
    )

    workflow.add_conditional_edges(
        "icp_reasoning",
        route_after_icp,
        {"qualified": "rag_retrieval", "disqualified": END}
    )

    workflow.add_edge("rag_retrieval", "media_generation")
    workflow.add_edge("media_generation", END)

    return workflow.compile()