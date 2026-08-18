import os

from langgraph.graph import StateGraph, END

from src.agents.state import RevOpsState, new_state
from src.agents.scraper_agent import scraper_node
from src.agents.scoring_agent import scoring_node
from src.agents.icp_agent import icp_node
from src.agents.rag_agent import rag_node
from src.agents.media_agent import media_node
from src.utils.guardrails import guardrail_node

EARLY_EXIT_THRESHOLD = 0.3


def guardrail_router(state: RevOpsState) -> str:
    return "blocked" if state.get("blocked") else "continue"


def score_router(state: RevOpsState) -> str:
    return "qualify_path" if (state.get("pytorch_score") or 0.0) >= EARLY_EXIT_THRESHOLD else "early_exit"


def early_exit_node(state: RevOpsState) -> RevOpsState:
    logs = state.get("audit_logs", [])
    logs.append(f"[supervisor] early exit — score below {EARLY_EXIT_THRESHOLD}")
    state["is_qualified"] = False
    state["current_step"] = "EARLY_EXIT"
    state["audit_logs"] = logs
    return state


def blocked_node(state: RevOpsState) -> RevOpsState:
    state["current_step"] = "BLOCKED"
    return state


def build_revops_graph():
    graph = StateGraph(RevOpsState)
    graph.add_node("guardrails", guardrail_node)
    graph.add_node("scraper", scraper_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("icp", icp_node)
    graph.add_node("rag", rag_node)
    graph.add_node("media", media_node)
    graph.add_node("early_exit", early_exit_node)
    graph.add_node("blocked", blocked_node)

    graph.set_entry_point("guardrails")
    graph.add_conditional_edges("guardrails", guardrail_router, {"continue": "scraper", "blocked": "blocked"})
    graph.add_edge("blocked", END)
    graph.add_edge("scraper", "scoring")
    graph.add_conditional_edges("scoring", score_router, {"qualify_path": "icp", "early_exit": "early_exit"})
    graph.add_edge("early_exit", END)
    graph.add_edge("icp", "rag")
    graph.add_edge("rag", "media")
    graph.add_edge("media", END)
    return graph.compile()


def run_qualification(domain: str) -> RevOpsState:
    graph = build_revops_graph()
    return graph.invoke(new_state(domain))