import os
from typing import Dict, Any

from src.agents.state import RevOpsState

QUALIFY_SCORE_THRESHOLD = 0.5


def _rule_based_reasoning(signals: Dict[str, float], pytorch_score: float) -> Dict[str, Any]:
    reasons_for, reasons_against = [], []
    if signals.get("pricing", 0) > 0.3:
        reasons_for.append("actively evaluating pricing")
    else:
        reasons_against.append("no visible pricing-evaluation intent")
    if signals.get("demo", 0) > 0.3:
        reasons_for.append("has a clear trial/demo conversion path")
    else:
        reasons_against.append("no obvious trial/demo funnel")
    if signals.get("enterprise", 0) > 0.3:
        reasons_for.append("shows enterprise-readiness")

    fit = pytorch_score >= QUALIFY_SCORE_THRESHOLD and len(reasons_for) >= 1
    return {
        "method": "rule_based",
        "steps": [f"score={pytorch_score:.2f}", f"for={reasons_for}", f"against={reasons_against}"],
        "reasons_for": reasons_for, "reasons_against": reasons_against, "is_qualified": fit,
    }


def _llm_reasoning(domain, scraped_text, signals, pytorch_score) -> Dict[str, Any]:
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"You are a RevOps analyst. Domain: {domain}\n"
            f"Scraped homepage excerpt: {scraped_text[:1500]}\nSignals: {signals}\n"
            f"Score: {pytorch_score:.2f}\nReason step by step, end with: QUALIFIED: yes/no."
        )
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=500,
                                       messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return {"method": "llm", "steps": [text], "reasons_for": [], "reasons_against": [],
                "is_qualified": "qualified: yes" in text.lower()}
    except Exception:
        return _rule_based_reasoning(signals, pytorch_score)


def icp_node(state: RevOpsState) -> RevOpsState:
    signals = (state.get("intent_data") or {}).get("signals", {})
    pytorch_score = state.get("pytorch_score") or 0.0
    scraped_text = state.get("scraped_text") or ""

    if os.environ.get("ANTHROPIC_API_KEY"):
        reasoning = _llm_reasoning(state["domain"], scraped_text, signals, pytorch_score)
    else:
        reasoning = _rule_based_reasoning(signals, pytorch_score)

    logs = state.get("audit_logs", [])
    logs.append(f"[icp_agent] method={reasoning['method']} is_qualified={reasoning['is_qualified']}")
    state["icp_reasoning"] = reasoning
    state["is_qualified"] = reasoning["is_qualified"]
    state["current_step"] = "ICP_EVALUATED"
    state["audit_logs"] = logs
    return state