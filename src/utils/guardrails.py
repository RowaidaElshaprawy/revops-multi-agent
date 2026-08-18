import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore (all|previous|the) instructions",
    r"disregard (all|previous|the) (system|prior) prompt",
    r"act as (system|admin|root)",
    r"reveal (your|the) (system prompt|instructions)",
    r"drop (table|database)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_text(text: str) -> Tuple[bool, str]:
    if not text:
        return False, ""
    for pattern in _COMPILED:
        if pattern.search(text):
            return True, f"matched injection pattern: {pattern.pattern}"
    return False, ""


def guardrail_node(state):
    blocked, reason = check_text(state.get("domain", ""))
    logs = state.get("audit_logs", [])
    logs.append(f"[guardrails] {'BLOCKED: ' + reason if blocked else 'domain input clear'}")
    state["blocked"] = blocked
    state["block_reason"] = reason
    state["current_step"] = "GUARDRAIL_BLOCKED" if blocked else "GUARDRAIL_PASSED"
    state["audit_logs"] = logs
    return state