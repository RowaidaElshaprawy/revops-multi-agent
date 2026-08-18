from typing import TypedDict, Optional, List, Dict, Any


class RevOpsState(TypedDict):
    domain: str
    scraped_text: Optional[str]
    intent_data: Optional[Dict[str, Any]]
    raw_features: Optional[List[float]]
    pytorch_score: Optional[float]
    icp_reasoning: Optional[Dict[str, Any]]
    is_qualified: Optional[bool]
    rag_context: Optional[List[str]]
    media_asset_info: Optional[Dict[str, Any]]
    blocked: Optional[bool]
    block_reason: Optional[str]
    current_step: str
    audit_logs: List[str]


def new_state(domain: str) -> RevOpsState:
    return RevOpsState(
        domain=domain, scraped_text=None, intent_data=None, raw_features=None,
        pytorch_score=None, icp_reasoning=None, is_qualified=None,
        rag_context=None, media_asset_info=None, blocked=None, block_reason=None,
        current_step="START", audit_logs=[],
    )