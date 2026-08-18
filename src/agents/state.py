from typing import TypedDict, Optional, List, Dict, Any


class RevOpsState(TypedDict):
    domain: str

    # يتملوا من scraper_agent.py — مش يدوي أبداً
    scraped_text: Optional[str]
    intent_data: Optional[Dict[str, Any]]
    raw_features: Optional[List[float]]

    # يتملوا من scoring_agent.py
    pytorch_score: Optional[float]

    # يتملوا من icp_agent.py
    icp_reasoning: Optional[Dict[str, Any]]
    is_qualified: Optional[bool]

    # يتملوا من rag_agent.py
    rag_context: Optional[List[str]]

    # يتملوا من media_agent.py
    media_asset_info: Optional[Dict[str, Any]]

    current_step: str
    audit_logs: List[str]


def new_state(domain: str) -> RevOpsState:
    return RevOpsState(
        domain=domain,
        scraped_text=None,
        intent_data=None,
        raw_features=None,
        pytorch_score=None,
        icp_reasoning=None,
        is_qualified=None,
        rag_context=None,
        media_asset_info=None,
        current_step="START",
        audit_logs=[],
    )