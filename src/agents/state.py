from typing import TypedDict, Optional, List, Dict, Any

class RevOpsState(TypedDict):
    domain: str
    raw_features: List[float]
    scraped_text: Optional[str]
    intent_data: Optional[Dict[str, Any]]
    pytorch_score: Optional[float]
    icp_reasoning: Optional[str]
    is_qualified: Optional[bool]
    rag_context: Optional[List[str]]          # Added in Phase 3
    media_asset_info: Optional[Dict[str, Any]] # Added in Phase 3
    current_step: str
    audit_logs: List[str]