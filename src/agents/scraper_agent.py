import re
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from src.agents.state import RevOpsState

USER_AGENT = "RevOpsQualificationBot/1.0 (+contact: revops@example.com)"

PRICING_KEYWORDS = ["pricing", "plans", "subscription", "per month", "per user", "get a quote"]
DEMO_KEYWORDS = ["book a demo", "request a demo", "free trial", "start free", "schedule a call"]
ENTERPRISE_KEYWORDS = ["enterprise", "sso", "soc 2", "compliance", "dedicated support", "sla", "security"]


def fetch_page_text(domain: str, timeout: int = 8) -> str:
    url = domain if domain.startswith("http") else f"https://{domain}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _keyword_density(text: str, keywords: List[str], cap_hits: int = 5) -> float:
    text_lower = text.lower()
    hits = sum(text_lower.count(k) for k in keywords)
    return min(hits / cap_hits, 1.0)


def extract_signals(text: str) -> Dict[str, float]:
    return {
        "pricing": _keyword_density(text, PRICING_KEYWORDS),
        "demo": _keyword_density(text, DEMO_KEYWORDS),
        "enterprise": _keyword_density(text, ENTERPRISE_KEYWORDS),
        "content_richness": min(len(text) / 5000.0, 1.0),
    }


def scraper_node(state: RevOpsState) -> RevOpsState:
    domain = state["domain"]
    logs = state.get("audit_logs", [])
    try:
        text = fetch_page_text(domain)
    except requests.RequestException as e:
        logs.append(f"[scraper_agent] FAILED to fetch {domain}: {e}")
        state["scraped_text"] = ""
        state["intent_data"] = {"top_intent": "unreachable", "signals": {}}
        state["raw_features"] = [0.0, 0.0, 0.0, 0.0]
        state["current_step"] = "SCRAPER_FAILED"
        state["audit_logs"] = logs
        return state

    signals = extract_signals(text)
    top_intent = max(signals, key=signals.get)

    state["scraped_text"] = text[:4000]
    state["intent_data"] = {"top_intent": top_intent, "signals": signals}
    state["raw_features"] = [signals["pricing"], signals["demo"], signals["enterprise"], signals["content_richness"]]
    logs.append(f"[scraper_agent] fetched {len(text)} chars from {domain} — top_intent={top_intent}")
    state["current_step"] = "SCRAPED"
    state["audit_logs"] = logs
    return state