from typing import Dict, Any, Optional
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ICPReasoningAgent:
    def __init__(self, llm: Optional[Any] = None):
        """
        Initializes the ICP Reasoning Agent.
        :param llm: Optional LangChain LLM instance (e.g. HuggingFacePipeline with QLoRA model).
        """
        self.llm = llm
        
        # Prompt engineering using ReAct framework (Module 4)
        self.react_template = """
You are an expert SDR & RevOps Analyst evaluating an inbound prospect company.

[PROSPECT DATA]
Company Domain: {domain}
Scraped Website Content: {scraped_text}
Inferred Intent: {intent}
PyTorch Quantitative Score: {pytorch_score}

[INSTRUCTIONS]
Follow a ReAct (Reason + Act) approach:
1. Thought: Analyze the business model, scraped content, and quantitative score.
2. Reasoning: Does this company fit our B2B SaaS Ideal Customer Profile (ICP)?
3. Final Decision: Output 'QUALIFIED' or 'DISQUALIFIED' followed by a short 2-sentence rationale.

Perform your reasoning below:
"""
        self.prompt = PromptTemplate.from_template(self.react_template)
        
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()

    def evaluate_qualitative_fit(self, state_data: dict) -> Dict[str, Any]:
        """
        Evaluates qualitative ICP fit using ReAct style reasoning over RevOpsState context.
        Returns dictionary updates for RevOpsState.
        """
        domain = state_data.get("domain", "Unknown")
        scraped_text = state_data.get("scraped_text", "") or "No text available."
        intent_data = state_data.get("intent_data") or {}
        intent = intent_data.get("top_intent", "Unknown")
        score = state_data.get("pytorch_score", 0.0)

        if self.llm:
            # Execution using real LLM / HuggingFace Pipeline
            reasoning_str = self.chain.invoke({
                "domain": domain,
                "scraped_text": scraped_text[:1000],  # Truncate text to fit context window
                "intent": intent,
                "pytorch_score": score
            })
        else:
            # Fallback mock for demonstration/testing without GPU
            reasoning_str = (
                f"Thought: Domain {domain} demonstrates high commercial intent ({intent}) "
                f"with a strong score of {score}.\n"
                f"Reasoning: Their business model aligns with enterprise automation software.\n"
                f"Final Decision: QUALIFIED - Strong technical match and business scale."
            )

        is_qualified = "QUALIFIED" in reasoning_str and "DISQUALIFIED" not in reasoning_str
        
        return {
            "icp_reasoning": reasoning_str,
            "is_qualified": is_qualified
        }


def icp_agent_node(state: dict) -> dict:
    """
    LangGraph node wrapper for RevOpsState integration.
    """
    agent = ICPReasoningAgent()
    result = agent.evaluate_qualitative_fit(state)
    
    audit_logs = list(state.get("audit_logs", []))
    audit_logs.append(f"[ICP Agent] Evaluated qualitative fit: Qualified={result['is_qualified']}")
    
    return {
        "icp_reasoning": result["icp_reasoning"],
        "is_qualified": result["is_qualified"],
        "current_step": "icp_reasoning",
        "audit_logs": audit_logs
    }
