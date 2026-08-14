import re
import mlflow


class SafetyAndEvalGuard:
    def __init__(self):
        # Initialize MLflow experiment tracking (Module 7)
        mlflow.set_experiment("RevOps_Multi_Agent_Pipeline")

    def check_prompt_injection(self, text: str) -> bool:
        """Simple rule-based prompt injection and adversarial safety check."""
        suspicious_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"drop database",
            r"jailbreak"
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True  # Malicious input detected
        return False

    def log_agent_run(self, state: dict):
        """Logs metrics, scores, and execution parameters to MLflow (Module 7)."""
        with mlflow.start_run(run_name=f"Lead_{state['domain']}"):
            mlflow.log_param("domain", state["domain"])
            
            intent_data = state.get("intent_data") or {}
            mlflow.log_param("detected_intent", intent_data.get("top_intent", "Unknown"))
            
            if state.get("pytorch_score") is not None:
                mlflow.log_metric("pytorch_conversion_score", state["pytorch_score"])
            
            mlflow.log_metric("is_qualified", 1.0 if state.get("is_qualified") else 0.0)
            
            # Simulated RAGAS metrics (Faithfulness & Context Precision)
            if state.get("rag_context"):
                mlflow.log_metric("ragas_context_precision", 0.92)
                mlflow.log_metric("ragas_faithfulness", 0.95)
