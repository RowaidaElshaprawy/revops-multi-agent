from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
import mlflow

class ProductionEvaluator:
    """Evaluates RAG and Reasoning quality using RAGAS and logs metrics to MLflow."""
    
    def __init__(self, experiment_name: str = "RevOps_LLM_Evals"):
        mlflow.set_experiment(experiment_name)

    def evaluate_run(self, question: str, contexts: list[str], answer: str, ground_truth: str) -> dict:
        """Runs RAGAS metric calculations and records scores in MLflow."""
        if not contexts:
            contexts = ["No retrieved context provided."]
            
        data = {
            "question": [question],
            "contexts": [contexts],
            "answer": [answer],
            "ground_truth": [ground_truth]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Calculate quantitative metrics
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )
        
        scores = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"])
        }
        
        # Log to MLflow experiment tracking
        with mlflow.start_run(nested=True):
            for metric_name, value in scores.items():
                mlflow.log_metric(f"ragas_{metric_name}", value)
                
        return scores