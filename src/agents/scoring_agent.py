import os

import torch
import torch.nn as nn

from src.agents.state import RevOpsState

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scoring_model.pt")
FEATURE_NAMES = ["pricing", "demo", "enterprise", "content_richness"]


class LeadScorer(nn.Module):
    def __init__(self, in_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _heuristic_weights(model: "LeadScorer") -> None:
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([
            [0.9, 0.3, 0.2, 0.1], [0.2, 0.8, 0.1, 0.1], [0.1, 0.1, 0.9, 0.2], [0.3, 0.3, 0.3, 0.6],
            [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.4, 0.2, 0.4, 0.2], [0.25, 0.25, 0.25, 0.25],
        ]))
        model.net[0].bias.zero_()
        model.net[2].weight.copy_(torch.tensor([[0.3, 0.25, 0.25, 0.05, 0.05, 0.05, 0.03, 0.02]]))
        model.net[2].bias.zero_()


_model = None
_using_trained_weights = False


def _load_model() -> LeadScorer:
    global _model, _using_trained_weights
    if _model is not None:
        return _model
    model = LeadScorer(in_features=len(FEATURE_NAMES))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _using_trained_weights = True
    else:
        _heuristic_weights(model)
        _using_trained_weights = False
    model.eval()
    _model = model
    return model


def scoring_node(state: RevOpsState) -> RevOpsState:
    features = state.get("raw_features") or [0.0] * len(FEATURE_NAMES)
    model = _load_model()
    with torch.no_grad():
        x = torch.tensor([features], dtype=torch.float32)
        score = float(model(x).item())

    logs = state.get("audit_logs", [])
    src = "trained model.pt" if _using_trained_weights else "heuristic weights (not yet trained)"
    logs.append(f"[scoring_agent] pytorch_score={score:.3f} features={features} weights={src}")
    state["pytorch_score"] = score
    state["current_step"] = "SCORED"
    state["audit_logs"] = logs
    return state