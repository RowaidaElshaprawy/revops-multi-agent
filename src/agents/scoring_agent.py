import torch
import torch.nn as nn
import numpy as np

# 1. Define custom PyTorch Multi-Layer Perceptron Model (Module 2)
class LeadScoringMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(LeadScoringMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output probability score between 0.0 and 1.0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PyTorchScoringAgent:
    def __init__(self):
        # Features: [employee_count_normalized, revenue_m_normalized, tech_stack_match_count, email_opened_count]
        self.input_dim = 4
        self.model = LeadScoringMLP(input_dim=self.input_dim)
        self.model.eval()  # Set to evaluation mode

    def score_lead(self, raw_features: list) -> float:
        """
        Converts tabular lead features into PyTorch Tensor and runs forward pass.
        raw_features example: [150/1000, 10/100, 3/5, 2/10]
        """
        tensor_features = torch.tensor([raw_features], dtype=torch.float32)
        with torch.no_grad():
            score = self.model(tensor_features).item()
        return round(score, 4)