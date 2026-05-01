import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DualHeadLayerRouter(nn.Module):
    """
    Phase 2: Multi-task Learning Router for Knowledge Distillation
    Heads:
    1. Introspection Head: Predicts ColBERT margin (Self-correction)
    2. Distillation Head: Predicts Mistral E5 semantic margin (Knowledge Distillation)
    """
    def __init__(self, 
                 num_layers: int = 5, 
                 embed_dim: int = 128, 
                 hidden_dims: List[int] = [256],
                 fusion_type: str = "interaction"):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        # 1. Feature Fusion Layer
        if fusion_type == "concat":
            input_dim = num_layers * embed_dim * 2
        elif fusion_type == "interaction":
            input_dim = num_layers * embed_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # 2. Shared Backbone
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        
        # 3. Dual Heads
        # Head 1: ColBERT Introspection (Binary classification or Margin regression)
        self.introspection_head = nn.Linear(curr_dim, 1)
        
        # Head 2: Mistral E5 Distillation (Margin regression)
        self.distillation_head = nn.Linear(curr_dim, 1)
        
    def forward(self, q_reps: torch.Tensor, d_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input shapes: (B, L, D)
        Returns: (introspection_logits, distillation_logits)
        """
        B, L, D = q_reps.shape
        
        # Fusion
        if self.fusion_type == "concat":
            x = torch.cat([q_reps, d_reps], dim=-1) # (B, L, 2D)
            x = x.view(B, -1) # (B, L*2D)
        elif self.fusion_type == "interaction":
            x = q_reps * d_reps # (B, L, D) - Hadamard product
            x = x.view(B, -1) # (B, L*D)
            
        # Shared representations
        shared_feat = self.backbone(x)
        
        # Multi-task outputs
        introspection_out = self.introspection_head(shared_feat).squeeze(-1)
        distillation_out = self.distillation_head(shared_feat).squeeze(-1)
        
        return introspection_out, distillation_out
