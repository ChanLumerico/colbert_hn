import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerRouter(nn.Module):
    def __init__(self, 
                 num_layers: int = 5, 
                 embed_dim: int = 128, 
                 hidden_dims: list = [256], 
                 dropout: float = 0.1,
                 fusion_type: str = "concat"):
        """
        Flexible Router supporting Phase 1+ Ablation Studies.
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        # Calculate input dimension based on fusion strategy [Ablation 2]
        feature_dim = num_layers * embed_dim
        if fusion_type == "concat":
            input_dim = feature_dim * 2  # [Q; D]
        elif fusion_type == "diff":
            input_dim = feature_dim  # [Q - D]
        elif fusion_type == "interaction":
            input_dim = feature_dim * 4  # [Q; D; Q-D; Q*D]
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # Build MLP layers [Ablation 3]
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, q_reps, d_reps):
        """
        Args:
            q_reps: (batch, num_layers, embed_dim)
            d_reps: (batch, num_layers, embed_dim)
        """
        # Flatten layer representations
        batch_size = q_reps.size(0)
        q_flat = q_reps.reshape(batch_size, -1)
        d_flat = d_reps.reshape(batch_size, -1)
        
        # Fusion logic [Ablation 2]
        if self.fusion_type == "concat":
            x = torch.cat([q_flat, d_flat], dim=-1)
        elif self.fusion_type == "diff":
            x = q_flat - d_flat
        elif self.fusion_type == "interaction":
            diff = q_flat - d_flat
            prod = q_flat * d_flat
            x = torch.cat([q_flat, d_flat, diff, prod], dim=-1)
        else:
             x = torch.cat([q_flat, d_flat], dim=-1)
            
        logits = self.mlp(x).squeeze(-1)
        return logits
